# BrainO engine

The Python analysis engine behind BrainO: data import, EEG/fMRI analysis, statistics and the local
AI analysis agent. It can be used on its own from Python or the command line.

> Status: alpha (v0.1). Documentation: <https://kmnuddin.github.io/BrainO_v2/>. Roadmap:
> [`docs/PLAN.md`](https://github.com/kmnuddin/BrainO_v2/blob/main/docs/PLAN.md).

## Install (development)

```bash
cd engine
python -m venv .venv && source .venv/bin/activate   # or: uv venv && source .venv/bin/activate
pip install -e ".[dev,eeg]"
```

Python 3.11 or newer is required. The `eeg` extra installs the scientific stack (MNE-Python,
MNE-BIDS, NumPy and extra file readers); without it, the core (tools, provenance, CLI) still
works.

## Command line

```bash
# Import recordings (any supported format) into a de-identified BIDS dataset. The key file links
# subject labels to real IDs: keep it private and outside the dataset.
braino import raw/patient_a.edf --subject-id "patient A" --task rest --dataset study_bids --key private/key.json
braino import --mapping recordings.csv --dataset study_bids --key private/key.json
#   recordings.csv columns: source,subject_id,task[,session,run,line_freq]

braino inspect --dataset study_bids             # subjects, tasks, channels, events per recording
braino qc --dataset study_bids                  # flat/noisy channels, line noise, artifacts
braino qc --dataset study_bids --subject 001 --json
```

```bash
braino version
braino tools list                 # registered analysis tools and their risk level
braino tools schema system.info   # JSON schema the AI agent sees for a tool
braino tools run system.info      # run a tool, print the result as JSON
braino tools run <tool> --args '{"key": "value"}'
braino tools run <tool> --dataset path/to/bids   # for tools that read a dataset or write outputs
braino runs list --dataset path/to/bids          # recorded runs and the tools they called
braino runs show <run-id> --dataset path/to/bids # full provenance record of a run, as JSON
braino proposals list --pending --dataset path/to/bids        # decisions waiting for you
braino proposals approve <id> --dataset path/to/bids [--edit '{"channels": ["T7"]}'] [--note ...]
braino proposals reject <id> --dataset path/to/bids [--note ...]
braino runs export <run-id> --dataset path/to/bids -o analysis.py   # a script that re-runs it
```

```bash
braino pipeline list                                   # built-in pipeline specs
braino pipeline show erpcore-p3 > p3.yaml              # copy one to adapt
braino pipeline validate p3.yaml                       # tools exist, arguments fit
braino pipeline run p3.yaml --dataset path/to/bids     # cached; stops at decisions
braino pipeline run p3.yaml --dataset path/to/bids --from epochs   # redo from a step
braino pipeline status p3.yaml --dataset path/to/bids
braino pipeline export p3.yaml --dataset path/to/bids -o p3.py
braino validate erpcore P3 [--subjects 001,002,003]    # reproduce the published P3
braino eval list                                       # agent evaluation tasks
braino eval run --agent oracle -o report.json          # score an agent (see the docs)
```

```bash
braino model hardware          # CPU, RAM, GPUs, and the model tier BrainO proposes
braino model setup             # install llama.cpp, download the model once, benchmark it
braino model status            # chosen model, installed files, measured speed
braino model serve             # run the model server in the foreground (development)
```

## Tools

Every analysis step is a **tool**: a function that takes one Pydantic model and returns another.
The same tool is used by the CLI, the desktop app and the AI agent.

```python
from pydantic import BaseModel, Field

from braino.tools import Risk, tool


class BandPowerInput(BaseModel):
    recording: str = Field(description="Path to a BIDS EEG recording")
    band: tuple[float, float] = Field(description="Frequency band in Hz, e.g. [8, 12]")


class BandPowerOutput(BaseModel):
    power_per_channel: dict[str, float]


@tool(name="eeg.band_power", risk=Risk.COMPUTE)
def band_power(params: BandPowerInput) -> BandPowerOutput:
    """Compute mean power in a frequency band for each channel."""
    ...
```

- The **docstring** and the **field descriptions** are what the language model reads, so write
  them for a reader who has never seen the code.
- **Risk levels:** `read` (inspect only), `compute` (produces new results), `decision` (changes
  which data or statistical model the results rest on, e.g. dropping channels; the agent must
  ask the user before running these).

### Run context

A tool that reads a dataset or writes outputs takes a second parameter, a `RunContext`. It says
which BIDS dataset the run works on, gives the run an ID, and hands out output folders under
`<dataset>/derivatives/braino/` (a BIDS-Derivatives dataset). The caller supplies the context;
it is not part of the tool's input schema, so the AI agent cannot choose where data is read from
or written to.

```python
from pathlib import Path

from braino.tools import RunContext, registry


@tool(name="eeg.band_power", risk=Risk.COMPUTE)
def band_power(params: BandPowerInput, ctx: RunContext) -> BandPowerOutput:
    """Compute mean power in a frequency band for each channel."""
    raw = ctx.record_input(ctx.dataset / params.recording)  # hashed for provenance
    out = ctx.record_output(ctx.output_dir("sub-01", "eeg") / "band_power.json")
    ...


ctx = RunContext(dataset_root=Path("path/to/bids"))
registry.get("eeg.band_power").run({"recording": "...", "band": [8, 12]}, ctx)
```

### Provenance

Every tool call run with a context is recorded: tool, risk level, validated arguments, result,
success or error, timing, and the SHA-256 hash of each file declared with `record_input` /
`record_output`. A run's records are written to `derivatives/braino/runs/<run-id>/`:
`run.json` holds the environment (BrainO, Python, OS and analysis-library versions) and
`calls.jsonl` holds one line per tool call. Runs without a dataset keep their records in memory
(`ctx.provenance.calls`). Tools must not call other tools; chaining is the job of the pipeline
runner and the agent.

### Decision tools

A `decision` tool works in two steps, so that nothing the results rest on changes without a
person's approval:

1. **Propose.** Running the tool computes a proposal (e.g. channels to drop), saves it to
   `derivatives/braino/proposals/<id>.json` and returns a `PendingProposal`. Nothing changes yet.
2. **Decide.** A person approves (optionally editing fields), or rejects, with
   `braino proposals approve|reject` or `braino.tools.decide(...)`. On approval the tool's `apply`
   function commits the final version. The proposal file keeps the original, the final version,
   who decided, when, and why.

```python
def apply_bad_channels(proposal: BadChannels, ctx: RunContext) -> Applied:
    """Mark the approved channels as bad in the derivatives."""
    ...


@tool(name="eeg.detect_bad_channels", risk=Risk.DECISION, apply=apply_bad_channels)
def detect_bad_channels(params: DetectInput, ctx: RunContext) -> BadChannels:
    """Find noisy or flat channels and propose them for removal."""
    ...
```

The AI agent can run decision tools, which only creates proposals; approving is never one of its
tools.

## Importing recordings

`braino.io.import_recording` turns one recording, in any supported format, into a de-identified
BIDS dataset entry: it detects the format, reads it, normalises channel names, types, positions
and events, removes personal details, and writes new BIDS files (the originals are only read).

```python
from braino.io import SubjectKey, import_recording

# The subject key links labels (sub-001, ...) to real identities. Keep it private, and outside
# the dataset folder; BrainO refuses to put it inside.
key = SubjectKey.open("private/subjects.json", dataset_root="study_bids")
report = import_recording(
    "raw/patient_a.edf", "study_bids", key=key, original_id="patient A", task="rest"
)
print(report.conversion.files)  # what was written, relative to the dataset
print(report.deidentification.free_text_events)  # event text to check by hand
```

Each step is also available on its own: `detect_format`, `read_raw`, `normalise`, `deidentify`
and `write_bids`.

## Analysing EEG

The analysis tools work on one recording at a time, then on the group. Each writes new files to
`derivatives/braino/` (nothing is changed in place) and the next one reads them:

| Step | Tool | Writes |
|---|---|---|
| Filter, resample, re-reference | `eeg.preprocess` | `*_desc-preproc_eeg.fif` |
| Remove eye/heart components (decision) | `eeg.ica_clean` | `*_desc-preproc_ica.fif`, then `*_desc-clean_eeg.fif` |
| Mark bad channels (decision) | `eeg.detect_bad_channels` | `*_desc-preproc_channels.tsv` |
| Epoch, reject or repair artifacts | `eeg.epoch` | `*_desc-<label>_epo.fif` |
| Averages, difference waves, amplitudes | `eeg.erp` | `*_ave.fif`, `*_erp.tsv` |
| Spectra and band power | `eeg.spectral` | `*_psd.npz`, `*_bands.tsv` |
| Time-frequency power | `eeg.tfr` | `*_tfr.npz` |
| Group tests (FDR/Bonferroni/max-stat) | `stats.mass_univariate` | `group/task-<task>/*_stats.tsv` |
| Cluster permutation tests | `stats.cluster_permutation` | `group/task-<task>/*_clusters.tsv` |

ICA before bad-channel detection works best: eye activity then no longer makes the channels near
the eyes look bad (they are needed to remove it), and a broken channel shows up in ICA as a
"channel noise" component. Approved bad channels are interpolated when epoching. If an earlier
step is re-run, decisions made on its old output are refused until they are made again.

```python
from braino.tools import RunContext, decide, registry

ctx = RunContext(dataset_root="study_bids")
run = lambda tool, **args: registry.get(tool).run({"subject": "001", "task": "P3", **args}, ctx)

run("eeg.preprocess", l_freq=0.1, h_freq=30, resample_hz=256, reference=["P9", "P10"])
proposal = run("eeg.ica_clean")  # proposes components to remove
decide(proposal.proposal_id, ctx, approve=True, decided_by="user:alice")
proposal = run("eeg.detect_bad_channels")  # proposes channels to mark bad
decide(proposal.proposal_id, ctx, approve=True, decided_by="user:alice")
run("eeg.epoch", conditions={"rare": ["stimulus/11"], "frequent": ["stimulus/12"]})
run(
    "eeg.erp",
    contrasts=[{"name": "difference", "plus": "rare", "minus": "frequent"}],
    measures=[{"name": "P3", "channels": ["Pz"], "tmin": 0.3, "tmax": 0.6}],
)

registry.get("stats.cluster_permutation").run({"task": "P3", "conditions": ["difference"]}, ctx)
```

From the command line, the same tools run with `braino tools run <tool> --args '{...}' --dataset
study_bids`, and decisions with `braino proposals approve|reject`.

## Validation against published results

BrainO's analysis pipeline is checked against ERP CORE (Kappenman et al., 2021, *NeuroImage*
225:117465; CC-BY-4.0): it must reproduce the published P3 and N170 effects, participant by
participant.

```python
from braino.datasets import erpcore
from braino.validation.erpcore import compare

root = erpcore.fetch("P3", ["001", "002"])  # BIDS data, hash-checked, cached in ~/.cache/braino
erpcore.references()["P3"]  # what the authors measured and found

scores = {"001": 6.1, "002": 17.0, "003": 9.5}  # BrainO's per-participant P3 amplitudes, µV
report = compare("P3", scores)  # group and participant-level checks
```

`braino validate erpcore P3` (or `braino.validation.erpcore.run("P3")`) does all of it: downloads
the data, runs the built-in `erpcore-p3` pipeline with its decisions auto-approved, excludes
participants with more than 25% of trials rejected (ERP CORE's rule) and compares.

The full download is large (P3 2.2 GB, N170 3.4 GB), so it is not part of CI; the test that
downloads one participant runs with `BRAINO_LARGE_DOWNLOADS=1`.

## Test data

Tests that need real recordings download small public sample files (one recording per supported
format, about 33 MB in total) from MNE-Python's test data repository, check their SHA-256
hashes, and cache them. They are marked `network`.

- Cache folder: the OS cache folder, or `BRAINO_TEST_DATA=/some/path`.
- Offline: `BRAINO_SKIP_NETWORK=1 pytest` skips the download tests.
- In tests: `sample_data.fetch_recording("brainvision")` returns the path to open, and the
  `synthetic_raw` fixture gives a small deterministic MNE recording that needs no download.
- Adding a file: see the docstring of [`tests/sample_data.py`](https://github.com/kmnuddin/BrainO_v2/blob/main/engine/tests/sample_data.py).

## Checks

Run these before committing; CI runs the same commands.

```bash
ruff check . && ruff format --check . && mypy && pytest
```
