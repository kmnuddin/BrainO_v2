# BrainO engine

The Python analysis engine behind BrainO: data import, EEG/fMRI analysis, statistics and the local
AI analysis agent. It can be used on its own from Python or the command line.

> Status: pre-alpha (milestone M1.0). See [`docs/PLAN.md`](../docs/PLAN.md) for the roadmap.

## Install (development)

```bash
cd engine
python -m venv .venv && source .venv/bin/activate   # or: uv venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Command line

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

## Checks

Run these before committing; CI runs the same commands.

```bash
ruff check . && ruff format --check . && mypy && pytest
```
