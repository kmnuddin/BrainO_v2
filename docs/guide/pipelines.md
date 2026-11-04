# Pipelines

A pipeline is an analysis written down: the steps to run on each subject's recording, then the
group steps, in a YAML (or JSON) file.

```yaml
name: p3-oddball
description: P3 to rare vs frequent letters
task: P3
steps:
  - id: preprocess
    tool: eeg.preprocess
    args: {l_freq: 0.1, h_freq: 30, resample_hz: 256, reference: [P9, P10]}
  - id: ica
    tool: eeg.ica_clean
  - id: bad_channels
    tool: eeg.detect_bad_channels
  - id: epochs
    tool: eeg.epoch
    args:
      conditions: {rare: [stimulus/11, stimulus/22], frequent: [stimulus/12, stimulus/21]}
      baseline: [-0.2, 0.0]
  - id: erp
    tool: eeg.erp
    args:
      contrasts: [{name: difference, plus: rare, minus: frequent}]
      measures: [{name: P3, channels: [Pz], tmin: 0.3, tmax: 0.6}]
group:
  - id: clusters
    tool: stats.cluster_permutation
    args: {conditions: [difference]}
```

The runner fills in `subject`, `task` and `session` for each step, and the group steps' list of
subjects. Optional fields: `session`, `run`, `subjects` (default: everyone with the task) and
`exclude_subjects`. Every step's arguments are checked against its tool when the spec is loaded:

```bash
braino pipeline validate p3.yaml
braino pipeline validate p3.yaml --dataset study_bids   # also against the data
```

## Plan checks

A spec whose every step is valid can still be a wrong plan. Before a pipeline runs, its steps
are checked together, as a reviewer would:

| Rule | Example of what it catches |
|---|---|
| Each step's input is made by an earlier step | `eeg.erp` with no `eeg.epoch` before it; a group test of ERPs with no `eeg.erp` step |
| Cleaning comes before epoching | `eeg.ica_clean` after `eeg.epoch` (the epochs would not use it) |
| Names are defined | a contrast of condition `target` when the epochs define `rare` and `frequent`; an epochs label no step made |
| Windows fit the epochs | a measure at 300–1200 ms in epochs that end at 800 ms; a baseline outside the epoch |
| Group tests are sound | fewer than 3 subjects; no multiple-comparison correction (warning); a cluster test with too few subjects for its permutations (warning) |
| The plan fits the data (with `--dataset`) | subjects without the task; a condition whose events never occur (the events that do are listed); channels not in the recordings; a low-pass at or above half the sampling rate |

Errors stop the plan: `braino pipeline run` refuses to start and prints them. Warnings are
printed and the run goes on.

## Templates

For common analyses, a template writes the spec from a few settings: the two conditions and
their events, and where and when to measure. Everything else keeps the steps and defaults
BrainO validated on ERP CORE.

| Template | What it does |
|---|---|
| `erp-comparison` | preprocess (0.1–30 Hz), epoch both conditions, average, measure the difference wave (mean or peak) in a window at some channels, and, with 3 subjects or more, test it across subjects (mass-univariate with FDR by default, or a cluster test), at the measured channels and window or, with `"test_scope": "all"`, at every channel and time point |
| `spectral-comparison` | preprocess (1–45 Hz), epoch both conditions, band power (alpha by default) per condition, and a corrected group test of the spectra in the band |

Both can add ICA (`"ica": true`) and bad-channel detection (`"bad_channels": true`), whose
proposals you review for each subject.

```bash
braino pipeline templates      # the templates and their settings
braino pipeline template erp-comparison --dataset study_bids -o p3.yaml --settings \
  '{"condition_a": "rare", "events_a": ["stimulus/rare"], "condition_b": "frequent",
    "events_b": ["stimulus/frequent"], "channels": ["Pz"], "tmin": 0.3, "tmax": 0.6}'
braino pipeline run p3.yaml --dataset study_bids
```

The spec it writes is an ordinary one, checked against the dataset: read it, change it, run
it. The [AI agent](agent.md#comparing-conditions) uses the same templates.

## Running

```bash
braino pipeline run p3.yaml --dataset study_bids
braino pipeline status p3.yaml --dataset study_bids
```

- **Results are reused.** A step is skipped when its tool, its arguments and the BrainO version
  are the same as last time and every file it read and wrote still has the same hash. Change a
  step's settings and that step and the ones after it run again; nothing else does.
- **Re-run from a step:** `--from epochs` runs `epochs` and everything after it again.
- **Decisions wait for you.** At a decision step the subject stops; the command exits with code
  3 and lists what waits. Approve or reject with `braino proposals`, then run again: the
  pipeline continues from there.
- **Failures stay contained.** A step that fails stops that subject only; the group steps wait
  until every subject has finished.
- `--auto-approve` approves every proposal as made, in your name, and records it as
  auto-approved. Use it for validation runs, not for deciding about data you have not seen.

The state (which tool call produced each current result) is kept in
`derivatives/braino/pipelines/<name>/state.json`; [script export](reproducibility.md) reads it.

## Built-in pipelines

```bash
braino pipeline list
braino pipeline show erpcore-p3 > my_p3.yaml   # copy and adapt
```

`erpcore-p3` and `erpcore-n170` reproduce the ERP CORE analyses ([validation](validation.md)).

## From Python

```python
from pathlib import Path

from braino.pipelines.checks import check_plan
from braino.pipelines.templates import build

spec = build("erp-comparison", {"condition_a": "rare", "events_a": ["stimulus/rare"],
                                "condition_b": "frequent", "events_b": ["stimulus/frequent"],
                                "channels": ["Pz"], "tmin": 0.3, "tmax": 0.6}, Path("study_bids"))
for finding in check_plan(spec, dataset=Path("study_bids")).findings:
    print(finding)
```

Running one:

```python
from braino.pipelines import load_spec, run_pipeline
from braino.tools import RunContext

report = run_pipeline(load_spec("p3.yaml"), RunContext(dataset_root="study_bids"))
print(report.count("ran"), report.count("cached"), [o.proposal_id for o in report.pending])
```
