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
```

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
They are also the first of the validated templates that the AI agent will fill in for common
analyses.

## From Python

```python
from braino.pipelines import load_spec, run_pipeline
from braino.tools import RunContext

report = run_pipeline(load_spec("p3.yaml"), RunContext(dataset_root="study_bids"))
print(report.count("ran"), report.count("cached"), [o.proposal_id for o in report.pending])
```
