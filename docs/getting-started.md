# Getting started

This page analyses one participant of ERP CORE, a public dataset, from download to a P3
amplitude, first from the command line and then from Python.

## Install

```bash
pip install "braino[eeg]"
braino version
```

## Get some data

ERP CORE's P3 task is a visual oddball: letters A-E, one of them the target in each block.
BrainO can download it, checking every file against a pinned SHA-256 hash:

```bash
python -c "from braino.datasets import erpcore; print(erpcore.fetch('P3', ['001']))"
```

The printed folder (by default under `~/.cache/braino/erpcore/P3`) is a BIDS dataset. Look at
it:

```bash
braino inspect --dataset ~/.cache/braino/erpcore/P3
braino qc --dataset ~/.cache/braino/erpcore/P3 --subject 001
```

## Run the analysis steps

Each step is a tool. Steps that change which data the results rest on are *decision* tools:
they propose, and you approve.

```bash
DATA=~/.cache/braino/erpcore/P3
braino tools run eeg.preprocess --dataset $DATA --args \
  '{"subject": "001", "l_freq": 0.1, "h_freq": 30, "resample_hz": 256, "reference": ["P9", "P10"]}'

braino tools run eeg.ica_clean --dataset $DATA --args '{"subject": "001"}'
#   ... proposal 20261025T101500Z-1a2b3c-2 awaits your decision
braino proposals show 20261025T101500Z-1a2b3c-2 --dataset $DATA    # which components, why
braino proposals approve 20261025T101500Z-1a2b3c-2 --dataset $DATA
```

Then `eeg.detect_bad_channels` (approve its proposal the same way), `eeg.epoch` and `eeg.erp`.
Or let a pipeline do it all; the built-in ERP CORE pipeline is exactly these steps:

```bash
braino pipeline show erpcore-p3            # the steps and their settings
braino pipeline run erpcore-p3 --dataset $DATA --subjects 001
#   stops at each decision: approve it, then run again to continue
```

## The same from Python

```python
from braino.datasets import erpcore
from braino.tools import RunContext, decide, registry

root = erpcore.fetch("P3", ["001"])
ctx = RunContext(dataset_root=root)


def run(tool, **args):
    return registry.get(tool).run({"subject": "001", "task": "P3", **args}, ctx)


run("eeg.preprocess", l_freq=0.1, h_freq=30, resample_hz=256, reference=["P9", "P10"])

proposal = run("eeg.ica_clean")
print(proposal.proposal["exclude"])  # components proposed for removal
decide(proposal.proposal_id, ctx, approve=True, decided_by="user:me")

proposal = run("eeg.detect_bad_channels")
print(proposal.proposal["reasons"])
decide(proposal.proposal_id, ctx, approve=True, decided_by="user:me")

rare = [f"stimulus/{d}{d}" for d in range(1, 6)]
frequent = [f"stimulus/{a}{b}" for a in range(1, 6) for b in range(1, 6) if a != b]
run("eeg.epoch", conditions={"rare": rare, "frequent": frequent}, baseline=[-0.2, 0])
result = run(
    "eeg.erp",
    contrasts=[{"name": "difference", "plus": "rare", "minus": "frequent"}],
    measures=[{"name": "P3", "channels": ["Pz"], "tmin": 0.3, "tmax": 0.6}],
)
for m in result.measures:
    print(m.waveform, m.value_uv)  # the published difference for 001 is 6.59 µV
```

Everything was written to `derivatives/braino/` in the dataset, and every call was recorded:

```bash
braino runs list --dataset $DATA
braino runs export <run-id> --dataset $DATA -o p3_001.py   # a script that re-runs it
```

Continue with the [guide](guide/analysis.md) or the [examples](examples.md).
