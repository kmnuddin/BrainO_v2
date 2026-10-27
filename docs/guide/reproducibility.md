# Reproducibility

## The provenance record

Every tool call made with a dataset is recorded in `derivatives/braino/runs/<run-id>/`:

- `run.json`: when, BrainO, Python and operating-system versions, and the versions of every
  analysis library (MNE, NumPy, SciPy, autoreject, PyPREP, ...);
- `calls.jsonl`: one line per call, with the tool, its validated arguments, its result,
  success or error, timing, and the path, size and SHA-256 hash of every file it read and
  wrote.

Decisions are kept in `derivatives/braino/proposals/`: what was proposed, the final version
applied, who decided, when and why.

```bash
braino runs list --dataset study_bids
braino runs show <run-id> --dataset study_bids     # the full record, as JSON
```

Files outside the dataset (the originals an import reads) are recorded by hash, not by name,
because names often identify the person.

## Exporting a script

Any run, or a pipeline's current results, exports as a standalone Python script:

```bash
braino runs export <run-id> --dataset study_bids -o analysis.py
braino pipeline export p3.yaml --dataset study_bids -o analysis.py
```

The script lists every step with its recorded arguments and every decision with the reviewer's
edits and notes. It needs BrainO and the dataset:

```bash
python analysis.py path/to/a/copy/of/study_bids
```

It first checks that the dataset's files have the recorded SHA-256 hashes (`--no-verify` skips
this), notes any library version that differs from the recorded one, then runs every step and
makes the recorded decisions again. After each step it compares the new result with the
recorded one, number by number (relative tolerance 10⁻⁴), and ends with either

```text
reproduced: all 42 steps gave the recorded results
```

or the list of differences and exit code 1. Output files are not compared byte for byte, as
MNE stores the time of writing in each FIF file; every number a tool reported is compared.

!!! warning
    The script writes to the dataset's `derivatives/braino/`, like the original run. Run it on a
    copy of the dataset to keep the original outputs.

An export from a pipeline includes steps that were reused from earlier runs, so it rebuilds the
whole analysis even when the last run only redid part of it.
