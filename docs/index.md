# BrainO

BrainO is an open-source EEG analysis engine for researchers. It imports recordings in any
common format into de-identified BIDS datasets, runs a validated analysis pipeline (filtering,
ICA, bad channels, epoching, ERPs, spectra, time-frequency and group statistics), and records
everything it does, so that every analysis can be exported as a script that re-runs it and
checks its results.

It is built on [MNE-Python](https://mne.tools), [MNE-BIDS](https://mne.tools/mne-bids/),
[PyPREP](https://pyprep.readthedocs.io), [autoreject](https://autoreject.github.io) and
[mne-icalabel](https://mne.tools/mne-icalabel/), and is the engine of a larger platform with a
local AI analysis agent and a desktop app (see the [roadmap](PLAN.md)).

!!! note "Status: alpha (v0.1)"
    The EEG engine, its command line and its Python API are usable; the AI agent, the desktop
    app and the 3D viewer are in development. Interfaces may still change between 0.x releases.

## What it does

- **Import** EDF/BDF, BrainVision, FIF, EEGLAB, Neuroscan CNT, EGI MFF, GDF, Curry, Nihon Kohden
  and Persyst recordings into BIDS, removing names, dates and other personal details on the
  way. [More](guide/importing.md)
- **Analyse** with typed *tools*: one function per step, with a JSON schema for its parameters.
  The same tools are used from Python, the command line and (later) the AI agent.
  [More](guide/analysis.md)
- **Decide** what the results rest on: steps that drop data (bad channels, ICA components) only
  *propose*; a person approves, edits or rejects, and the decision is recorded.
- **Pipelines** declared in YAML run over a whole dataset, skipping steps whose inputs and
  settings have not changed. [More](guide/pipelines.md)
- **Reproduce**: every tool call is recorded with input and output hashes and library
  versions; any run or pipeline exports as a Python script that re-runs it and reports whether
  each result is the same. [More](guide/reproducibility.md)
- **Validated** against ERP CORE (Kappenman et al., 2021): the P3 and N170 pipelines reproduce
  the published effects, participant by participant (r = 0.95 and 0.97). [More](guide/validation.md)

## Install

```bash
pip install "braino[eeg]"
```

Python 3.11 or newer. The `eeg` extra installs the scientific stack; without it, the core
(tools, provenance, pipelines specs, command line) still installs and imports.

Next: [Getting started](getting-started.md).
