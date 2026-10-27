# BrainO

An open-source **EEG and fMRI analysis platform** with a **local AI analysis agent** and
interactive 3D visualisation.

- Import any common EEG/MEG and MRI/fMRI format, normalised to BIDS
- Validated preprocessing, statistics, machine learning and connectivity, built on MNE-Python,
  nilearn and scikit-learn
- An analysis agent running on a local, open-weight model (Qwen): it plans and runs analyses
  through real tools, asks before key decisions, and never sends data off your machine
- A desktop app with an embedded Unity 3D brain viewer

> **Status:** alpha. The EEG engine (v0.1: import, analysis pipeline, provenance and script
> export, validated against ERP CORE) is usable from Python and the command line; the agent,
> app and viewer are in development. See the roadmap.

| | |
|---|---|
| Documentation | <https://kmnuddin.github.io/BrainO_v2/> |
| Plan and roadmap | [`docs/PLAN.md`](docs/PLAN.md) |
| Python engine | [`engine/`](engine/README.md) |
| Example notebooks | [`examples/`](examples/) |
| Changes | [`CHANGELOG.md`](CHANGELOG.md) |
| Contributing | [`CONTRIBUTING.md`](CONTRIBUTING.md) |

## Quick start

```bash
pip install "braino[eeg]"
braino pipeline list                                  # built-in analyses
braino validate erpcore P3 --subjects 001,002,003     # reproduce a published effect
```

For development:

```bash
cd engine
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,eeg]"
```

## History

BrainO 2 is a new codebase. It succeeds BrainO 1.x, a Unity tool for visualising brain
connectivity on standard atlases (Moinuddin, Mohammed & Bidelman, 2019,
doi:10.5281/zenodo.3403183).

## License

GPL-3.0-or-later (see [`LICENSE`](LICENSE)). The licence choice is still open; see
[`docs/PLAN.md`](docs/PLAN.md), section 11.
