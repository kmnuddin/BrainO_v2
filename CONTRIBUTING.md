# Contributing to BrainO

BrainO is an open-source EEG and fMRI analysis platform with a local AI analysis
agent. The plan and roadmap are in [`docs/PLAN.md`](docs/PLAN.md).

## Repository layout

| Path | What it is |
|---|---|
| `engine/` | Python analysis engine (`braino` package): import, analysis, tools, agent, CLI |
| `docs/` | Documentation site (MkDocs) and the plan |
| `examples/` | Example notebooks, with their outputs |
| `.github/` | CI, documentation and release workflows |

## Working on the engine

See [`engine/README.md`](engine/README.md) for setup. Before opening a pull request, run:

```bash
cd engine
ruff check . && ruff format --check . && mypy && pytest
```

To preview the documentation site (`mkdocs.yml` at the repository root):

```bash
pip install -e "engine[docs]"
mkdocs serve
```

Guidelines:

- **New analysis steps are tools** (`braino.tools`): one Pydantic input model, one output model,
  a clear docstring and field descriptions, and the right risk level. See `engine/README.md`.
- **Science needs evidence.** A new analysis method comes with a test against known results
  (synthetic data with a known answer, or a published result on a public dataset).
- **Keep tools deterministic** where possible (fixed random seeds as parameters), so analyses can
  be reproduced from their provenance record.
- Type hints are required (`mypy --strict`).

## Releasing

1. Update `CHANGELOG.md` and `__version__` in `engine/src/braino/__init__.py`.
2. Tag the commit `v<version>` and push the tag. The `release` workflow runs the checks, builds
   and tests the wheel, publishes it to PyPI (trusted publishing, environment `pypi`) and
   creates a GitHub release with the changelog entry.

## Reporting issues

Please include the BrainO version (`braino version`), your operating system, the data format, and
the smallest steps that reproduce the problem. Never attach data that could identify a
participant or patient.
