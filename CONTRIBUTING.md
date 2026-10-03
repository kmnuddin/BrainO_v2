# Contributing to BrainO

BrainO is an open-source EEG and fMRI analysis platform with a local AI analysis
agent. The plan and roadmap are in [`docs/PLAN.md`](docs/PLAN.md).

## Repository layout

| Path | What it is |
|---|---|
| `engine/` | Python analysis engine (`braino` package): import, analysis, tools, agent, CLI |
| `docs/` | Plan and design documents |
| `.github/` | CI workflows |

## Working on the engine

See [`engine/README.md`](engine/README.md) for setup. Before opening a pull request, run:

```bash
cd engine
ruff check . && ruff format --check . && mypy && pytest
```

Guidelines:

- **New analysis steps are tools** (`braino.tools`): one Pydantic input model, one output model,
  a clear docstring and field descriptions, and the right risk level. See `engine/README.md`.
- **Science needs evidence.** A new analysis method comes with a test against known results
  (synthetic data with a known answer, or a published result on a public dataset).
- **Keep tools deterministic** where possible (fixed random seeds as parameters), so analyses can
  be reproduced from their provenance record.
- Type hints are required (`mypy --strict`).

## Reporting issues

Please include the BrainO version (`braino version`), your operating system, the data format, and
the smallest steps that reproduce the problem. Never attach data that could identify a
participant or patient.
