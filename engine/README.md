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
    out_dir = ctx.output_dir("sub-01", "eeg")   # created on first use
    ...


ctx = RunContext(dataset_root=Path("path/to/bids"))
registry.get("eeg.band_power").run({"recording": "...", "band": [8, 12]}, ctx)
```

## Checks

Run these before committing; CI runs the same commands.

```bash
ruff check . && ruff format --check . && mypy && pytest
```
