"""Training data for fine-tuning the agent's model (for development; users never need it).

Episodes (:mod:`braino.training.episodes`) are requests with a scripted way to handle them,
played through the real agent loop by a teacher standing in for the model, so the examples are
exactly the conversations the model sees at run time. Families
(:mod:`braino.training.families`) make random episodes of each kind of request. The
evaluation's tasks are held out. :func:`build` plays them and keeps the clean ones;
``braino train data`` writes them as JSONL for the training scripts in ``training/``.
"""

from braino.training.build import BuildReport, build, split
from braino.training.episodes import (
    Ask,
    Call,
    Decide,
    Episode,
    Example,
    Rejected,
    Say,
    run_episode,
    write_jsonl,
)
from braino.training.families import FAMILIES, check_held_out

__all__ = [
    "FAMILIES",
    "Ask",
    "BuildReport",
    "Call",
    "Decide",
    "Episode",
    "Example",
    "Rejected",
    "Say",
    "build",
    "check_held_out",
    "run_episode",
    "split",
    "write_jsonl",
]
