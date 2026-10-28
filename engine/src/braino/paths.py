"""Where BrainO keeps downloaded and cached files.

Everything goes under one folder: ``$BRAINO_DATA`` if set, otherwise ``~/.cache/braino``.
It holds validation datasets (``erpcore/``), eval datasets (``evals/``), the model runtime
(``runtime/``) and the AI models (``models/``). Deleting it only costs a new download.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_ENV = "BRAINO_DATA"


def data_root() -> Path:
    return Path(os.environ.get(DATA_ENV) or Path.home() / ".cache" / "braino")
