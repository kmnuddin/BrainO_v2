"""Where BrainO keeps its files.

Downloaded and cached files go under one folder: ``$BRAINO_DATA`` if set, otherwise
``~/.cache/braino``. It holds validation datasets (``erpcore/``), eval datasets (``evals/``),
the model runtime (``runtime/``) and the AI models (``models/``). Deleting it only costs a new
download.

What the user makes, such as pipelines built in the app, goes elsewhere: ``$BRAINO_HOME`` if
set, otherwise the operating system's folder for application data (``%APPDATA%\\BrainO`` on
Windows, ``~/Library/Application Support/BrainO`` on macOS, ``~/.config/braino`` on Linux).
Clearing the cache never deletes it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

DATA_ENV = "BRAINO_DATA"
HOME_ENV = "BRAINO_HOME"


def data_root() -> Path:
    return Path(os.environ.get(DATA_ENV) or Path.home() / ".cache" / "braino")


def user_root() -> Path:
    """The folder for what the user makes; see the module description."""
    if os.environ.get(HOME_ENV):
        return Path(os.environ[HOME_ENV])
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        return Path(appdata) / "BrainO" if appdata else Path.home() / "BrainO"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "BrainO"
    config = os.environ.get("XDG_CONFIG_HOME")
    return (Path(config) if config else Path.home() / ".config") / "braino"


def user_pipelines_dir() -> Path:
    """Pipelines the user saved, one ``<name>.yaml`` each."""
    return user_root() / "pipelines"
