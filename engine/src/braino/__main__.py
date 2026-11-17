"""``python -m braino``: the command line, for when the ``braino`` launcher is not on the path
(the desktop app starts the engine service this way)."""

from braino.cli import app

if __name__ == "__main__":
    app(prog_name="braino")
