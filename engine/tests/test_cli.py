from __future__ import annotations

import json

from typer.testing import CliRunner

from braino import __version__
from braino.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_tools_list_includes_builtin() -> None:
    result = runner.invoke(app, ["tools", "list"])
    assert result.exit_code == 0
    assert "system.info" in result.stdout


def test_tools_schema_is_json() -> None:
    result = runner.invoke(app, ["tools", "schema", "system.info"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["function"]["name"] == "system__info"


def test_tools_run() -> None:
    result = runner.invoke(app, ["tools", "run", "system.info"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["braino_version"] == __version__


def test_tools_run_unknown_tool_fails() -> None:
    result = runner.invoke(app, ["tools", "run", "nope.missing"])
    assert result.exit_code == 1


def test_tools_run_rejects_non_object_args() -> None:
    result = runner.invoke(app, ["tools", "run", "system.info", "--args", "[1, 2]"])
    assert result.exit_code == 1
