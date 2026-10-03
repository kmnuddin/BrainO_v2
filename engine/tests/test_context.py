from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from braino import __version__
from braino.context import NoDatasetError, RunContext, new_run_id


def test_run_id_format_and_uniqueness() -> None:
    first, second = new_run_id(), new_run_id()
    assert re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{6}", first)
    assert first != second


def test_context_without_dataset() -> None:
    ctx = RunContext()
    assert ctx.dataset_root is None
    with pytest.raises(NoDatasetError):
        _ = ctx.derivatives_root


def test_dataset_root_is_resolved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "ds").mkdir()
    monkeypatch.chdir(tmp_path)
    ctx = RunContext(dataset_root=Path("ds"))
    assert ctx.dataset == (tmp_path / "ds").resolve()


def test_missing_dataset_root_rejected(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        RunContext(dataset_root=tmp_path / "missing")


def test_output_dir_creates_bids_derivatives(tmp_path: Path) -> None:
    ctx = RunContext(dataset_root=tmp_path)
    out = ctx.output_dir("sub-01", "eeg")

    root = tmp_path.resolve() / "derivatives" / "braino"
    assert out == root / "sub-01" / "eeg"
    assert out.is_dir()
    description = json.loads((root / "dataset_description.json").read_text(encoding="utf-8"))
    assert description["DatasetType"] == "derivative"
    assert description["GeneratedBy"] == [{"Name": "BrainO", "Version": __version__}]


def test_output_dir_keeps_existing_description(tmp_path: Path) -> None:
    ctx = RunContext(dataset_root=tmp_path)
    ctx.output_dir()
    description = ctx.derivatives_root / "dataset_description.json"
    description.write_text('{"Name": "edited"}', encoding="utf-8")
    ctx.output_dir("sub-02")
    assert description.read_text(encoding="utf-8") == '{"Name": "edited"}'


@pytest.mark.parametrize("parts", [("..",), ("sub-01", "..", "..")])
def test_output_dir_cannot_escape_derivatives(tmp_path: Path, parts: tuple[str, ...]) -> None:
    ctx = RunContext(dataset_root=tmp_path)
    with pytest.raises(ValueError, match="outside"):
        ctx.output_dir(*parts)
