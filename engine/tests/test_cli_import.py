"""braino import, inspect and qc, end to end."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("mne")
pytest.importorskip("mne_bids")
pytest.importorskip("pybv")

from test_io_bids import planted_source
from test_io_deidentify import PLANTED
from typer.testing import CliRunner, Result

from braino.cli import app
from braino.io import MappingError, read_mapping
from braino.provenance import list_runs, load_run

runner = CliRunner()


@pytest.fixture
def setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    source_dir = tmp_path / "Jane Doe records"
    source_dir.mkdir()
    return planted_source(source_dir), tmp_path / "bids", tmp_path / "private" / "key.json"


def run_import(*args: str) -> Result:
    return runner.invoke(app, ["import", *args])


def test_import_one_recording(setup: tuple[Path, Path, Path]) -> None:
    source, bids, key = setup
    result = run_import(
        str(source),
        "--dataset",
        str(bids),
        "--key",
        str(key),
        "--subject-id",
        "Jane Doe",
        "--task",
        "rest",
    )
    assert result.exit_code == 0, result.output
    assert "OK      sub-001/eeg/sub-001_task-rest_eeg.vhdr" in result.stdout
    assert "check this event text by hand" in result.stdout
    assert "1 of 1 recordings imported" in result.stdout
    assert key.is_file()


def test_provenance_of_an_import_holds_no_personal_details(
    setup: tuple[Path, Path, Path],
) -> None:
    source, bids, key = setup
    run_import(
        str(source),
        "--dataset",
        str(bids),
        "--key",
        str(key),
        "--subject-id",
        "Jane Doe",
        "--task",
        "rest",
    )
    [run_id] = list_runs(bids / "derivatives" / "braino")
    [call] = load_run(bids / "derivatives" / "braino", run_id).calls
    assert call.tool == "dataset.import"
    assert call.arguments == {"subject": "001", "task": "rest", "session": None, "run": None}
    assert all(f.path.startswith("<outside-dataset>/") for f in call.inputs)
    assert any(f.path.endswith("_eeg.eeg") for f in call.outputs)
    written = "".join(
        p.read_text(encoding="utf-8") for p in (bids / "derivatives").rglob("*") if p.is_file()
    ).lower()
    for value in [*PLANTED, "records", "jane_doe"]:
        assert value.lower() not in written, value


def test_failed_import_is_recorded_without_personal_details(tmp_path: Path) -> None:
    bids, key = tmp_path / "bids", tmp_path / "key.json"
    missing = tmp_path / "Jane Doe" / "Jane_Doe_raw.fif"
    result = run_import(
        str(missing),
        "--dataset",
        str(bids),
        "--key",
        str(key),
        "--subject-id",
        "MRN12345678",
        "--task",
        "rest",
    )
    assert result.exit_code == 1
    assert "FAILED" in result.stderr
    [run_id] = list_runs(bids / "derivatives" / "braino")
    [call] = load_run(bids / "derivatives" / "braino", run_id).calls
    assert call.status == "error"
    assert call.error is not None
    for value in ("Jane", "MRN12345678"):
        assert value not in call.error
        assert value not in json.dumps(call.arguments)


def test_import_from_a_mapping_file(setup: tuple[Path, Path, Path]) -> None:
    source, bids, key = setup
    mapping = source.parent.parent / "mapping.csv"
    mapping.write_text(
        "source,subject_id,task,session,run\n"
        f"{source.parent.name}/{source.name},Jane Doe,rest,pre,1\n"
        f"{source.parent.name}/{source.name},Jane Doe,rest,post,\n"
        "missing.fif,John Roe,rest,,\n",
        encoding="utf-8",
    )
    result = run_import("--mapping", str(mapping), "--dataset", str(bids), "--key", str(key))
    assert result.exit_code == 1  # the third row fails
    assert "2 of 3 recordings imported" in result.stdout
    assert (
        bids / "sub-001" / "ses-pre" / "eeg" / "sub-001_ses-pre_task-rest_run-1_eeg.vhdr"
    ).is_file()
    assert (bids / "sub-001" / "ses-post" / "eeg" / "sub-001_ses-post_task-rest_eeg.vhdr").is_file()


def test_mapping_file_checks(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    inside = bids / "code" / "mapping.csv"
    inside.parent.mkdir(parents=True)
    inside.write_text("source,subject_id,task\na.fif,A,rest\n", encoding="utf-8")
    with pytest.raises(MappingError, match="outside the dataset"):
        read_mapping(inside, dataset_root=bids)
    bad = tmp_path / "bad.csv"
    bad.write_text("source,subject,task\na.fif,A,rest\n", encoding="utf-8")
    with pytest.raises(MappingError, match="subject_id"):
        read_mapping(bad, dataset_root=bids)


def test_key_is_required_and_must_be_outside(setup: tuple[Path, Path, Path]) -> None:
    source, bids, _ = setup
    args = [str(source), "--dataset", str(bids), "--subject-id", "A", "--task", "rest"]
    assert "--key is required" in run_import(*args).stderr
    inside = run_import(*args, "--key", str(bids / "key.json"))
    assert inside.exit_code == 1
    assert "outside the dataset" in inside.stderr


def test_inspect_and_qc(setup: tuple[Path, Path, Path]) -> None:
    source, bids, key = setup
    run_import(
        str(source),
        "--dataset",
        str(bids),
        "--key",
        str(key),
        "--subject-id",
        "Jane Doe",
        "--task",
        "rest",
    )
    inspected = runner.invoke(app, ["inspect", "--dataset", str(bids)])
    assert inspected.exit_code == 0, inspected.output
    assert "1 subjects, 1 recordings" in inspected.stdout
    assert (
        "sub-001/eeg/sub-001_task-rest_eeg.vhdr  100 Hz, 10.0 s; 3 eeg; 5 events"
        in inspected.stdout
    )

    as_json = runner.invoke(app, ["inspect", "--dataset", str(bids), "--json"])
    assert json.loads(as_json.stdout)["n_recordings"] == 1

    qc = runner.invoke(app, ["qc", "--dataset", str(bids)])
    assert qc.exit_code == 0, qc.output
    assert qc.stdout.startswith("sub-001/eeg/sub-001_task-rest_eeg.vhdr: 3 channels")
    assert (
        bids
        / "derivatives"
        / "braino"
        / "sub-001"
        / "eeg"
        / "sub-001_task-rest_desc-qc_channels.tsv"
    ).is_file()

    one = runner.invoke(app, ["qc", "--dataset", str(bids), "--subject", "001", "--json"])
    assert json.loads(one.stdout)[0]["n_channels_checked"] == 3
    missing = runner.invoke(app, ["qc", "--dataset", str(bids), "--subject", "999"])
    assert missing.exit_code == 1
    assert "no recording matches" in missing.stderr


def test_manufacturer_comes_from_the_source_format(setup: tuple[Path, Path, Path]) -> None:
    source, bids, key = setup
    run_import(
        str(source),
        "--dataset",
        str(bids),
        "--key",
        str(key),
        "--subject-id",
        "Jane Doe",
        "--task",
        "rest",
    )
    sidecar = json.loads(
        (bids / "sub-001" / "eeg" / "sub-001_task-rest_eeg.json").read_text(encoding="utf-8")
    )
    assert sidecar["Manufacturer"] == "n/a"  # EEG-only FIF says nothing about its maker
