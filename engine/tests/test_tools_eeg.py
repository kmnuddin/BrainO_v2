"""The EEG and statistics tools end to end, on a synthetic BIDS dataset with a known effect."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mne")
pytest.importorskip("mne_bids")
pytest.importorskip("pybv")
pytest.importorskip("picard")

from synthetic_eeg import P3_WINDOW, write_dataset
from typer.testing import CliRunner

from braino.cli import app
from braino.provenance import list_runs, load_run
from braino.tools import PendingProposal, RunContext, decide, registry
from braino.tools.eeg import MissingStepError, StaleStepError

SUBJECTS = ["01", "02", "03", "04"]
TASK = "oddball"
CONDITIONS = {"rare": ["stimulus/rare"], "frequent": ["stimulus/frequent"]}
CONTRAST = {"name": "difference", "plus": "rare", "minus": "frequent"}
P3 = {"name": "P3", "channels": ["Pz"], "tmin": P3_WINDOW[0], "tmax": P3_WINDOW[1]}


def run(ctx: RunContext, name: str, subject: str, **args: Any) -> Any:
    return registry.get(name).run({"subject": subject, "task": TASK, **args}, ctx)


def approve(ctx: RunContext, result: Any, **edits: Any) -> Any:
    assert isinstance(result, PendingProposal)
    record = decide(result.proposal_id, ctx, approve=True, decided_by="user:test", edits=edits)
    assert record.decision is not None
    return record.decision.apply_result


@pytest.fixture(scope="module")
def processed(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Every subject taken through the whole chain once."""
    root = write_dataset(tmp_path_factory.mktemp("bids"), SUBJECTS, task=TASK, seconds=90.0)
    ctx = RunContext(dataset_root=root, run_id="chain")
    for s in SUBJECTS:
        run(ctx, "eeg.preprocess", s, h_freq=30.0)
        approve(ctx, run(ctx, "eeg.detect_bad_channels", s, method="qc"))
        approve(ctx, run(ctx, "eeg.ica_clean", s, labeler="eog"))
        run(ctx, "eeg.epoch", s, conditions=CONDITIONS, reject="peak_to_peak")
        run(ctx, "eeg.erp", s, contrasts=[CONTRAST], measures=[P3])
        run(ctx, "eeg.spectral", s)
        run(ctx, "eeg.tfr", s, fmin=8.0, n_freqs=6, decim=5)
    return root


def derivatives(root: Path, subject: str) -> Path:
    return root / "derivatives" / "braino" / f"sub-{subject}" / "eeg"


def test_every_step_writes_its_files(processed: Path) -> None:
    names = {p.name for p in derivatives(processed, "01").iterdir()}
    stem = f"sub-01_task-{TASK}_"
    for suffix in [
        "desc-preproc_eeg.fif",
        "desc-preproc_eeg.json",
        "desc-preproc_channels.tsv",
        "desc-preproc_ica.fif",
        "desc-clean_eeg.fif",
        "desc-clean_eeg.json",
        "desc-epochs_epo.fif",
        "desc-epochs_ave.fif",
        "desc-epochs_erp.tsv",
        "desc-epochs_psd.npz",
        "desc-epochs_bands.tsv",
        "desc-epochs_tfr.npz",
    ]:
        assert stem + suffix in names


def test_provenance_records_the_chain(processed: Path) -> None:
    calls = load_run(processed / "derivatives" / "braino", "chain").calls
    tools = [
        c.tool for c in calls if c.arguments.get("subject", "01") == "01" or ".apply" in c.tool
    ]
    assert tools[:5] == [
        "eeg.preprocess",
        "eeg.detect_bad_channels",
        "eeg.detect_bad_channels.apply",
        "eeg.ica_clean",
        "eeg.ica_clean.apply",
    ]
    assert all(c.status == "ok" for c in calls)
    preprocess = calls[0]
    assert any(i.path.endswith("_eeg.vhdr") for i in preprocess.inputs)
    assert {o.path.rsplit("_", 1)[-1] for o in preprocess.outputs} == {"eeg.fif", "eeg.json"}


def test_erp_recovers_the_planted_effect(processed: Path) -> None:
    ctx = RunContext(dataset_root=processed)
    result = run(ctx, "eeg.erp", "02", contrasts=[CONTRAST], measures=[P3])
    values = {m.waveform: m.value_uv for m in result.measures}
    assert values["difference"] > 3.0  # 8 µV at Pz, less the average reference's share
    assert abs(values["frequent"]) < 1.5
    with (derivatives(processed, "02") / f"sub-02_task-{TASK}_desc-epochs_erp.tsv").open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    assert {r["waveform"] for r in rows} == {"rare", "frequent", "difference"}


def test_spectral_and_tfr_summaries(processed: Path) -> None:
    ctx = RunContext(dataset_root=processed)
    spectral = run(ctx, "eeg.spectral", "01", source="continuous", method="welch")
    assert spectral.conditions["continuous"].peak_alpha_hz == pytest.approx(10, abs=0.6)
    assert spectral.output.endswith("desc-continuous_psd.npz")
    tfr = run(ctx, "eeg.tfr", "01", fmin=8.0, n_freqs=6)
    assert tfr.unit == "dB"
    assert set(tfr.n_epochs) == {"rare", "frequent"}
    with pytest.raises(ValueError, match="Nyquist"):
        run(ctx, "eeg.tfr", "01", fmax=200.0)


def test_group_statistics_find_the_effect(processed: Path) -> None:
    ctx = RunContext(dataset_root=processed)
    args = {"task": TASK, "conditions": ["difference"], "tmin": 0.25, "tmax": 0.55}
    # With an average reference, channels far from Pz carry the P3's mirror image; look at the
    # back of the head, where it is positive.
    posterior = ["P3", "Pz", "P4", "O1", "Oz", "O2"]
    uni: Any = registry.get("stats.mass_univariate").run(
        {**args, "correction": "fdr", "channels": posterior}, ctx
    )
    assert uni.subjects == SUBJECTS
    assert uni.result.peak_channel in posterior
    assert uni.result.peak_stat > 0
    table = processed / uni.table
    assert table.is_file() and table.parent.name == f"task-{TASK}"
    clusters: Any = registry.get("stats.cluster_permutation").run(
        {**args, "n_permutations": 200}, ctx
    )
    assert clusters.result.n_clusters >= 1
    assert clusters.result.clusters[0].sign == "positive"
    paired: Any = registry.get("stats.mass_univariate").run(
        {**args, "conditions": ["rare", "frequent"], "channels": ["Pz"]}, ctx
    )
    assert paired.n_channels == 1
    assert paired.result.test.startswith("paired")
    psd: Any = registry.get("stats.mass_univariate").run(
        {"task": TASK, "data": "psd", "conditions": ["rare", "frequent"], "fmin": 5, "fmax": 20},
        ctx,
    )
    assert psd.unit.startswith("dB")
    assert psd.window["freq"][0] >= 5


def test_group_statistics_errors(processed: Path) -> None:
    ctx = RunContext(dataset_root=processed)
    tool = registry.get("stats.mass_univariate")
    with pytest.raises(ValueError, match="has no 'nope'"):
        tool.run({"task": TASK, "conditions": ["nope"]}, ctx)
    with pytest.raises(ValueError, match="at least 3"):
        tool.run({"task": TASK, "conditions": ["difference"], "subjects": ["01", "02"]}, ctx)
    with pytest.raises(ValueError, match="no frequency axis"):
        tool.run({"task": TASK, "conditions": ["difference"], "fmin": 1}, ctx)
    with pytest.raises(ValueError, match="one condition"):
        registry.get("stats.cluster_permutation").run(
            {"task": TASK, "conditions": ["rare", "frequent", "difference"]}, ctx
        )


def test_steps_need_their_inputs(tmp_path: Path) -> None:
    root = write_dataset(tmp_path, ["01"], task=TASK, seconds=20.0)
    ctx = RunContext(dataset_root=root)
    for name in ("eeg.detect_bad_channels", "eeg.ica_clean"):
        with pytest.raises(MissingStepError, match=r"run eeg\.preprocess first"):
            run(ctx, name, "01")
    with pytest.raises(MissingStepError, match=r"run eeg\.epoch first"):
        run(ctx, "eeg.erp", "01")


def test_bad_channel_decision_edits_and_reference(tmp_path: Path) -> None:
    root = write_dataset(tmp_path, ["01"], task=TASK, seconds=30.0, noisy=["T8"])
    ctx = RunContext(dataset_root=root)
    run(ctx, "eeg.preprocess", "01", reference=["P7", "P8"])
    proposal = run(ctx, "eeg.detect_bad_channels", "01", method="qc")
    assert "T8" in proposal.proposal["channels"]
    with pytest.raises(ValueError, match="is the reference"):
        approve(ctx, proposal, channels=["T8", "P7"])
    applied = approve(ctx, proposal, channels=["T8", "C3"])
    assert applied["bad_channels"] == ["C3", "T8"]
    table = root / applied["table"]
    rows = {r["name"]: r for r in csv.DictReader(table.open(), delimiter="\t")}
    assert rows["C3"]["status_description"] == "marked by reviewer"
    assert rows["Cz"]["status"] == "good"

    # The ICA and epochs leave the approved channels out / interpolate them.
    approve(ctx, run(ctx, "eeg.ica_clean", "01", labeler="eog"))
    epochs = run(ctx, "eeg.epoch", "01", conditions=CONDITIONS, reject="none")
    assert epochs.interpolated_channels == ["C3", "T8"]
    assert epochs.source == "clean"


def test_decisions_on_an_old_preprocessing_are_refused(tmp_path: Path) -> None:
    root = write_dataset(tmp_path, ["01"], task=TASK, seconds=30.0)
    ctx = RunContext(dataset_root=root)
    run(ctx, "eeg.preprocess", "01")
    approve(ctx, run(ctx, "eeg.detect_bad_channels", "01", method="qc"))
    approve(ctx, run(ctx, "eeg.ica_clean", "01", labeler="eog"))
    run(ctx, "eeg.preprocess", "01", h_freq=20.0)  # preprocessing changed after the decisions
    with pytest.raises(StaleStepError, match="ica_clean again"):
        run(ctx, "eeg.epoch", "01", conditions=CONDITIONS)

    # ICA can be run again without the old bad channels; epoching still refuses them.
    ica = run(ctx, "eeg.ica_clean", "01", labeler="eog")
    assert any("earlier preprocessing" in n for n in ica.proposal["notes"])
    assert ica.proposal["bad_channels"] == []
    approve(ctx, ica)
    with pytest.raises(StaleStepError, match="detect_bad_channels again"):
        run(ctx, "eeg.epoch", "01", conditions=CONDITIONS)
    detect = run(ctx, "eeg.detect_bad_channels", "01", method="qc")
    assert any("ICA-cleaned" in n for n in detect.proposal["notes"])
    approve(ctx, detect)
    result = run(ctx, "eeg.epoch", "01", conditions=CONDITIONS, reject="none")
    assert result.source == "clean"


def test_bad_channels_fall_back_when_the_cleaned_recording_is_stale(tmp_path: Path) -> None:
    root = write_dataset(tmp_path, ["01"], task=TASK, seconds=30.0)
    ctx = RunContext(dataset_root=root)
    run(ctx, "eeg.preprocess", "01")
    approve(ctx, run(ctx, "eeg.ica_clean", "01", labeler="eog"))
    run(ctx, "eeg.preprocess", "01", h_freq=20.0)
    detect = run(ctx, "eeg.detect_bad_channels", "01", method="qc")
    assert any("earlier preprocessing" in n for n in detect.proposal["notes"])


def test_proposal_paths_must_be_derivatives(tmp_path: Path) -> None:
    root = write_dataset(tmp_path, ["01"], task=TASK, seconds=20.0)
    ctx = RunContext(dataset_root=root)
    run(ctx, "eeg.preprocess", "01")
    proposal = run(ctx, "eeg.detect_bad_channels", "01", method="qc")
    with pytest.raises(ValueError, match="not a BrainO derivative"):
        approve(ctx, proposal, recording="sub-01/eeg/sub-01_task-oddball_eeg.vhdr")


def test_cli_runs_the_tools(tmp_path: Path) -> None:
    root = write_dataset(tmp_path, ["01"], task=TASK, seconds=20.0)
    runner = CliRunner()
    args = json.dumps({"subject": "01", "h_freq": 30})
    result = runner.invoke(
        app, ["tools", "run", "eeg.preprocess", "--args", args, "--dataset", str(root)]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["output"].endswith("desc-preproc_eeg.fif")
    detect = json.dumps({"subject": "01", "method": "qc"})
    result = runner.invoke(
        app, ["tools", "run", "eeg.detect_bad_channels", "--args", detect, "--dataset", str(root)]
    )
    assert result.exit_code == 0, result.output
    assert "awaits your decision" in result.output
    assert list_runs(root / "derivatives" / "braino")
