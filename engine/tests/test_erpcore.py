"""ERP CORE: pinned manifest, published references, and the comparison with them."""

from __future__ import annotations

import re

import pytest

from braino.datasets import erpcore


def test_manifest_pins_every_file() -> None:
    for task in erpcore.TASKS:
        entries = erpcore.manifest()["tasks"][task]["files"]
        subjects = {e["path"].split("/")[0] for e in entries if e["path"].startswith("sub-")}
        assert subjects == {f"sub-{s}" for s in erpcore.SUBJECTS}
        for entry in entries:
            assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), entry["path"]
            assert entry["url"].startswith("https://osf.io/download/")
            assert entry["size"] > 0


def test_each_subject_has_a_full_bids_recording() -> None:
    names = {e["path"] for e in erpcore.files("P3", ["001"])}
    for suffix in ("eeg.set", "eeg.fdt", "eeg.json", "channels.tsv", "events.tsv"):
        assert f"sub-001/eeg/sub-001_task-P3_{suffix}" in names
    assert "dataset_description.json" in names
    assert not any(n.startswith("sub-002/") for n in names)


def test_stimuli_only_on_request() -> None:
    without = erpcore.files("N170", [])
    with_stimuli = erpcore.files("N170", [], stimuli=True)
    assert all("/" not in e["path"] for e in without)
    assert any(e["path"].endswith(".bmp") for e in with_stimuli)


@pytest.mark.parametrize("bad", ["041", "0", "sub-abc"])
def test_subject_labels_are_checked(bad: str) -> None:
    with pytest.raises(ValueError, match="001-040"):
        erpcore.files("P3", [bad])


def test_unknown_task() -> None:
    with pytest.raises(ValueError, match="unknown ERP CORE task"):
        erpcore.files("MMN")


def test_references_match_the_paper() -> None:
    """Kappenman et al. (2021), Table 3."""
    p3, n170 = erpcore.references()["P3"], erpcore.references()["N170"]
    assert (p3.n, round(p3.mean_uv, 2), round(p3.sd_uv, 2)) == (34, 6.29, 3.39)
    assert (p3.electrode, p3.window_ms) == ("Pz", (300, 600))
    assert p3.excluded_subjects == ["006", "009", "010", "030", "035", "040"]
    assert (n170.n, round(n170.mean_uv, 2), round(n170.sd_uv, 2)) == (37, -3.37, 2.71)
    assert (n170.electrode, n170.window_ms) == ("PO8", (110, 150))
    assert "NeuroImage" in erpcore.citation()


# Comparison with the published results


def test_published_scores_reproduce_themselves() -> None:
    pytest.importorskip("scipy")
    from braino.validation.erpcore import compare

    published = erpcore.references()["P3"]
    report = compare("P3", published.subject_scores_uv, excluded=published.excluded_subjects)
    assert report.passed
    assert report.correlation == 1.0
    assert report.n_compared == 34
    assert report.braino_mean_uv == pytest.approx(6.29, abs=0.01)
    assert report.exclusions.only_braino == report.exclusions.only_published == []


def test_noisy_scores_fail_the_participant_check() -> None:
    pytest.importorskip("scipy")
    import numpy as np

    from braino.validation.erpcore import compare

    published = erpcore.references()["P3"].subject_scores_uv
    rng = np.random.default_rng(0)
    noisy = {s: v + float(rng.normal(0, 6)) for s, v in published.items()}
    report = compare("P3", noisy)
    assert report.correlation is not None
    assert report.correlation < 0.9
    assert not report.checks["participant_correlation"]
    assert not report.passed


def test_wrong_sign_fails() -> None:
    pytest.importorskip("scipy")
    from braino.validation.erpcore import compare

    published = erpcore.references()["N170"].subject_scores_uv
    flipped = {s: -v for s, v in published.items()}
    report = compare("N170", flipped)
    assert not report.checks["significant_in_published_direction"]
    assert not report.checks["ci_includes_published_mean"]


def test_exclusions_are_compared() -> None:
    pytest.importorskip("scipy")
    from braino.validation.erpcore import compare

    published = erpcore.references()["P3"]
    kept = {s: v for s, v in published.subject_scores_uv.items() if s != "001"}
    report = compare("P3", kept, excluded=["001", "006", "009"])
    assert report.exclusions.both == ["006", "009"]
    assert report.exclusions.only_braino == ["001"]
    assert report.exclusions.only_published == ["010", "030", "035", "040"]


@pytest.mark.large_download
def test_fetch_one_participant(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("mne_bids")
    import mne_bids

    from braino.io import scan_dataset

    root = erpcore.fetch("P3", ["001"], verify=True)
    summary = scan_dataset(root)
    [recording] = summary.recordings
    assert recording.subject == "001"
    assert recording.channels.get("eeg", 0) >= 28
    raw = mne_bids.read_raw_bids(
        mne_bids.BIDSPath(subject="001", task="P3", datatype="eeg", root=root), verbose="error"
    )
    assert "Pz" in raw.ch_names
    assert len(raw.annotations) > 100
