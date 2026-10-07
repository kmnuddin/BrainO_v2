"""Format detection, on small fake files (no MNE or downloads needed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from braino.io import FORMATS, MissingFileError, UnsupportedFormatError, detect_format

EDF_HEADER = b"0       " + b" " * 248
BDF_HEADER = b"\xffBIOSEMI" + b" " * 248


def write(path: Path, content: bytes | str = b"") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, str):
        path.write_text(content, encoding="utf-8")
    else:
        path.write_bytes(content)
    return path


def brainvision(folder: Path, stem: str = "rec", data: str | None = None) -> Path:
    data = data or f"{stem}.eeg"
    header = write(
        folder / f"{stem}.vhdr",
        "Brain Vision Data Exchange Header File Version 1.0\n"
        f"[Common Infos]\nDataFile={data}\nMarkerFile={stem}.vmrk\n",
    )
    write(folder / data, b"\x00" * 8)
    write(folder / f"{stem}.vmrk", "Brain Vision Data Exchange Marker File Version 1.0\n")
    return header


def test_every_format_has_extensions() -> None:
    for spec in FORMATS.values():
        assert spec.extensions
        assert all(ext == ext.lower() and ext.startswith(".") for ext in spec.extensions)


@pytest.mark.parametrize(
    ("name", "header", "expected"),
    [
        ("rec.edf", EDF_HEADER, "edf"),
        ("rec.EDF", EDF_HEADER, "edf"),
        ("rec.bdf", BDF_HEADER, "bdf"),
        ("rec.edf", BDF_HEADER, "bdf"),  # BioSemi data saved with an .edf extension
        ("rec.gdf", b"GDF 2.10" + b"\x00" * 8, "gdf"),
        ("rec_raw.fif", b"\x00" * 16, "fif"),
        ("rec_raw.fif.gz", b"\x00" * 16, "fif"),
        ("rec.cnt", b"Version 3.0\x00", "cnt"),
        ("rec.set", b"MATLAB 5.0 MAT-file", "eeglab"),
        ("rec.EEG", b"EEG-1100C V01.00", "nihon_kohden"),
        ("unknown.xyz", EDF_HEADER, "edf"),  # recognised by its header alone
    ],
)
def test_single_file_formats(tmp_path: Path, name: str, header: bytes, expected: str) -> None:
    path = write(tmp_path / name, header)
    detected = detect_format(path)
    assert detected.format.id == expected
    assert detected.path == path
    assert detected.files[0] == path


def test_wrong_header_for_extension(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedFormatError, match=r"no \.edf header"):
        detect_format(write(tmp_path / "rec.edf", b"not an edf file"))


def test_ant_cnt_is_reported_as_unsupported(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedFormatError, match="ANT Neuro"):
        detect_format(write(tmp_path / "rec.cnt", b"RIFF\x00\x00\x00\x00CNT "))


def test_unknown_format(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedFormatError, match="unrecognised format"):
        detect_format(write(tmp_path / "notes.txt", "hello"))


def test_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        detect_format(tmp_path / "missing.edf")


def test_plain_folder_is_not_a_recording(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedFormatError, match="folder"):
        detect_format(tmp_path)


def test_mff_folder(tmp_path: Path) -> None:
    folder = tmp_path / "rec.mff"
    write(folder / "info.xml", "<fileInfo/>")
    detected = detect_format(folder)
    assert detected.format.id == "egi_mff"
    assert detected.files == (folder,)


@pytest.mark.parametrize("pick", [".vhdr", ".eeg", ".vmrk"])
def test_brainvision_from_any_of_its_files(tmp_path: Path, pick: str) -> None:
    header = brainvision(tmp_path)
    detected = detect_format(tmp_path / f"rec{pick}")
    assert detected.format.id == "brainvision"
    assert detected.path == header
    assert {p.name for p in detected.files} == {"rec.vhdr", "rec.eeg", "rec.vmrk"}


def test_brainvision_follows_renamed_data_file(tmp_path: Path) -> None:
    header = brainvision(tmp_path, data="old_name.eeg")
    detected = detect_format(tmp_path / "old_name.eeg")
    assert detected.path == header
    assert tmp_path / "old_name.eeg" in detected.files


def test_brainvision_missing_data_file(tmp_path: Path) -> None:
    header = brainvision(tmp_path)
    (tmp_path / "rec.eeg").unlink()
    with pytest.raises(MissingFileError, match=r"rec\.eeg"):
        detect_format(header)


def test_orphan_eeg_file(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedFormatError, match="BrainVision"):
        detect_format(write(tmp_path / "rec.eeg", b"\x00" * 16))


def test_eeglab_with_fdt(tmp_path: Path) -> None:
    header = write(tmp_path / "rec.set", b"MATLAB 5.0 MAT-file")
    write(tmp_path / "rec.fdt", b"\x00" * 8)
    for pick in ("rec.set", "rec.fdt"):
        detected = detect_format(tmp_path / pick)
        assert detected.path == header
        assert [p.name for p in detected.files] == ["rec.set", "rec.fdt"]


def test_nihon_kohden_from_companion(tmp_path: Path) -> None:
    data = write(tmp_path / "MB01.EEG", b"EEG-2100  V01.00")
    for ext in (".PNT", ".LOG", ".21E"):
        write(tmp_path / f"MB01{ext}", b"\x00")
    detected = detect_format(tmp_path / "MB01.PNT")
    assert detected.format.id == "nihon_kohden"
    assert detected.path == data
    assert len(detected.files) == 4


@pytest.mark.parametrize("pick", ["rec x.cdt", "rec x.cdt.dpa", "rec x.cdt.cef"])
def test_curry8(tmp_path: Path, pick: str) -> None:
    data = write(tmp_path / "rec x.cdt", b"\x00" * 8)
    write(tmp_path / "rec x.cdt.dpa", "Curry header")
    write(tmp_path / "rec x.cdt.cef", "Curry events")
    detected = detect_format(tmp_path / pick)
    assert detected.format.id == "curry"
    assert detected.path == data
    assert len(detected.files) == 3


def test_curry7(tmp_path: Path) -> None:
    data = write(tmp_path / "rec.dat", b"\x00" * 8)
    write(tmp_path / "rec.dap", "Curry header")
    write(tmp_path / "rec.rs3", "Curry labels")
    for pick in ("rec.dat", "rec.dap"):
        detected = detect_format(tmp_path / pick)
        assert detected.format.id == "curry"
        assert detected.path == data


def test_curry_missing_header(tmp_path: Path) -> None:
    with pytest.raises(MissingFileError, match=r"\.cdt\.dpa"):
        detect_format(write(tmp_path / "rec.cdt", b"\x00"))


@pytest.mark.parametrize("pick", ["rec.lay", "data.dat"])
def test_persyst(tmp_path: Path, pick: str) -> None:
    layout = write(tmp_path / "rec.lay", "[FileInfo]\nFile=data.dat\nSamplingRate=200\n")
    write(tmp_path / "data.dat", b"\x00" * 8)
    detected = detect_format(tmp_path / pick)
    assert detected.format.id == "persyst"
    assert detected.path == layout
    assert [p.name for p in detected.files] == ["rec.lay", "data.dat"]


def test_dat_without_owner(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedFormatError, match="neither Persyst"):
        detect_format(write(tmp_path / "rec.dat", b"\x00"))
