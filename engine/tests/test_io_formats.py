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


def mff(folder: Path) -> Path:
    write(folder / "info.xml", "<fileInfo/>")
    write(folder / "signal1.bin", b"\x00" * 8)
    write(folder / "Contents" / "PkgInfo", "????????")
    return folder


@pytest.mark.parametrize("pick", ["", "signal1.bin", "Contents/PkgInfo"])
def test_mff_lists_the_files_inside_the_folder(tmp_path: Path, pick: str) -> None:
    folder = mff(tmp_path / "rec.mff")
    detected = detect_format(folder / pick if pick else folder)
    assert detected.format.id == "egi_mff"
    assert detected.path == folder
    assert [p.relative_to(folder).as_posix() for p in detected.files] == [
        "Contents/PkgInfo",
        "info.xml",
        "signal1.bin",
    ]


def test_mff_without_signal(tmp_path: Path) -> None:
    folder = mff(tmp_path / "rec.mff")
    (folder / "signal1.bin").unlink()
    with pytest.raises(MissingFileError, match=r"signal1\.bin"):
        detect_format(folder)


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


def eeglab_set(path: Path, datfile: str, *, wrapped: bool = True) -> Path:
    """A minimal EEGLAB .set whose data is in ``datfile``, saved with or without the ``EEG``
    struct around its fields (EEGLAB has written both)."""
    scipy_io = pytest.importorskip("scipy.io")
    fields = {"nbchan": 1, "srate": 100.0, "datfile": datfile}
    scipy_io.savemat(path, {"EEG": fields} if wrapped else fields, appendmat=False)
    return path


@pytest.mark.parametrize("wrapped", [True, False])
def test_eeglab_renamed_data_file(tmp_path: Path, wrapped: bool) -> None:
    header = eeglab_set(tmp_path / "rec.set", "other_name.fdt", wrapped=wrapped)
    write(tmp_path / "other_name.fdt", b"\x00" * 8)
    for pick in ("rec.set", "other_name.fdt"):
        detected = detect_format(tmp_path / pick)
        assert detected.path == header
        assert [p.name for p in detected.files] == ["rec.set", "other_name.fdt"]


def test_eeglab_orphan_fdt(tmp_path: Path) -> None:
    eeglab_set(tmp_path / "rec.set", "rec_data.fdt")
    with pytest.raises(MissingFileError, match=r"orphan\.fdt"):
        detect_format(write(tmp_path / "orphan.fdt", b"\x00"))


@pytest.mark.parametrize("pick", ["rec_raw.fif", "rec_raw-1.fif", "rec_raw-2.fif"])
def test_split_fif_neuromag_naming(tmp_path: Path, pick: str) -> None:
    for name in ("rec_raw.fif", "rec_raw-1.fif", "rec_raw-2.fif", "rec_raw-4.fif"):
        write(tmp_path / name, b"\x00" * 16)
    detected = detect_format(tmp_path / pick)
    assert detected.path == tmp_path / "rec_raw.fif"
    # rec_raw-4.fif is not a part: rec_raw-3.fif is missing.
    assert [p.name for p in detected.files] == ["rec_raw.fif", "rec_raw-1.fif", "rec_raw-2.fif"]


@pytest.mark.parametrize("pick", ["01", "02"])
def test_split_fif_bids_naming(tmp_path: Path, pick: str) -> None:
    for n in ("01", "02"):
        write(tmp_path / f"sub-01_task-rest_split-{n}_meg.fif", b"\x00" * 16)
    detected = detect_format(tmp_path / f"sub-01_task-rest_split-{pick}_meg.fif")
    assert [p.name for p in detected.files] == [
        "sub-01_task-rest_split-01_meg.fif",
        "sub-01_task-rest_split-02_meg.fif",
    ]


def test_fif_name_ending_in_a_number_is_not_a_part(tmp_path: Path) -> None:
    path = write(tmp_path / "session-2.fif", b"\x00" * 16)
    assert detect_format(path).files == (path,)


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
