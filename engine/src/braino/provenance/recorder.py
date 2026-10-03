"""Records each tool call of a run: arguments, result, files read and written, environment."""

from __future__ import annotations

import hashlib
import platform
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from pydantic import BaseModel

from braino import __version__
from braino.provenance.models import FileRecord, RunRecord, ToolCallRecord
from braino.provenance.store import append_call, write_run

TRACKED_PACKAGES = (
    "autoreject",
    "mne",
    "mne-bids",
    "mne-connectivity",
    "mne-icalabel",
    "nibabel",
    "nilearn",
    "numpy",
    "pydantic",
    "pyprep",
    "scikit-learn",
    "scipy",
)
"""Libraries whose versions affect results; the installed ones are recorded for every run."""

_HASH_CHUNK = 1 << 20


def installed_versions(packages: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str]:
    versions = {}
    for name in packages:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            continue
    return versions


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class ActiveCall:
    """The tool call in progress. The tool sets :attr:`result` before the call ends."""

    tool: str
    risk: str
    arguments: BaseModel
    started_at: datetime
    inputs: list[FileRecord] = field(default_factory=list)
    output_paths: list[Path] = field(default_factory=list)
    result: BaseModel | None = None


class ProvenanceRecorder:
    """Collects the records of one run and, if the run has a dataset, writes them to disk.

    ``run_dir`` creates and returns the run's folder; it is only called once the first tool
    call finishes, so runs that call no tools leave nothing behind.
    """

    def __init__(
        self,
        run_id: str,
        dataset_root: Path | None,
        run_dir: Callable[[], Path] | None = None,
    ) -> None:
        self.run = RunRecord(
            run_id=run_id,
            started_at=_now(),
            dataset_root=str(dataset_root) if dataset_root else None,
            braino_version=__version__,
            python_version=sys.version.split()[0],
            platform=f"{platform.system()} {platform.machine()}",
            packages=installed_versions(),
        )
        self.calls: list[ToolCallRecord] = []
        self._dataset_root = dataset_root
        self._run_dir_factory = run_dir
        self._run_dir: Path | None = None
        self._active: ActiveCall | None = None
        self._hash_cache: dict[tuple[Path, int, int], str] = {}

    @contextmanager
    def tool_call(self, tool: str, risk: str, arguments: BaseModel) -> Iterator[ActiveCall]:
        """Record the tool call executed inside the ``with`` block.

        An exception raised in the block is recorded as a failed call and re-raised.
        """
        if self._active is not None:
            raise RuntimeError(
                f"cannot start {tool!r} while {self._active.tool!r} is running; "
                "tools must not call other tools"
            )
        active = ActiveCall(tool=tool, risk=risk, arguments=arguments, started_at=_now())
        self._active = active
        try:
            yield active
            outputs = [self._file_record(path) for path in active.output_paths]
        except Exception as exc:
            self._finish(active, outputs=[], error=f"{type(exc).__name__}: {exc}")
            raise
        else:
            self._finish(active, outputs=outputs, error=None)
        finally:
            self._active = None

    def record_input(self, path: Path) -> None:
        """Hash a file the current tool call reads."""
        self._require_active("record_input").inputs.append(self._file_record(path))

    def record_output(self, path: Path) -> None:
        """Declare a file the current tool call writes; it is hashed when the call ends."""
        self._require_active("record_output").output_paths.append(Path(path))

    def _require_active(self, method: str) -> ActiveCall:
        if self._active is None:
            raise RuntimeError(f"{method} can only be used while a tool is running")
        return self._active

    def _finish(self, active: ActiveCall, outputs: list[FileRecord], error: str | None) -> None:
        record = ToolCallRecord(
            run_id=self.run.run_id,
            call_index=len(self.calls) + 1,
            tool=active.tool,
            risk=active.risk,
            arguments=active.arguments.model_dump(mode="json"),
            result=active.result.model_dump(mode="json") if active.result else None,
            status="error" if error else "ok",
            error=error,
            started_at=active.started_at,
            finished_at=_now(),
            inputs=active.inputs,
            outputs=outputs,
        )
        self.calls.append(record)
        if self._run_dir_factory is None:
            return
        if self._run_dir is None:
            self._run_dir = self._run_dir_factory()
            write_run(self._run_dir, self.run)
        append_call(self._run_dir, record)

    def _file_record(self, path: Path) -> FileRecord:
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"recorded file {resolved} does not exist")
        stat = resolved.stat()
        key = (resolved, stat.st_size, stat.st_mtime_ns)
        digest = self._hash_cache.get(key)
        if digest is None:
            digest = _sha256(resolved)
            self._hash_cache[key] = digest
        return FileRecord(path=self._display_path(resolved), sha256=digest, size=stat.st_size)

    def _display_path(self, resolved: Path) -> str:
        if self._dataset_root is not None and resolved.is_relative_to(self._dataset_root):
            return resolved.relative_to(self._dataset_root).as_posix()
        return resolved.as_posix()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_HASH_CHUNK):
            h.update(chunk)
    return h.hexdigest()
