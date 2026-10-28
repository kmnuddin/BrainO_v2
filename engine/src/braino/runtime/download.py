"""Downloading large pinned files: resumable, retried, and checked before use.

Models are gigabytes, home connections drop, and Hugging Face resets connections now and
then, so a download continues where it stopped (HTTP range requests), retries with a growing
pause, and only becomes the real file once its size and SHA-256 match the pin. Until then it is
``<name>.part``, and an interrupted download is resumed by the next call.
"""

from __future__ import annotations

import hashlib
import http.client
import os
import shutil
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from braino import __version__

CHUNK = 1 << 20
#: Free disk space to keep beyond the file itself.
DISK_MARGIN = 512 << 20

Progress = Callable[[int, int], None]


class Pinned(Protocol):
    @property
    def size(self) -> int: ...

    @property
    def sha256(self) -> str: ...


class DownloadError(RuntimeError):
    pass


def sha256_of(path: Path, progress: Progress | None = None) -> str:
    digest = hashlib.sha256()
    total = path.stat().st_size
    done = 0
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            digest.update(chunk)
            done += len(chunk)
            if progress is not None:
                progress(done, total)
    return digest.hexdigest()


def is_downloaded(target: Path, pin: Pinned, *, verify: bool = False) -> bool:
    """Whether ``target`` holds the pinned file (by size; by hash too if ``verify``)."""
    if not target.is_file() or target.stat().st_size != pin.size:
        return False
    return not verify or sha256_of(target) == pin.sha256


def download(
    url: str,
    target: Path,
    pin: Pinned,
    *,
    progress: Progress | None = None,
    retries: int = 8,
    timeout: float = 60.0,
    pause: float = 2.0,
) -> Path:
    """Download ``url`` to ``target`` and check it against ``pin``.

    Reuses ``target`` if it already has the pinned size. Raises :class:`DownloadError` if the
    disk is too full, the server keeps failing, or the file does not match its hash (the
    partial file is then deleted, so the next try starts clean).
    """
    if is_downloaded(target, pin):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    have = partial.stat().st_size if partial.is_file() else 0
    if have > pin.size:
        partial.unlink()
        have = 0
    _check_space(target.parent, pin.size - have)

    failures = 0
    while have < pin.size:
        before = have
        error: Exception | None = None
        try:
            have = _fetch(url, partial, have, pin.size, progress, timeout)
        except (OSError, http.client.HTTPException) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code in (401, 403, 404):
                raise DownloadError(f"{url}: {e.code} {e.reason}") from e
            error = e
            have = partial.stat().st_size if partial.is_file() else 0
        if have > before:
            failures = 0
            if error is None:
                continue
        failures += 1
        if failures > retries:
            raise DownloadError(
                f"{url}: download failed {failures} times in a row ({error or 'no data'}); "
                f"run the command again to resume from {have / 2**20:.0f} MiB"
            ) from error
        time.sleep(pause * failures)
    if have != pin.size:
        partial.unlink(missing_ok=True)
        raise DownloadError(f"{url}: got {have} bytes, expected {pin.size}")
    if sha256_of(partial) != pin.sha256:
        partial.unlink(missing_ok=True)
        raise DownloadError(
            f"{url}: the downloaded file does not match its pinned SHA-256; it was deleted"
        )
    os.replace(partial, target)
    return target


def _fetch(
    url: str,
    partial: Path,
    have: int,
    total: int,
    progress: Progress | None,
    timeout: float,
) -> int:
    """One request, appending to ``partial`` from byte ``have``; returns the new size."""
    request = urllib.request.Request(url, headers={"User-Agent": f"braino/{__version__}"})
    if have:
        request.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if have and response.status != 206:  # the server ignored the range: start over
            have = 0
        mode = "ab" if have else "wb"
        with partial.open(mode) as f:
            while chunk := response.read(CHUNK):
                f.write(chunk)
                have += len(chunk)
                if progress is not None:
                    progress(have, total)
                if have > total:
                    break
    return have


def _check_space(folder: Path, needed: int) -> None:
    free = shutil.disk_usage(folder).free
    if free < needed + DISK_MARGIN:
        raise DownloadError(
            f"not enough disk space in {folder}: {needed / 2**30:.1f} GB needed, "
            f"{free / 2**30:.1f} GB free (set BRAINO_DATA to use another drive)"
        )
