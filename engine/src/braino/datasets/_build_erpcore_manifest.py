"""Rebuild ``erpcore_manifest.json`` from the ERP CORE OSF project.

Run only when ERP CORE changes: ``python -m braino.datasets._build_erpcore_manifest``. The
manifest pins every file's download URL, size and SHA-256, so fetching never depends on the OSF
API and every download is checked.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
from typing import Any

API = "https://api.osf.io/v2"
# OSF component of each task, and the folder in it that holds the BIDS-compatible raw data.
TASKS = {
    "P3": ("etdkz", "P3 Raw Data BIDS-Compatible"),
    "N170": ("pfde9", "N170 Raw Data BIDS-Compatible"),
}
OUT = Path(__file__).with_name("erpcore_manifest.json")


def _get(url: str) -> Any:
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.load(response)
        except OSError:
            if attempt == 4:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _list(url: str) -> list[Any]:
    """All entries of an OSF folder listing, following its pages."""
    entries: list[Any] = []
    page: str | None = url + ("&" if "?" in url else "?") + "page[size]=100"
    while page:
        data = _get(page)
        entries += data["data"]
        page = data["links"].get("next")
    return entries


def _walk(url: str, prefix: str) -> list[dict[str, Any]]:
    files = []
    for entry in _list(url):
        attributes = entry["attributes"]
        path = f"{prefix}{attributes['name']}"
        if attributes["kind"] == "folder":
            files += _walk(entry["relationships"]["files"]["links"]["related"]["href"], path + "/")
        else:
            files.append(
                {
                    "path": path,
                    "url": entry["links"]["download"],
                    "size": attributes["size"],
                    "sha256": attributes["extra"]["hashes"]["sha256"],
                }
            )
    return files


def build() -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    for task, (node, folder_name) in TASKS.items():
        [folder] = [
            e
            for e in _list(f"{API}/nodes/{node}/files/osfstorage/")
            if e["attributes"]["name"] == folder_name
        ]
        files = _walk(folder["relationships"]["files"]["links"]["related"]["href"], "")
        tasks[task] = {
            "osf_node": node,
            "folder": folder_name,
            "files": sorted(files, key=lambda f: f["path"]),
        }
        print(f"{task}: {len(files)} files, {sum(f['size'] for f in files) / 1e9:.2f} GB")
    return {
        "dataset": "ERP CORE",
        "citation": (
            "Kappenman, E. S., Farrens, J. L., Zhang, W., Stewart, A. X., & Luck, S. J. (2021). "
            "ERP CORE: An open resource for human event-related potential research. "
            "NeuroImage, 225, 117465. https://doi.org/10.1016/j.neuroimage.2020.117465"
        ),
        "license": "CC-BY-4.0",
        "source": "https://osf.io/thsqg/",
        "tasks": tasks,
    }


if __name__ == "__main__":
    OUT.write_text(json.dumps(build(), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
