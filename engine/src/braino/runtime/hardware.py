"""What the user's computer has: CPU, memory and GPUs.

Detection never needs administrator rights and never fails: anything that cannot be found is
left out (a GPU that is not detected just means the CPU is used). GPUs are found with
``nvidia-smi`` when it works, the Windows registry (which also covers AMD and Intel cards),
Linux's ``/sys`` files for AMD cards, and the chip itself on Apple silicon, whose GPU shares the
main memory.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

Vendor = Literal["nvidia", "amd", "intel", "apple", "other"]

#: GPUs with less dedicated memory than this are not worth offloading a model to.
MIN_GPU_GB = 2.0


class Gpu(BaseModel):
    name: str
    vendor: Vendor
    memory_gb: float = Field(description="Dedicated memory; on Apple silicon, the shared memory")
    unified: bool = Field(False, description="Shares the main memory (Apple silicon)")
    driver: str | None = None
    source: str = Field(description="How it was detected")


class Hardware(BaseModel):
    platform: str = Field(description="Operating system and machine, e.g. 'Windows AMD64'")
    os: Literal["windows", "linux", "macos", "other"]
    arch: Literal["x64", "arm64", "other"]
    cpu: str | None
    cpu_count: int | None = Field(description="Logical processors")
    memory_gb: float | None = Field(description="Total RAM")
    gpus: list[Gpu] = []

    def best_gpu(self) -> Gpu | None:
        """The GPU a model would run on: Apple silicon's, or the dedicated GPU with the most
        memory (at least ``MIN_GPU_GB``). Integrated GPUs without their own memory don't
        count: running on them is no faster than the CPU."""
        apple = [g for g in self.gpus if g.unified]
        if apple:
            return apple[0]
        dedicated = [g for g in self.gpus if g.memory_gb >= MIN_GPU_GB]
        return max(dedicated, key=lambda g: g.memory_gb, default=None)


def detect() -> Hardware:
    """Detect the hardware of this computer."""
    system = {"Windows": "windows", "Linux": "linux", "Darwin": "macos"}.get(
        platform.system(), "other"
    )
    machine = platform.machine().lower()
    arch = (
        "x64"
        if machine in ("amd64", "x86_64")
        else "arm64"
        if machine in ("arm64", "aarch64")
        else "other"
    )
    memory = total_memory_gb()
    return Hardware(
        platform=f"{platform.system()} {platform.machine()}",
        os=system,
        arch=arch,
        cpu=_cpu_name(),
        cpu_count=os.cpu_count(),
        memory_gb=memory,
        gpus=_gpus(system, arch, memory),
    )


def vendor_of(name: str) -> Vendor:
    lowered = name.lower()
    if any(k in lowered for k in ("nvidia", "geforce", "quadro", "tesla", "rtx")):
        return "nvidia"
    if any(k in lowered for k in ("amd", "radeon", "ati ")):
        return "amd"
    if "intel" in lowered or " arc" in lowered:
        return "intel"
    if "apple" in lowered:
        return "apple"
    return "other"


def parse_nvidia_smi(text: str) -> list[Gpu]:
    """Parse ``nvidia-smi --query-gpu=name,memory.total,driver_version
    --format=csv,noheader,nounits`` (memory in MiB)."""
    gpus = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            memory = round(float(parts[1]) / 1024, 1)
        except ValueError:
            continue
        gpus.append(
            Gpu(
                name=parts[0],
                vendor="nvidia",
                memory_gb=memory,
                driver=parts[2] if len(parts) > 2 and parts[2] else None,
                source="nvidia-smi",
            )
        )
    return gpus


def total_memory_gb() -> float | None:
    try:
        if sys.platform == "win32":
            import ctypes

            class Status(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("load", ctypes.c_ulong),
                    ("total", ctypes.c_ulonglong),
                    ("available", ctypes.c_ulonglong),
                    ("total_page", ctypes.c_ulonglong),
                    ("available_page", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended", ctypes.c_ulonglong),
                ]

            status = Status()
            status.length = ctypes.sizeof(Status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return round(float(status.total) / 2**30, 1)
            return None
        if sys.platform == "darwin":
            out = _run(["sysctl", "-n", "hw.memsize"])
            return round(int(out) / 2**30, 1) if out else None
        pages = float(os.sysconf("SC_PHYS_PAGES")) * float(os.sysconf("SC_PAGE_SIZE"))
        return round(pages / 2**30, 1)
    except (OSError, ValueError, AttributeError):
        return None


def _gpus(system: str, arch: str, memory_gb: float | None) -> list[Gpu]:
    if system == "macos":
        if arch == "arm64" and memory_gb:
            name = _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or "Apple silicon"
            return [
                Gpu(name=name, vendor="apple", memory_gb=memory_gb, unified=True, source="apple")
            ]
        return []
    gpus = _nvidia_smi()
    if system == "windows":
        found = _windows_registry_gpus()
    elif system == "linux":
        found = _linux_amd_gpus()
    else:
        found = []
    if gpus:  # nvidia-smi is the better source for NVIDIA cards
        found = [g for g in found if g.vendor != "nvidia"]
    return gpus + found


def _nvidia_smi() -> list[Gpu]:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return []
    out = _run(
        [
            exe,
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    return parse_nvidia_smi(out) if out else []


def _windows_registry_gpus() -> list[Gpu]:
    if sys.platform != "win32":
        return []
    import winreg

    key_path = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    gpus: list[Gpu] = []
    try:
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path)
    except OSError:
        return []
    with root:
        index = 0
        while True:
            try:
                sub = winreg.EnumKey(root, index)
            except OSError:
                break
            index += 1
            if not sub.isdigit():
                continue
            try:
                with winreg.OpenKey(root, sub) as key:
                    name = str(winreg.QueryValueEx(key, "DriverDesc")[0])
                    raw = winreg.QueryValueEx(key, "HardwareInformation.qwMemorySize")[0]
                    try:
                        driver = str(winreg.QueryValueEx(key, "DriverVersion")[0])
                    except OSError:
                        driver = None
            except OSError:
                continue  # no dedicated memory reported: integrated or virtual adapter
            size = int.from_bytes(raw, "little") if isinstance(raw, bytes) else int(raw)
            if size <= 0:
                continue
            gpus.append(
                Gpu(
                    name=name,
                    vendor=vendor_of(name),
                    memory_gb=round(size / 2**30, 1),
                    driver=driver,
                    source="registry",
                )
            )
    return gpus


def _linux_amd_gpus() -> list[Gpu]:
    gpus = []
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
        device = card / "device"
        try:
            if (device / "vendor").read_text().strip() != "0x1002":
                continue
            size = int((device / "mem_info_vram_total").read_text().strip())
        except (OSError, ValueError):
            continue
        name_file = device / "product_name"
        name = name_file.read_text().strip() if name_file.is_file() else "AMD Radeon GPU"
        gpus.append(Gpu(name=name, vendor="amd", memory_gb=round(size / 2**30, 1), source="sysfs"))
    return gpus


def _cpu_name() -> str | None:
    try:
        if sys.platform == "win32":
            import winreg

            path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        if sys.platform == "darwin":
            return _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or None
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith(("model name", "hardware")):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _run(command: list[str]) -> str | None:
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None
