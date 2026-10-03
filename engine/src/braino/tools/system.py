"""Tools that describe the BrainO installation itself."""

from __future__ import annotations

import platform
import sys

from pydantic import BaseModel, Field

from braino import __version__
from braino.tools.base import Risk
from braino.tools.registry import tool


class SystemInfoInput(BaseModel):
    pass


class SystemInfoOutput(BaseModel):
    braino_version: str = Field(description="Installed BrainO engine version")
    python_version: str
    platform: str = Field(description="Operating system and architecture")


@tool(name="system.info", risk=Risk.READ)
def system_info(params: SystemInfoInput) -> SystemInfoOutput:
    """Report the BrainO version, Python version and operating system."""
    return SystemInfoOutput(
        braino_version=__version__,
        python_version=sys.version.split()[0],
        platform=f"{platform.system()} {platform.machine()}",
    )
