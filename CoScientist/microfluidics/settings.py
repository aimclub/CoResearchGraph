"""Connection settings of the microfluidics services.

Kept out of the core settings: only this profile talks to these servers. The
environment variables are the ones the service owners hand out; ``.env`` is
already loaded into the environment by ``CoScientist.config``.

  MCP__MICROFLUIDICS_URL / MCP__MICROFLUIDICS_API_KEY   chip CFD / rig MCP
  MCP_MICROFLUIDIC_ECONOMIC                             economics (stage 5)
  MCP_MICROFLUIDIC_CFD_3_TOOLS / MICROFLUIDIC_CFD_3_TOOLS_KEY   CFD (stage 9)

The nested ``MCP__MICROFLUIDIC_*`` spellings still override the flat ones.
Unset means the agent keeps its stub.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import CoScientist.config  # noqa: F401  (loads .env into the environment)


def _env(*names: str) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


@dataclass(frozen=True)
class MicrofluidicsSettings:
    microfluidics_url: Optional[str] = None
    microfluidics_api_key: Optional[str] = None
    microfluidic_economic_url: Optional[str] = None
    microfluidic_cfd_url: Optional[str] = None
    microfluidic_cfd_api_key: Optional[str] = None


@lru_cache(maxsize=1)
def get_microfluidics_settings() -> MicrofluidicsSettings:
    return MicrofluidicsSettings(
        microfluidics_url=_env("MCP__MICROFLUIDICS_URL"),
        microfluidics_api_key=_env("MCP__MICROFLUIDICS_API_KEY"),
        microfluidic_economic_url=_env(
            "MCP__MICROFLUIDIC_ECONOMIC_URL", "MCP_MICROFLUIDIC_ECONOMIC"),
        microfluidic_cfd_url=_env(
            "MCP__MICROFLUIDIC_CFD_URL", "MCP_MICROFLUIDIC_CFD_3_TOOLS"),
        microfluidic_cfd_api_key=_env(
            "MCP__MICROFLUIDIC_CFD_API_KEY", "MICROFLUIDIC_CFD_3_TOOLS_KEY"),
    )


__all__ = ["MicrofluidicsSettings", "get_microfluidics_settings"]
