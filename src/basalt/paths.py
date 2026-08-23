# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Where the measured data lives, so an installed copy can find it.

The latency model and the mined stall table are not documentation, they are the
product: without them the checker falls back to assumed numbers and reports
findings it cannot ground. They used to be read from `data/` relative to the
working directory, which meant a wheel installed anywhere else silently used the
weaker model and then alleged hazards against clean code.

They ship inside the package now, and these are the paths everything reads.
Regeneration writes to the same files, so there is one copy and it cannot drift.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["DATA", "ISA_DATABASE", "LATENCIES", "OBSERVED_STALLS", "isa_database"]

DATA = Path(__file__).resolve().parent / "data"

ISA_DATABASE = DATA / "isa" / "sm_120a.json"
LATENCIES = DATA / "latency" / "rtx-5070-ti.json"
OBSERVED_STALLS = DATA / "latency" / "observed-stalls-sm120a.json"


def isa_database(arch: str = "sm_120a") -> Path:
    """The committed instruction database for one architecture."""
    return DATA / "isa" / f"{arch}.json"
