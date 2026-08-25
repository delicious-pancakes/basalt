# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Architecture identities and the boundary of Basalt's measured models.

Raw ``nvdisasm`` can decode several NVIDIA ISAs.  That does not make Basalt's
sm_120 control layout, latency data, or observed-stall tables valid for those
ISAs.  Keep that distinction in one place so a command cannot cross it merely
by accepting a different ``--arch`` spelling.
"""

from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "ArchitectureError",
    "ArchitectureIdentity",
    "architecture_identity",
    "artifact_architecture",
    "cubin_architecture",
    "require_architecture_match",
    "require_control_model",
]


class ArchitectureError(ValueError):
    """An artifact or operation crossed architecture authority."""


_ARCH = re.compile(r"^(?:sm_?)?(?P<code>[0-9]+)(?P<suffix>[af]?)$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ArchitectureIdentity:
    """Canonical spelling and compatibility family for one target."""

    canonical: str
    family: str

    @property
    def has_measured_control_model(self) -> bool:
        return self.family == "sm_120" and self.canonical in {
            "sm_120",
            "sm_120a",
            "sm_120f",
        }


def architecture_identity(arch: str) -> ArchitectureIdentity:
    """Normalise ptxas/nvdisasm spellings without broadening support."""
    text = arch.strip()
    match = _ARCH.fullmatch(text)
    if match is None:
        raise ArchitectureError(f"unrecognised NVIDIA architecture spelling: {arch!r}")
    code = match.group("code")
    suffix = match.group("suffix").lower()
    canonical = f"sm_{code}{suffix}"
    return ArchitectureIdentity(canonical=canonical, family=f"sm_{code}")


def require_control_model(arch: str, operation: str) -> ArchitectureIdentity:
    """Require the measured control layout used by mutation and hazard tools."""
    identity = architecture_identity(arch)
    if not identity.has_measured_control_model:
        raise ArchitectureError(
            f"{operation} has no measured control model for {identity.canonical}; "
            "Basalt currently supports sm_120, sm_120a, and sm_120f control words only"
        )
    return identity


def require_architecture_match(expected: str, observed: str, label: str) -> None:
    """Require two architecture records to name the same compatibility family."""
    want = architecture_identity(expected)
    got = architecture_identity(observed)
    if want.family != got.family:
        raise ArchitectureError(
            f"{label} architecture {got.canonical} does not match requested {want.canonical}"
        )


def artifact_architecture(path: Path) -> str:
    """Read the mandatory ``arch`` identity from a JSON evidence artifact."""
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ArchitectureError(f"cannot read {path} architecture: {exc}") from exc
    arch = raw.get("arch") if isinstance(raw, dict) else None
    if not isinstance(arch, str) or not arch.strip():
        raise ArchitectureError(f"{path} has no architecture identity")
    return architecture_identity(arch).canonical


# CUDA cubins are ELF64.  Toolkits through 12.x put the SM code in e_flags'
# low byte; CUDA 13 moved it to byte 1.  This list and two-position rule are
# intentionally conservative: an unknown flag is an authority failure, not an
# invitation to guess from the requested command-line target.
_KNOWN_SM_CODES = frozenset(
    {
        10,
        11,
        12,
        13,
        20,
        21,
        30,
        32,
        35,
        37,
        50,
        52,
        53,
        60,
        61,
        62,
        70,
        72,
        75,
        80,
        86,
        87,
        89,
        90,
        100,
        101,
        102,
        103,
        120,
    }
)


def cubin_architecture(path: Path) -> str:
    """Read a cubin's SM family from its ELF header without running a tool."""
    try:
        header = path.read_bytes()[:0x34]
    except OSError as exc:
        raise ArchitectureError(f"cannot read cubin architecture from {path}: {exc}") from exc
    if len(header) < 0x34 or header[:4] != b"\x7fELF":
        raise ArchitectureError(f"{path} is not an ELF cubin")
    if header[4] != 2 or header[5] != 1:
        raise ArchitectureError(f"{path} is not a little-endian ELF64 cubin")
    flags = struct.unpack_from("<I", header, 0x30)[0]
    low = flags & 0xFF
    byte1 = (flags >> 8) & 0xFF
    code = low if low in _KNOWN_SM_CODES else byte1 if byte1 in _KNOWN_SM_CODES else None
    if code is None:
        raise ArchitectureError(f"{path} has unsupported CUDA ELF flags {flags:#010x}")
    return f"sm_{code}"
