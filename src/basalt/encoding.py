# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""The 128-bit sm_120 instruction word.

Every Blackwell instruction is exactly 16 bytes, stored little-endian as two
64-bit words. nvdisasm prints them as a pair:

    /*0050*/  IADD R5, R5, 0x2a ;   /* 0x0000002a05057835 */
                                    /* 0x000fca00078e0000 */

The first printed word is bits 0..63, the second is bits 64..127. We keep the
whole thing as one Python int because the fields we care about straddle the
64-bit boundary and splitting them just invites off-by-64 bugs.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

__all__ = ["Word", "BitField", "CONTROL_FIELDS", "popcount", "bit_diff"]

WORD_BITS = 128
WORD_BYTES = 16


@dataclass(frozen=True, slots=True)
class BitField:
    """A named, contiguous run of bits inside the instruction word."""

    name: str
    lo: int
    width: int
    note: str = ""

    @property
    def hi(self) -> int:
        """Inclusive high bit index."""
        return self.lo + self.width - 1

    @property
    def mask(self) -> int:
        return ((1 << self.width) - 1) << self.lo

    def get(self, word: int) -> int:
        return (word >> self.lo) & ((1 << self.width) - 1)

    def set(self, word: int, value: int) -> int:
        if not 0 <= value < (1 << self.width):
            raise ValueError(
                f"{value:#x} does not fit in {self.width}-bit field {self.name}"
            )
        return (word & ~self.mask) | (value << self.lo)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        span = f"{self.hi}:{self.lo}" if self.width > 1 else f"{self.lo}"
        return f"BitField({self.name} @ {span})"


# The control section of the word. Layout is from Huerta et al. on Ampere,
# re-measured on sm_120 by the probe suite; see docs/control-bits.md for the
# experiments that pin each field. These are declared here rather than loaded
# from the ISA database because the assembler needs them before any database
# exists, and because they are architectural rather than per-instruction.
CONTROL_FIELDS: tuple[BitField, ...] = (
    BitField("stall", 105, 4, "cycles to stall before issuing the next instruction"),
    BitField("yield_", 109, 1, "hint that the warp scheduler may switch warps"),
    BitField("write_barrier", 110, 3, "scoreboard index to signal on write-back, 7 = none"),
    BitField("read_barrier", 113, 3, "scoreboard index to signal on operand read, 7 = none"),
    BitField("wait_mask", 116, 6, "bitmask of scoreboards to wait on before issuing"),
    BitField("reuse", 122, 4, "operand reuse-cache flags, one per source slot"),
)

_CONTROL_BY_NAME = {f.name: f for f in CONTROL_FIELDS}

# 7 in a 3-bit barrier field means "do not signal". Named because the literal
# shows up in scheduling decisions constantly and 7 reads as a magic number.
NO_BARRIER = 0b111


@dataclass(frozen=True, slots=True)
class Word:
    """One 128-bit instruction, with convenience access to the control bits."""

    value: int

    def __post_init__(self) -> None:
        if not 0 <= self.value < (1 << WORD_BITS):
            raise ValueError(f"{self.value:#x} is not a 128-bit value")

    # ---- construction -------------------------------------------------

    @classmethod
    def from_halves(cls, lo: int, hi: int) -> "Word":
        """Build from the two 64-bit words as nvdisasm prints them."""
        return cls((hi << 64) | lo)

    @classmethod
    def from_bytes(cls, raw: bytes) -> "Word":
        if len(raw) != WORD_BYTES:
            raise ValueError(f"expected {WORD_BYTES} bytes, got {len(raw)}")
        lo, hi = struct.unpack("<QQ", raw)
        return cls.from_halves(lo, hi)

    # ---- serialisation ------------------------------------------------

    @property
    def lo(self) -> int:
        return self.value & 0xFFFF_FFFF_FFFF_FFFF

    @property
    def hi(self) -> int:
        return self.value >> 64

    def to_bytes(self) -> bytes:
        return struct.pack("<QQ", self.lo, self.hi)

    def __str__(self) -> str:
        return f"{self.hi:016x}{self.lo:016x}"

    # ---- field access -------------------------------------------------

    def field(self, name: str) -> int:
        return _CONTROL_BY_NAME[name].get(self.value)

    def with_field(self, name: str, value: int) -> "Word":
        return Word(_CONTROL_BY_NAME[name].set(self.value, value))

    @property
    def control(self) -> dict[str, int]:
        return {f.name: f.get(self.value) for f in CONTROL_FIELDS}

    @property
    def payload(self) -> int:
        """The word with every control field zeroed.

        Two instructions with the same payload differ only in scheduling, which
        is exactly the equivalence class the opcode harvester wants to collapse.
        """
        masked = self.value
        for f in CONTROL_FIELDS:
            masked &= ~f.mask
        return masked


def popcount(value: int) -> int:
    return bin(value).count("1")


def bit_diff(a: int, b: int) -> list[int]:
    """Indices of bits that differ between two words, low to high.

    The workhorse of the probe: mutate one input field, diff the encodings, and
    the returned indices are the bits that field actually occupies.
    """
    x = a ^ b
    return [i for i in range(WORD_BITS) if (x >> i) & 1]
