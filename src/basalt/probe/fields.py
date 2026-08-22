# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Inferring what each bit of an instruction does, by changing it and looking.

Harvesting tells us which encodings exist. It does not tell us how they are
built, and an assembler needs the second thing: given `IADD R7, R3, 0x2a`, which
bits hold 7, which hold 3, and which hold 0x2a.

The method is differential. Take a known-good word, flip one bit, ask nvdisasm
to decode the result, and diff the printed text against the original. A bit that
moves the destination register is a destination bit. A bit that changes the
opcode is a selector. A bit that changes nothing observable is either reserved
or genuinely ignored, and the two are worth distinguishing.

This is only possible because nvdisasm decodes arbitrary words rather than only
words ptxas produced. Every mutation is a question the decoder answers directly,
so the ISA stops being something to guess at and becomes something to measure.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from ..disasm import Instruction, decode_words
from ..encoding import CONTROL_FIELDS, WORD_BITS, Word
from ..toolchain import Toolchain

__all__ = ["BitObservation", "BitRole", "FieldMap", "infer_fields", "probe_word"]

_CONTROL_BITS = {b for f in CONTROL_FIELDS for b in range(f.lo, f.lo + f.width)}

# operand tokens we care about telling apart
_REG = re.compile(r"\b(?:UR|R|P|UP|SR_|SB)\w*", re.IGNORECASE)
_IMM = re.compile(r"\b0x[0-9a-fA-F]+\b")


class BitRole(StrEnum):
    """What flipping a single bit was observed to do."""

    OPCODE = "opcode"  # the mnemonic changed
    MODIFIER = "modifier"  # same opcode, different suffixes
    OPERAND = "operand"  # same mnemonic, different operand text
    PREDICATE = "predicate"  # guard predicate changed
    CONTROL = "control"  # inside the known scheduling section
    INVALID = "invalid"  # decoder rejected the mutation
    INERT = "inert"  # nothing observable changed


@dataclass(frozen=True, slots=True)
class BitObservation:
    """The result of flipping one bit of one word."""

    bit: int
    role: BitRole
    before: str
    after: str

    @property
    def changed_operand_index(self) -> int | None:
        """Which comma-separated operand slot moved, if exactly one did."""
        a = [t.strip() for t in self.before.split(",")]
        b = [t.strip() for t in self.after.split(",")]
        if len(a) != len(b):
            return None
        moved = [i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y]
        return moved[0] if len(moved) == 1 else None


@dataclass
class FieldMap:
    """Inferred bit roles for one instruction form."""

    mnemonic: str
    base_encoding: str
    observations: list[BitObservation] = field(default_factory=list)

    def bits_with_role(self, role: BitRole) -> list[int]:
        return sorted(o.bit for o in self.observations if o.role is role)

    def runs(self, role: BitRole) -> list[tuple[int, int]]:
        """Contiguous bit runs for a role, as (lo, width).

        Fields are contiguous far more often than not, so collapsing to runs
        turns a scatter of bit indices into something that reads like a layout.
        """
        bits = self.bits_with_role(role)
        out: list[tuple[int, int]] = []
        for b in bits:
            if out and b == out[-1][0] + out[-1][1]:
                lo, width = out[-1]
                out[-1] = (lo, width + 1)
            else:
                out.append((b, 1))
        return out

    def operand_fields(self) -> dict[int, list[int]]:
        """Bits grouped by which operand slot they were seen to move."""
        groups: dict[int, list[int]] = defaultdict(list)
        for o in self.observations:
            if o.role is BitRole.OPERAND and (idx := o.changed_operand_index) is not None:
                groups[idx].append(o.bit)
        return {k: sorted(v) for k, v in sorted(groups.items())}

    def summary(self) -> str:
        parts = [
            f"{role.value}={len(self.bits_with_role(role))}"
            for role in BitRole
            if self.bits_with_role(role)
        ]
        return f"{self.mnemonic:<28} " + "  ".join(parts)


def _classify(base: Instruction, mutant: Instruction | None, bit: int) -> BitRole:
    if bit in _CONTROL_BITS:
        return BitRole.CONTROL
    if mutant is None or not mutant.is_valid:
        return BitRole.INVALID
    if mutant.text == base.text:
        return BitRole.INERT
    if mutant.opcode != base.opcode:
        return BitRole.OPCODE
    if mutant.modifiers != base.modifiers:
        return BitRole.MODIFIER

    # a leading @P guard is printed before the mnemonic, so a change there shows
    # up as a difference in the text head rather than in the operand list
    if mutant.text.startswith("@") != base.text.startswith("@"):
        return BitRole.PREDICATE
    return BitRole.OPERAND


def _operands_of(text: str) -> list[str]:
    """The operand list of a decoded instruction, split on top-level commas."""
    parts: list[str] = []
    current = ""
    depth = 0
    for char in text:
        if char in "[({":
            depth += 1
        elif char in "])}":
            depth -= 1
        if char == "," and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += char
    if current.strip():
        parts.append(current.strip())
    return parts


def _write_bits(word: int, bits: Sequence[int], value: int) -> int:
    for position, bit in enumerate(bits):
        word &= ~(1 << bit)
        word |= ((value >> position) & 1) << bit
    return word


def _promote_rendered_operands(
    tc: Toolchain, base: Word, origin, fmap: FieldMap, arch: str
) -> None:
    """Recover an operand field the disassembler hid behind a suffix.

    `IMAD R22, R2, 0x10, RZ` prints as `IMAD.SHL.U32` because the multiplier is a
    power of two, and as `IMAD.U32` the moment it is not. Every flip of an
    immediate bit therefore changes the suffix, the probe records those bits as
    modifier bits, and the immediate ends up with no field at all. The assembler
    then refuses every `IMAD.SHL.U32` in the corpus for want of anywhere to put
    its operand, which is the largest single group of refusals there was.

    Nothing here is assumed. The candidates are modifier bits that also moved
    exactly one operand, and they are kept only if writing values through them
    reproduces those values exactly and leaves every other operand alone. A bit
    that really does control a modifier fails that check and stays where it was.
    """
    candidates: dict[int, list[int]] = defaultdict(list)
    for observation in fmap.observations:
        if observation.role is not BitRole.MODIFIER:
            continue
        if (index := observation.changed_operand_index) is not None:
            candidates[index].append(observation.bit)
    if not candidates:
        return

    reference = _operands_of(origin.operands)
    promoted: set[int] = set()

    for slot, found in candidates.items():
        bits = sorted(found)
        if slot >= len(reference) or not _IMM.fullmatch(reference[slot]):
            # only immediates are recovered this way; a register that moves a
            # suffix is a genuinely different encoding rather than a rendering
            continue
        # values that fit, are distinct, and keep the top bit clear so nothing
        # turns on how the decoder prints a negative
        width = len(bits)
        probes = [v for v in (0x1, 0x2, 0x15, 0x7F) if v < (1 << (width - 1))]
        if len(probes) < 2:
            continue

        words = [Word(_write_bits(base.value, bits, v)) for v in probes]
        decoded = decode_words(tc, words, arch=arch)
        if len(decoded) != len(probes):
            continue
        pairs = zip(decoded, probes, strict=True)
        if all(_reads_back(got, reference, slot, want) for got, want in pairs):
            promoted.update(bits)

    if not promoted:
        return
    for position, observation in enumerate(fmap.observations):
        if observation.bit in promoted:
            fmap.observations[position] = BitObservation(
                observation.bit, BitRole.OPERAND, observation.before, observation.after
            )


def _reads_back(decoded, reference: list[str], slot: int, want: int) -> bool:
    """Did writing `want` into the field produce exactly `want` and nothing else?"""
    if decoded is None or not decoded.is_valid:
        return False
    got = _operands_of(decoded.operands)
    if len(got) != len(reference):
        return False
    for index, (a, b) in enumerate(zip(reference, got, strict=True)):
        if index == slot:
            try:
                if int(b, 0) != want:
                    return False
            except ValueError:
                return False
        elif a != b:
            return False
    return True


def probe_word(
    tc: Toolchain,
    base: Word,
    *,
    arch: str = "SM120a",
    skip_control: bool = True,
) -> FieldMap | None:
    """Flip every bit of one word and record what each flip did.

    All 128 mutations go through a single nvdisasm invocation. Process startup
    costs roughly three orders of magnitude more than the decode itself, so
    batching is the difference between a usable probe and an unusable one.
    """
    decoded = decode_words(tc, [base], arch=arch)
    if not decoded or decoded[0] is None or not decoded[0].is_valid:
        return None
    origin = decoded[0]

    bits = [b for b in range(WORD_BITS) if not (skip_control and b in _CONTROL_BITS)]
    mutants = [Word(base.value ^ (1 << b)) for b in bits]
    results = decode_words(tc, mutants, arch=arch)

    fmap = FieldMap(mnemonic=origin.mnemonic, base_encoding=str(base))

    for bit, mutant in zip(bits, results, strict=True):
        role = _classify(origin, mutant, bit)
        fmap.observations.append(
            BitObservation(bit, role, origin.operands, mutant.operands if mutant else "")
        )

    if skip_control:
        for bit in sorted(_CONTROL_BITS):
            fmap.observations.append(BitObservation(bit, BitRole.CONTROL, "", ""))

    fmap.observations.sort(key=lambda o: o.bit)
    _promote_rendered_operands(tc, base, origin, fmap, arch)
    return fmap


def infer_fields(
    tc: Toolchain,
    words: list[Word],
    *,
    arch: str = "SM120a",
    progress: bool = False,
) -> list[FieldMap]:
    """Probe a batch of representative encodings, one per instruction form."""
    out: list[FieldMap] = []
    for i, w in enumerate(words):
        fmap = probe_word(tc, w, arch=arch)
        if fmap is not None:
            out.append(fmap)
        if progress and (i + 1) % 25 == 0:
            print(f"  probed {i + 1}/{len(words)} forms")
    return out
