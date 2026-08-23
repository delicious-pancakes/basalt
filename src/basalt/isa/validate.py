# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Checking that the field maps can be used, not just read.

The prober records which bits moved an operand. That is enough to describe an
encoding and not obviously enough to build one: knowing that bits 16 to 23 move
the destination is only useful if writing 7 into them actually produces `R7`.

So this writes chosen values into each measured field and asks the decoder what
came out. A field map that survives is one an assembler can emit through. A
field map that does not is a description that happens to be wrong, which is the
failure mode a table built by observation is most prone to.

Every operand slot is exercised across several values rather than one, because a
single value can agree by accident: bits that hold a register number and bits
that hold something else both look right if the only value tried is zero.
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ..disasm import decode_words, raw_arch
from ..encoding import Word
from ..toolchain import Toolchain
from .database import InstructionForm, IsaDatabase

__all__ = ["FormValidation", "ValidationSummary", "validate_database"]

# distinct bit patterns, so a field narrower than measured shows up as a
# mismatch rather than as agreement on a value that happened to fit
PROBE_VALUES: tuple[int, ...] = (1, 2, 5, 9, 21)

# predicates too: several forms put one in the first slot, and a validator
# looking only for `R\d+` calls their field maps broken
_REGISTER = re.compile(r"\b(UR|R|UP|P)(\d+)\b")


@dataclass
class FormValidation:
    """Whether one form's operand fields behave as measured."""

    mnemonic: str
    controllable: list[int] = field(default_factory=list)
    uncontrollable: list[int] = field(default_factory=list)
    # branch targets, immediates and special-register names: controllable, but
    # not by a check that reads back a register number
    non_register: list[int] = field(default_factory=list)
    undecodable: int = 0
    detail: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.controllable) and not self.uncontrollable

    @property
    def slots(self) -> int:
        """Register slots only, since those are what this can decide."""
        return len(self.controllable) + len(self.uncontrollable)


@dataclass
class ValidationSummary:
    """The whole database's answer to "can this be assembled through"."""

    results: list[FormValidation] = field(default_factory=list)

    @property
    def usable(self) -> list[FormValidation]:
        return [r for r in self.results if r.ok]

    @property
    def broken(self) -> list[FormValidation]:
        return [r for r in self.results if r.uncontrollable]

    def summary(self) -> str:
        slots = sum(r.slots for r in self.results)
        good = sum(len(r.controllable) for r in self.results)
        other = sum(len(r.non_register) for r in self.results)
        if not slots:
            return "nothing to validate"
        return (
            f"{len(self.results)} forms, {slots} register slots, "
            f"{good} controllable ({good / slots:.1%}), "
            f"{other} slots holding something other than a register"
        )


def _split_operands(text: str) -> list[str]:
    """Split on commas outside brackets, matching how the prober counted slots.

    A descriptor operand such as `desc[UR4][R2.64]` can contain commas, and
    splitting naively would shift every slot after it.
    """
    out: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in text:
        if ch in "[{(":
            depth += 1
        elif ch in "]})":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if last := "".join(current).strip():
        out.append(last)
    return out


def _printed_number(text: str, slot: int) -> int | None:
    """The register number printed in one operand slot, of any register class."""
    parts = _split_operands(text)
    if slot >= len(parts):
        return None
    m = _REGISTER.search(parts[slot])
    return int(m.group(2)) if m else None


def _write_field(word: Word, bits: tuple[int, ...], value: int) -> Word:
    """Write `value` across `bits`, lowest bit index first.

    The bits of a field are not always contiguous, so the value is spread over
    them in the order they were measured rather than shifted into a range.
    """
    result = word.value
    for position, bit in enumerate(sorted(bits)):
        if (value >> position) & 1:
            result |= 1 << bit
        else:
            result &= ~(1 << bit)
    return Word(result)


def validate_form(
    tc: Toolchain,
    form: InstructionForm,
    *,
    arch: str = "SM120a",
) -> FormValidation:
    """Write values into each operand field and see whether they come back."""
    result = FormValidation(mnemonic=form.mnemonic)
    base = form.word

    candidates: list[tuple[int, int, Word]] = []
    for operand in form.operands:
        # only a slot that prints a register in the base form can be checked by
        # reading a register back out of it
        if _printed_number(form.operand_text, operand.slot) is None:
            result.non_register.append(operand.slot)
            continue
        # a field narrower than the value cannot represent it
        usable = [v for v in PROBE_VALUES if v < (1 << len(operand.bits))]
        for value in usable:
            candidates.append((operand.slot, value, _write_field(base, operand.bits, value)))

    if not candidates:
        result.detail = (
            "no register-bearing operand slot to check"
            if result.non_register
            else "no operand fields were measured for this form"
        )
        return result

    decoded = decode_words(tc, [w for _, _, w in candidates], arch=arch)

    observed: dict[int, list[bool]] = {}
    for (slot, value, _), instr in zip(candidates, decoded, strict=True):
        if instr is None or not instr.is_valid:
            result.undecodable += 1
            continue
        if instr.mnemonic != form.mnemonic:
            # writing an operand changed the instruction, so those bits are not
            # purely an operand field
            observed.setdefault(slot, []).append(False)
            continue
        got = _printed_number(instr.operands, slot)
        observed.setdefault(slot, []).append(got == value)

    for slot, outcomes in sorted(observed.items()):
        # a slot counts as controllable only if every value written came back,
        # since one agreement out of five is chance
        (result.controllable if outcomes and all(outcomes) else result.uncontrollable).append(slot)

    return result


def validate_database(
    tc: Toolchain,
    db: IsaDatabase,
    *,
    arch: str | None = None,
    limit: int | None = None,
    progress: bool = True,
) -> ValidationSummary:
    """Validate every form's operand fields against the decoder."""
    raw = raw_arch(arch or db.arch)
    summary = ValidationSummary()
    forms = [f for f in db.forms.values() if f.operands]
    if limit is not None:
        forms = forms[:limit]

    # every form is an independent chain of decoder calls, so this scales with
    # cores rather than with the size of the database
    workers = min(32, (os.cpu_count() or 4) * 2)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(lambda form: validate_form(tc, form, arch=raw), forms):
            summary.results.append(result)
            done += 1
            if progress and done % 50 == 0:
                print(f"  validated {done}/{len(forms)} forms")

    return summary
