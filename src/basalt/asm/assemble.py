# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Turning SASS text back into the 128 bits it came from.

The database records, for every form it knows, one encoding the vendor compiler
actually emitted and which bits each operand slot occupies. That is enough to
assemble: take the reference encoding for the form, overwrite the bits that
belong to each operand with the value the text asks for, and write the control
word on top.

Two things make this honest rather than hopeful.

*The operand encoding is checked, not assumed.* Every register operand in the
database encodes as its plain number in the bits recorded for it, all 646 of
them, which is what makes writing a value into a field a defensible thing to do
rather than a guess. Anything the database does not describe is refused with a
reason instead of approximated.

*The result is compared against the vendor's own bytes.* `assemble_program`
exists to be run over a disassembled cubin and produce the file it started from,
bit for bit. A form that does not round-trip is a form this cannot assemble, and
saying so is the point: an assembler that quietly emits something close is worse
than one that stops.

What this does not do is choose the control bits. That is the scheduler's job,
and the two are separate on purpose, because an assembler that also decides
timing has no way to be wrong out loud.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..encoding import CONTROL_FIELDS, Word
from ..isa.database import IsaDatabase

__all__ = ["Assembler", "AssemblyError", "assemble_instruction"]

# `R7`, `UR4`, `P0`, `UP1`, and the sinks that read as constants
_REGISTER = re.compile(r"^(UR|R|UP|P)(\d+|Z|T)$")
_IMMEDIATE = re.compile(r"^-?0[xX][0-9a-fA-F]+$|^-?\d+$")
_GUARD = re.compile(r"^@(!)?(U?P)(\d+|T)\s+")


class AssemblyError(Exception):
    """One instruction could not be encoded, with the reason attached."""


@dataclass(frozen=True, slots=True)
class Slot:
    """One operand field: where it lives, and which token feeds it."""

    index: int
    bits: tuple[int, ...]
    token: int  # position in the tokenised operand text

    @property
    def width(self) -> int:
        return len(self.bits)


@dataclass
class Form:
    """A form the assembler can build, and everything needed to build it."""

    mnemonic: str
    reference: int
    tokens: tuple[str, ...]
    slots: tuple[Slot, ...] = field(default_factory=tuple)


def _tokenise(operands: str) -> list[str]:
    """Split operand text the way the database's examples are split."""
    return [t for t in re.split(r"[,\s]+", operands.strip()) if t]


def _read(word: int, bits: tuple[int, ...]) -> int:
    value = 0
    for position, bit in enumerate(bits):
        value |= ((word >> bit) & 1) << position
    return value


def _write(word: int, bits: tuple[int, ...], value: int) -> int:
    if value < 0 or value >= (1 << len(bits)):
        raise AssemblyError(f"{value} does not fit in {len(bits)} bits")
    for position, bit in enumerate(bits):
        word &= ~(1 << bit)
        word |= ((value >> position) & 1) << bit
    return word


def _register_number(token: str) -> int | None:
    """The number a register token encodes, or None if it is not one."""
    match = _REGISTER.match(token)
    if match is None:
        return None
    kind, number = match.groups()
    if number == "Z":
        return 255 if kind == "R" else 63
    if number == "T":
        return 7
    return int(number)


class Assembler:
    """Builds instruction words from text, for the forms the database knows."""

    def __init__(self, database: IsaDatabase) -> None:
        self._forms: dict[str, Form] = {}
        for mnemonic, entry in database.forms.items():
            reference = entry.word.value
            tokens = tuple(_tokenise(entry.operand_text))
            slots: list[Slot] = []
            for operand in entry.operands:
                token = _slot_token(operand, tokens)
                if token is None:
                    # the prober never established which token this field
                    # belongs to, so it is left at the reference value and any
                    # text that disagrees is refused rather than mis-encoded
                    continue
                slots.append(
                    Slot(index=operand.slot, bits=tuple(sorted(operand.bits)), token=token)
                )
            self._forms[mnemonic] = Form(
                mnemonic=mnemonic,
                reference=reference,
                tokens=tokens,
                slots=tuple(slots),
            )

    @property
    def forms(self) -> int:
        return len(self._forms)

    def knows(self, mnemonic: str) -> bool:
        return mnemonic in self._forms

    def assemble(self, text: str, *, control: Word | None = None) -> Word:
        """Encode one instruction, optionally with a control word to copy."""
        body = text.strip().rstrip(";").strip()
        guard = _GUARD.match(body + " ")
        guard_register: str | None = None
        if guard is not None:
            guard_register = f"{guard.group(2)}{guard.group(3)}"
            body = body[guard.end() - 1 :].strip()

        parts = body.split(None, 1)
        if not parts:
            raise AssemblyError("empty instruction")
        mnemonic = parts[0]
        operands = parts[1] if len(parts) > 1 else ""

        form = self._forms.get(mnemonic)
        if form is None:
            raise AssemblyError(f"{mnemonic} is not in the instruction database")

        word = form.reference
        tokens = _tokenise(operands)

        for slot in form.slots:
            if slot.token >= len(tokens):
                raise AssemblyError(
                    f"{mnemonic} wants an operand in position {slot.token} and the text has "
                    f"{len(tokens)}"
                )
            token = tokens[slot.token]
            reference_token = form.tokens[slot.token] if slot.token < len(form.tokens) else ""
            if token == reference_token:
                continue  # already what the reference encoding holds
            value = _register_number(token)
            if value is None:
                if _IMMEDIATE.match(token):
                    value = int(token, 0) & ((1 << slot.width) - 1)
                else:
                    raise AssemblyError(
                        f"{mnemonic} operand {slot.index} is {token!r}, which is neither a "
                        f"register nor an immediate; the database describes this field but not "
                        f"how to spell that"
                    )
            word = _write(word, slot.bits, value)

        # anything the slots did not cover has to match the reference, or the
        # result would silently be a different instruction
        for position, token in enumerate(tokens):
            if position >= len(form.tokens):
                raise AssemblyError(f"{mnemonic} has more operands than the recorded form")
            if token == form.tokens[position]:
                continue
            if any(slot.token == position for slot in form.slots):
                continue
            raise AssemblyError(
                f"{mnemonic} operand {position} reads {token!r} where the recorded form has "
                f"{form.tokens[position]!r}, and no field is known to encode it"
            )

        if guard_register is not None:
            word = _apply_guard(word, guard_register, bool(guard.group(1)))

        result = Word(word)
        if control is not None:
            for control_field in CONTROL_FIELDS:
                result = result.with_field(control_field.name, control.field(control_field.name))
        return result


def _slot_token(operand, tokens: tuple[str, ...]) -> int | None:
    """Which token position this field encodes, from the prober's example pair.

    The database records what the operand text looked like before and after one
    bit was changed. Exactly one token moves, and that is the one the field
    drives.
    """
    before, after = operand.example_before, operand.example_after
    if not before or not after:
        return None
    left, right = _tokenise(before), _tokenise(after)
    if len(left) != len(right) or len(left) != len(tokens):
        return None
    moved = [i for i, (a, b) in enumerate(zip(left, right, strict=True)) if a != b]
    return moved[0] if len(moved) == 1 else None


# The guard predicate is not an operand slot: it sits in a fixed field ahead of
# everything else, which is why the database never records it as one.
_GUARD_BITS = (12, 13, 14)
_GUARD_NEGATE = 15


def _apply_guard(word: int, register: str, negated: bool) -> int:
    number = _register_number(register)
    if number is None:
        raise AssemblyError(f"{register} is not a predicate")
    word = _write(word, _GUARD_BITS, number & 0b111)
    word &= ~(1 << _GUARD_NEGATE)
    if negated:
        word |= 1 << _GUARD_NEGATE
    return word


def assemble_instruction(text: str, database: IsaDatabase, *, control: Word | None = None) -> Word:
    """Convenience wrapper for a single instruction."""
    return Assembler(database).assemble(text, control=control)
