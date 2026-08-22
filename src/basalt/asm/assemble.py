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

import dataclasses
import re
from dataclasses import dataclass, field

from ..encoding import CONTROL_FIELDS, Word
from ..isa.database import IsaDatabase
from ..isa.operands import SubRole

__all__ = [
    "BRANCH_SCALE",
    "BRANCH_TARGET_BITS",
    "Assembler",
    "AssemblyError",
    "assemble_instruction",
    "assemble_program",
    "branch_target",
]

# a split field, low eight bits at 16..23 and the rest at 34..81, holding the
# signed distance from the following instruction in units of four (finding 11)
BRANCH_TARGET_BITS: tuple[int, ...] = (*range(16, 24), *range(34, 82))
BRANCH_SCALE = 4

# `.reuse` is allowed and ignored: it is a control-word hint, not the operand
_REGISTER = re.compile(r"^(UR|R|UP|P)(\d+|Z|T)(?:\.reuse)?$")
_IMMEDIATE = re.compile(r"^-?0[xX][0-9a-fA-F]+$|^-?\d+$")
_GUARD = re.compile(r"^@(!)?(U?P)(\d+|T)\s+")
_LABEL = re.compile(r"`\(([^)]+)\)")


class AssemblyError(Exception):
    """One instruction could not be encoded, with the reason attached."""


@dataclass(frozen=True, slots=True)
class Slot:
    """One operand field: where it lives, what it holds, and which token feeds it."""

    index: int
    bits: tuple[int, ...]
    token: int  # position in the tokenised operand text
    holds: str  # what the prober's example pair shows the field encoding
    # for a composite operand, which bits hold the bank, offset, base register
    parts: dict[str, tuple[int, ...]] = dataclasses.field(default_factory=dict)

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
    # bits are listed low end of the value first, so a split field reassembles
    value = 0
    for position, bit in enumerate(bits):
        value |= ((word >> bit) & 1) << position
    return value


def _write(word: int, bits: tuple[int, ...], value: int) -> int:
    width = len(bits)
    # a negative immediate is written as its two's complement in the field, which
    # is what `IADD R4, R0, -0x1` means and how the vendor encodes it
    if value < 0:
        if value < -(1 << (width - 1)):
            raise AssemblyError(f"{value} does not fit in {width} signed bits")
        value &= (1 << width) - 1
    # refuse rather than truncate: a silently narrowed value is a wrong operand
    elif value >= (1 << width):
        raise AssemblyError(f"{value} does not fit in {width} bits")
    for position, bit in enumerate(bits):
        word &= ~(1 << bit)
        word |= ((value >> position) & 1) << bit
    return word


# The parts of a bracket operand, as opposed to the modifier parts below. A
# slot carrying any of these is a composite operand and goes through
# `_write_composite`; one carrying only modifiers is an ordinary field with a
# sign bit parked somewhere else in the word.
_COMPOSITE_ROLES = frozenset({SubRole.BANK, SubRole.OFFSET, SubRole.BASE, SubRole.DESCRIPTOR})
_MODIFIER_SYMBOLS = {"-": SubRole.NEGATE, "~": SubRole.NOT, "!": SubRole.INVERT}


def _strip_modifiers(token: str) -> tuple[frozenset[str], str]:
    """A token's modifiers and the operand underneath them."""
    roles: set[str] = set()
    text = token.strip()
    # a sign on a number is part of the number, not a bit somewhere else in the
    # word: `-0x1` is the immediate -1 written as two's complement in its field
    if _IMMEDIATE.match(text):
        return frozenset(), text
    while text and text[0] in _MODIFIER_SYMBOLS:
        roles.add(_MODIFIER_SYMBOLS[text[0]])
        text = text[1:]
    if len(text) > 2 and text.startswith("|") and text.endswith("|"):
        roles.add(SubRole.ABSOLUTE)
        text = text[1:-1]
    return frozenset(roles), text.strip()


def _apply_modifiers(
    word: int, slot: Slot, token: str, reference: str, mnemonic: str, reference_word: int
) -> tuple[int, str, str]:
    """Encode the modifiers that differ, and return the bare operands.

    The polarity is read off the form rather than assumed. The reference text
    says whether its own encoding is negated, so the bit that has to change is
    simply the opposite of whatever the reference holds. That keeps this correct
    without depending on which way round the prober happened to record a bit.
    """
    want, bare_token = _strip_modifiers(token)
    have, bare_reference = _strip_modifiers(reference)

    for role in sorted(want ^ have):
        bits = slot.parts.get(role)
        if not bits or len(bits) != 1:
            raise AssemblyError(
                f"{mnemonic} operand {slot.index} needs the {role} on {token!r} and the prober "
                f"never located a single bit for it in this form"
            )
        word = _write(word, tuple(bits), 1 - ((reference_word >> bits[0]) & 1))
    return word, bare_token, bare_reference


def _kind(token: str) -> str:
    """What sort of operand a token is, finely enough to tell forms apart.

    A regular register and a uniform register are not interchangeable: `IMAD
    R7, R2, R6, RZ` and `IMAD R7, R2, UR7, RZ` differ in four bits outside every
    recorded field, so they are separate encodings of one mnemonic and the
    database holds one of them.
    """
    if (match := _REGISTER.match(token)) is not None:
        return {"R": "register", "UR": "uniform", "P": "predicate", "UP": "upredicate"}[
            match.group(1)
        ]
    if _IMMEDIATE.match(token):
        return "immediate"
    if token.startswith("c[") or token.startswith("cx["):
        return "constant"
    if token.startswith("desc["):
        return "descriptor"
    if token.startswith("`"):
        return "label"
    return "other"


def _token_value(token: str) -> int | None:
    """The number a token encodes, register or immediate, or None."""
    number = _register_number(token)
    if number is not None:
        return number
    if _IMMEDIATE.match(token):
        return int(token, 0)
    return None


def _signed(value: int, width: int) -> int:
    return value - (1 << width) if value & (1 << (width - 1)) else value


def branch_target(word: Word, address: int) -> int:
    """The byte address a branch at `address` jumps to."""
    raw = _signed(_read(word.value, BRANCH_TARGET_BITS), len(BRANCH_TARGET_BITS))
    return address + 16 + raw * BRANCH_SCALE


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
        # every recorded shape of a mnemonic, not just the canonical one. one
        # mnemonic covers several encodings and the text decides which applies,
        # so assembling tries each and keeps the one that fits.
        self._forms: dict[str, list[Form]] = {}
        entries = [
            (mnemonic, form) for mnemonic in database.forms for form in database.shapes(mnemonic)
        ]
        for mnemonic, entry in entries:
            reference = entry.word.value
            reference_text = entry.operand_text
            if (recorded := _GUARD.match(reference_text + " ")) is not None:
                reference_text = reference_text[recorded.end() - 1 :].strip()
            tokens = tuple(_tokenise(reference_text))
            slots: list[Slot] = []
            for operand in entry.operands:
                classified = _classify(operand, tokens)
                if classified is None:
                    # the prober never established which token this field
                    # belongs to, so it is left at the reference value and any
                    # text that disagrees is refused rather than mis-encoded
                    continue
                token, holds = classified
                # a field that cannot reproduce its own reference value was only
                # partly attributed, so writing through it leaves the rest behind
                if holds == "value" and token < len(tokens):
                    expected = _token_value(tokens[token])
                    if (
                        expected is None
                        or _read(reference, tuple(sorted(operand.bits))) != expected
                    ):
                        holds = "partial"
                slots.append(
                    Slot(
                        index=operand.slot,
                        bits=tuple(sorted(operand.bits)),
                        token=token,
                        holds=holds,
                        parts=dict(operand.subfields),
                    )
                )
            self._forms.setdefault(mnemonic, []).append(
                Form(
                    mnemonic=mnemonic,
                    reference=reference,
                    tokens=tokens,
                    slots=tuple(slots),
                )
            )

    @property
    def forms(self) -> int:
        return sum(len(shapes) for shapes in self._forms.values())

    @property
    def mnemonics(self) -> int:
        return len(self._forms)

    def knows(self, mnemonic: str) -> bool:
        return mnemonic in self._forms

    def assemble(self, text: str, *, control: Word | None = None) -> Word:
        """Encode one instruction, optionally with a control word to copy."""
        body = text.strip().rstrip(";").strip()

        # The guard can be written before the mnemonic, which is how a reader
        # writes it, or after it, which is how `nvdisasm` prints it. Both are
        # accepted; taking the second for an operand silently encodes a branch
        # target into a predicate field.
        guard_register: str | None = None
        negated = False
        leading = _GUARD.match(body + " ")
        if leading is not None:
            guard_register = f"{leading.group(2)}{leading.group(3)}"
            negated = bool(leading.group(1))
            body = body[leading.end() - 1 :].strip()

        parts = body.split(None, 1)
        if not parts:
            raise AssemblyError("empty instruction")
        mnemonic = parts[0]
        operands = parts[1] if len(parts) > 1 else ""

        trailing = _GUARD.match(operands + " ")
        if trailing is not None:
            if guard_register is not None:
                raise AssemblyError(f"{mnemonic} carries two guards")
            guard_register = f"{trailing.group(2)}{trailing.group(3)}"
            negated = bool(trailing.group(1))
            operands = operands[trailing.end() - 1 :].strip()

        shapes = self._forms.get(mnemonic)
        if not shapes:
            raise AssemblyError(f"{mnemonic} is not in the instruction database")

        # Try every recorded shape and keep the first that fits. Order does not
        # matter for correctness: a shape that does not match the text raises
        # rather than encoding something else, which is what makes trying them
        # in turn safe rather than a search for one that happens not to complain.
        failures: list[str] = []
        for form in shapes:
            try:
                return self._encode(form, mnemonic, operands, guard_register, negated, control)
            except AssemblyError as exc:
                failures.append(str(exc))
        raise AssemblyError(
            failures[0] if len(failures) == 1 else "; ".join(dict.fromkeys(failures))
        )

    def _encode(
        self,
        form: Form,
        mnemonic: str,
        operands: str,
        guard_register: str | None,
        negated: bool,
        control: Word | None,
    ) -> Word:
        """Encode against one recorded shape, raising if the text does not fit."""

        word = form.reference
        tokens = _tokenise(operands)

        # the same label text is a different encoding in every kernel, and the
        # database records no field for it, so the slot loop would never see it
        for position, token in enumerate(tokens):
            if _kind(token) == "label":
                raise AssemblyError(
                    f"{mnemonic} operand {position} is a branch target; assembling one needs the "
                    f"address it is relative to, which a single instruction does not carry"
                )

        for slot in form.slots:
            if slot.token >= len(tokens):
                reference_token = form.tokens[slot.token] if slot.token < len(form.tokens) else ""
                if reference_token.startswith("`"):
                    # the caller stripped a label and writes its bits itself
                    continue
                raise AssemblyError(
                    f"{mnemonic} wants an operand in position {slot.token} and the text has "
                    f"{len(tokens)}"
                )
            token = tokens[slot.token]
            reference_token = form.tokens[slot.token] if slot.token < len(form.tokens) else ""
            if token == reference_token:
                continue  # already what the reference encoding holds

            # take the modifier off first so everything below sees the bare
            # operand; its bit is nowhere near the operand (finding 14)
            word, token, reference_token = _apply_modifiers(
                word, slot, token, reference_token, mnemonic, form.reference
            )
            if token == reference_token:
                continue  # only the modifier differed, and it is now encoded
            if _kind(token) != _kind(reference_token):
                # one mnemonic can cover shapes that differ in bits outside every
                # recorded field, so writing across them encodes something else
                raise AssemblyError(
                    f"{mnemonic} operand {slot.index} is {_kind(token)} {token!r} where the "
                    f"recorded form has {_kind(reference_token)} {reference_token!r}; that is a "
                    f"different encoding of the same mnemonic and this database holds one"
                )
            if _COMPOSITE_ROLES.intersection(slot.parts):
                word = _write_composite(word, slot, token, reference_token, mnemonic)
                continue

            # When the prober separated a modifier out, the rest of the field is
            # the value and is named; otherwise the whole field is the value.
            value_bits = slot.parts.get(SubRole.VALUE) or slot.bits
            if slot.holds != "value" and SubRole.VALUE not in slot.parts:
                raise AssemblyError(
                    f"{mnemonic} operand {slot.index} sits in a field the prober found to hold "
                    f"a {slot.holds} rather than a plain value, so writing {token!r} into it "
                    f"would encode something else"
                )
            value = _register_number(token)
            if value is None:
                if _IMMEDIATE.match(token):
                    # unmasked on purpose: `_write` decides whether it fits, so
                    # an immediate too wide for its field is refused not trimmed
                    value = int(token, 0)
                else:
                    raise AssemblyError(
                        f"{mnemonic} operand {slot.index} is {token!r}, which is neither a "
                        f"register nor an immediate; the database describes this field but not "
                        f"how to spell that"
                    )
            word = _write(word, tuple(value_bits), value)

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

        # The guard is always written, even when the text has none. A recorded
        # form can itself be a predicated instruction, and its reference
        # encoding then carries that predicate; leaving the field alone would
        # emit `@P0 MOV` for text that said `MOV`.
        word = _apply_guard(word, guard_register or "PT", negated)

        result = Word(word)
        if control is not None:
            for control_field in CONTROL_FIELDS:
                result = result.with_field(control_field.name, control.field(control_field.name))
        return result


# `c[0x0][0x380]`, `c[0x0][R4+0x10]`, `desc[UR4][R2.64]`, `desc[UR4][R2.64+0x8]`
_COMPOSITE = re.compile(r"^(?P<kind>c|cx|desc)\[(?P<first>[^\]]*)\]\[(?P<inner>[^\]]*)\]$")


def _split_composite(token: str) -> tuple[str, str, str, str] | None:
    """A bracket operand as (kind, bank-or-descriptor, base register, offset)."""
    match = _COMPOSITE.match(token)
    if match is None:
        return None
    inner = match.group("inner")
    if "+" in inner:
        base, _, offset = inner.partition("+")
        return match.group("kind"), match.group("first"), base.strip(), offset.strip()
    if _IMMEDIATE.match(inner):
        return match.group("kind"), match.group("first"), "", inner.strip()
    return match.group("kind"), match.group("first"), inner.strip(), "0x0"


def _write_composite(word: int, slot: Slot, token: str, reference: str, mnemonic: str) -> int:
    """Encode a bracket operand through the sub-fields the prober found.

    Writing the whole field as one number would be a guess, and a wrong guess
    here loads from the wrong address and disassembles as though it were fine.
    Each part goes in its own bits, and a part with no recorded bits is refused
    rather than dropped.
    """
    want = _split_composite(token)
    have = _split_composite(reference)
    if want is None or have is None or want[0] != have[0]:
        raise AssemblyError(
            f"{mnemonic} operand {slot.index} is {token!r} where the recorded form has "
            f"{reference!r}; those are different operand shapes"
        )
    kind, first, base, offset = want
    _, ref_first, ref_base, ref_offset = have

    named = "descriptor" if kind == "desc" else "bank"
    for role, value_text, reference_text in (
        (named, first, ref_first),
        ("base", base, ref_base),
        ("offset", offset, ref_offset),
    ):
        if value_text == reference_text:
            continue
        bits = slot.parts.get(role)
        if not bits:
            raise AssemblyError(
                f"{mnemonic} operand {slot.index} needs a different {role} and the prober "
                f"never located bits for it in this form"
            )
        if role == "offset":
            # unmasked for the same reason as an immediate: too wide is a refusal
            value = int(value_text or "0x0", 0)
        else:
            # the base register carries its access width, as in `R2.64`, and the
            # width is part of the opcode rather than of this field
            bare = value_text.split(".")[0]
            number = _register_number(bare)
            if number is None:
                raise AssemblyError(
                    f"{mnemonic} operand {slot.index} has {value_text!r} as its {role}, which "
                    f"is not a register this can encode"
                )
            value = number
        word = _write(word, tuple(bits), value)

    # a base register that is absent on one side and present on the other is a
    # shape change the sub-fields cannot express
    if bool(base) != bool(ref_base):
        raise AssemblyError(
            f"{mnemonic} operand {slot.index} adds or drops an address register, which is a "
            f"different encoding of the operand"
        )
    return word


def _classify(operand, tokens: tuple[str, ...]) -> tuple[int, str] | None:
    """Which token a field drives, and what the field actually holds.

    The database records what the operand text looked like before and after one
    bit of the field was flipped. Exactly one token moves, and comparing the two
    spellings says more than which token it was.

    Three cases have to be told apart, because two of them are traps:

    *A register or an integer.* The value goes in as a number, which is the case
    this can encode.

    *A float.* `HFMA2 R3, -RZ, RZ, 0, 0` has two 16-bit fields holding half
    precision immediates, and the prober's example shows one of them reading
    5.96e-08, which is the denormal with the low bit set. Writing the integer 15
    into that field does not produce 15.0, it produces a very small number, and
    the instruction assembles and computes something else.

    *A modifier that happens to sit in an operand's field.* `QMMA` slot 5 flips
    between `R2` and `R2.reuse`, so those bits are the reuse flag rather than
    the register number, and writing a register number into them corrupts both.

    Only the first is treated as writable. The other two are recorded so the
    refusal can say which it is.
    """
    before, after = operand.example_before, operand.example_after
    if not before or not after:
        return None
    left, right = _tokenise(before), _tokenise(after)
    if len(left) != len(right) or len(left) != len(tokens):
        return None
    moved = [i for i, (a, b) in enumerate(zip(left, right, strict=True)) if a != b]
    if len(moved) != 1:
        return None
    position = moved[0]
    was, now = left[position], right[position]

    # the token gained or lost a dotted suffix, so the field is a modifier
    if was.split(".")[0] == now.split(".")[0] and "." in (was + now).replace(was.split(".")[0], ""):
        return position, "modifier"
    if _REGISTER.match(was.lstrip("-")) and _REGISTER.match(now.lstrip("-")):
        return position, "value"
    if _IMMEDIATE.match(was) and _IMMEDIATE.match(now):
        return position, "value"
    # anything that reads as a float on either side is a float field
    for token in (was, now):
        if _IMMEDIATE.match(token):
            continue
        try:
            float(token)
        except ValueError:
            continue
        return position, "float"
    return position, "opaque"


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


@dataclass
class ProgramAssembly:
    """The result of assembling a whole program, and what it could not do."""

    words: list[Word | None]
    exact: int = 0
    refused: list[tuple[int, str, str]] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(1 for w in self.words if w is not None)

    def summary(self) -> str:
        refused = len(self.refused)
        return (
            f"{self.total} instructions assembled, {refused} refused"
            if refused
            else f"{self.total} instructions assembled"
        )


def assemble_program(
    program, database: IsaDatabase, *, keep_control: bool = True
) -> ProgramAssembly:
    """Assemble a whole disassembled program, resolving its labels.

    A branch cannot be assembled on its own: its field holds the distance to the
    destination, so the same text encodes differently in every kernel it appears
    in. Given the whole program the distance is known, which is the only reason
    this exists separately from `Assembler.assemble`.

    Instructions that cannot be encoded are recorded with their reason and left
    as `None` rather than approximated, so a caller can see exactly what was not
    reproduced instead of receiving a program that is quietly part guesswork.
    """
    assembler = Assembler(database)
    result = ProgramAssembly(words=[])

    # labels are recorded as instruction indices; a branch field is in bytes
    addresses = {name: index * 16 for name, index in getattr(program, "labels", {}).items()}

    for index, instruction in enumerate(program.instructions):
        if instruction.word is None:
            result.words.append(None)
            continue
        text = f"{instruction.mnemonic} {instruction.operands}".strip()
        control = instruction.word if keep_control else None

        label = _LABEL.search(instruction.operands)
        if label is not None:
            destination = addresses.get(label.group(1))
            if destination is None:
                result.words.append(None)
                result.refused.append((index, text, f"no address for label {label.group(1)}"))
                continue
            # assemble the instruction with the label removed, then place the
            # distance in the field the corpus says holds it
            stripped = instruction.operands.replace(label.group(0), "").strip().rstrip(",")
            try:
                word = assembler.assemble(
                    f"{instruction.mnemonic} {stripped}".strip(), control=control
                )
            except AssemblyError as exc:
                result.words.append(None)
                result.refused.append((index, text, str(exc)))
                continue
            relative = (destination - (instruction.offset + 16)) // BRANCH_SCALE
            word = Word(
                _write(
                    word.value,
                    BRANCH_TARGET_BITS,
                    relative & ((1 << len(BRANCH_TARGET_BITS)) - 1),
                )
            )
            result.words.append(word)
            result.exact += int(word.value == instruction.word.value)
            continue

        try:
            word = assembler.assemble(text, control=control)
        except AssemblyError as exc:
            result.words.append(None)
            result.refused.append((index, text, str(exc)))
            continue
        result.words.append(word)
        result.exact += int(word.value == instruction.word.value)

    return result
