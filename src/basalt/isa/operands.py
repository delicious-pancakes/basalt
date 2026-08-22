# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Taking a composite operand field apart.

The prober groups bits by which operand slot they move, which is the right
granularity for reading an instruction and too coarse for writing one. A
constant-bank reference is a single slot holding three separate things:

    LDC R2, c[0x0][0x380]
                ^bank ^offset

and a descriptor load holds three more:

    LDG.E R0, desc[UR4][R2.64+0x8]
                   ^desc ^base ^offset

Assembling `c[0x0][0x37c]` from a form harvested as `c[0x0][0x380]` needs to know
which bits inside that 29-bit field hold the offset. Writing the whole field as
one number would be guessing, and guessing here produces an instruction that
loads from the wrong address and disassembles as though it were fine.

The information is already in the harvest. The prober flips each bit and records
the operand text before and after, so the sub-role of a bit is simply which part
of the bracket expression moved. This classifies that, and costs no extra calls
to anything.

What it does not do is invent a role for a bit whose effect it cannot read. Those
stay unclassified, and an assembler that meets one refuses rather than writes.
"""

from __future__ import annotations

import re

__all__ = ["SubRole", "classify_bit", "subfields"]

# `c[0x0][0x380]`, `cx[UR4][0x10]`, and the register-offset forms of both
_CONSTANT = re.compile(r"\b(c|cx)\[([^\]]*)\]\[([^\]]*)\]")
# `desc[UR4][R2.64+0x8]`
_DESCRIPTOR = re.compile(r"\bdesc\[([^\]]*)\]\[([^\]]*)\]")
_HEX = re.compile(r"^-?0[xX][0-9a-fA-F]+$")


class SubRole:
    """What a bit inside a composite operand field controls."""

    BANK = "bank"  # which constant bank
    OFFSET = "offset"  # the immediate displacement
    BASE = "base"  # the address register
    DESCRIPTOR = "descriptor"  # the uniform register naming the descriptor
    NEGATE = "negate"  # the `-` on `-R0`
    NOT = "not"  # the `~` on `~R5`
    ABSOLUTE = "absolute"  # the bars on `|R3|`
    INVERT = "invert"  # the `!` on `!P0`
    SUFFIX = "suffix"  # the `.ROW` on `R12.ROW`, which is not part of the number
    VALUE = "value"  # the register number or immediate the modifiers apply to
    WHOLE = "whole"  # the field is not composite; the bit is the value
    UNKNOWN = "unknown"  # the effect was not readable


# a modifier is one bit, and it sits nowhere near the operand it modifies, so
# the prober groups the two together and this splits them (finding 14)
_MODIFIERS = {"-": SubRole.NEGATE, "~": SubRole.NOT, "!": SubRole.INVERT}
_MODIFIER_ROLES = frozenset({*_MODIFIERS.values(), SubRole.ABSOLUTE, SubRole.SUFFIX})


def _parts(text: str) -> tuple[str, ...] | None:
    """The components of the first bracket expression, or None if there is none."""
    if (m := _DESCRIPTOR.search(text)) is not None:
        inner = m.group(2)
        base, _, offset = inner.partition("+")
        return ("desc", m.group(1), base.strip(), offset.strip())
    if (m := _CONSTANT.search(text)) is not None:
        inner = m.group(3)
        if "+" in inner:
            base, _, offset = inner.partition("+")
            return ("const", m.group(2), base.strip(), offset.strip())
        return ("const", m.group(2), "", inner.strip())
    return None


def _operands(text: str) -> list[str]:
    """Split operand text on commas that are not inside brackets."""
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


def _modifier_change(before: str, after: str) -> str | None:
    """The modifier a bit toggles, if that is all it does.

    Returns None when the two texts differ in some other way, so a bit that
    changes a register number is not mistaken for one that negates it.
    """
    left, right = _operands(before), _operands(after)
    if len(left) != len(right):
        return None
    differing = [(a, b) for a, b in zip(left, right, strict=True) if a != b]
    if len(differing) != 1:
        return None

    a, b = differing[0]
    for symbol, role in _MODIFIERS.items():
        if a == f"{symbol}{b}" or b == f"{symbol}{a}":
            return role
    if a == f"|{b}|" or b == f"|{a}|":
        return SubRole.ABSOLUTE
    # `R5.COL` against `R5.ROW` is one operand with a selector attached, and the
    # bit that moves it is not part of the register number: reading it as one
    # makes `IMMA`'s second source read 261 where the text says 5. Bracket
    # operands are excluded because a dot inside one is an access width, and
    # `desc[UR4][R2.64]` against `desc[UR4][R2.64+0x8]` is an offset.
    plain = "[" not in a and "[" not in b
    if plain and a.split(".")[0] == b.split(".")[0] and a.split(".")[1:] != b.split(".")[1:]:
        return SubRole.SUFFIX
    return None


def classify_bit(before: str, after: str) -> str:
    """Which part of a composite operand a single bit controls.

    `before` and `after` are the operand text with the bit clear and set. When
    neither is a bracket expression the field is not composite and the bit is
    just part of the value.
    """
    if (modifier := _modifier_change(before, after)) is not None:
        return modifier
    left, right = _parts(before), _parts(after)
    if left is None and right is None:
        # Exactly one operand moving is what "this bit is part of that value"
        # looks like. A bit that moves two of them, or none, is doing something
        # this cannot read, and reading it as the value would write a register
        # number over whatever else it controls.
        differing = sum(
            1 for a, b in zip(_operands(before), _operands(after), strict=False) if a != b
        )
        same_count = len(_operands(before)) == len(_operands(after))
        return SubRole.WHOLE if same_count and differing == 1 else SubRole.UNKNOWN
    if left is None or right is None or left[0] != right[0]:
        # the bracket appeared, vanished or changed kind, which is a structural
        # change rather than a field within one
        return SubRole.UNKNOWN

    kind, first, base, offset = left
    _, r_first, r_base, r_offset = right
    changed = [
        role
        for role, a, b in (
            (SubRole.DESCRIPTOR if kind == "desc" else SubRole.BANK, first, r_first),
            (SubRole.BASE, base, r_base),
            (SubRole.OFFSET, offset or "0x0", r_offset or "0x0"),
        )
        if a != b
    ]
    if len(changed) != 1:
        return SubRole.UNKNOWN
    role = changed[0]
    # An absent displacement is a displacement of zero, not a different shape:
    # `desc[UR4][R2.64]` and `desc[UR4][R2.64+0x8]` are the same operand with
    # different offsets, and treating the first as having no offset field loses
    # every offset bit on every descriptor load in the database.
    if role is SubRole.OFFSET:
        left_offset = offset or "0x0"
        right_offset = r_offset or "0x0"
        if not (_HEX.match(left_offset) and _HEX.match(right_offset)):
            return SubRole.UNKNOWN
    return role


def subfields(observations) -> dict[str, tuple[int, ...]]:
    """Group an operand field's bits by what part of the operand each controls.

    `observations` is any iterable of objects carrying `bit`, `before` and
    `after`, which is what the prober already produces.
    """
    grouped: dict[str, list[int]] = {}
    plain: list[int] = []
    unreadable = 0
    for observation in observations:
        # a bit whose effect could not be read is left out, not guessed at
        role = classify_bit(observation.before, observation.after)
        if role is SubRole.UNKNOWN:
            unreadable += 1
            continue
        if role is SubRole.WHOLE:
            plain.append(observation.bit)
            continue
        grouped.setdefault(role, []).append(observation.bit)

    # only worth splitting when a modifier is present, and refused outright if
    # any bit was unreadable: a truncated value is a different register
    if plain and _MODIFIER_ROLES.intersection(grouped):
        if unreadable:
            return {}
        grouped[SubRole.VALUE] = plain
    return {role: tuple(sorted(bits)) for role, bits in sorted(grouped.items())}
