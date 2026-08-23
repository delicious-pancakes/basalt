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
# `[R3]` and `[UR4+0x400]`, matched only where no name precedes the bracket so a
# constant bank or a descriptor is never read as one of these
_MEMORY = re.compile(r"(?<![\w\]])\[([^\]]*)\]")
_HEX = re.compile(r"^-?0[xX][0-9a-fA-F]+$")
# the file a register belongs to, which is not part of its number
_REGISTER_FILE = re.compile(r"^(UR|R|UP|P)(?:\d+|Z|T)\b")


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
# a sign on a literal belongs to the literal: the `-` on `-2.875` is IEEE bit 31,
# and splitting it off writes 31 bits of a 32-bit float
_NUMERIC = re.compile(r"^(0[xX][0-9a-fA-F]|\d|\.\d|INF|QNAN|NAN)", re.IGNORECASE)


def _file(token: str) -> str | None:
    """Which register file a token names, or None when it names no register."""
    match = _REGISTER_FILE.match(token.strip())
    return match.group(1) if match else None


def _parts(text: str) -> tuple[str, ...] | None:
    """The components of the first bracket expression, or None if there is none."""
    if (m := _DESCRIPTOR.search(text)) is not None:
        inner = m.group(2)
        base, _, offset = inner.partition("+")
        return ("desc", m.group(1), base.strip(), offset.strip())
    if (m := _CONSTANT.search(text)) is not None:
        inner = m.group(3).strip()
        if "+" in inner:
            base, _, offset = inner.partition("+")
            return ("const", m.group(2), base.strip(), offset.strip())
        # `c[0x3][R0]` indexes by register; calling that an offset failed the
        # hex check and left 875 constant loads unassemblable
        if _HEX.match(inner):
            return ("const", m.group(2), "", inner)
        return ("const", m.group(2), inner, "")
    # a plain address: the register and the displacement are different bits of
    # one wide field, and unsplit the whole field reads as unattributable
    if (m := _MEMORY.search(text)) is not None:
        inner = m.group(1)
        base, _, offset = inner.partition("+")
        return ("mem", "", base.strip(), offset.strip())
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
            base = b if a == f"{symbol}{b}" else a
            return None if _NUMERIC.match(base) else role
    if a == f"|{b}|" or b == f"|{a}|":
        return SubRole.ABSOLUTE
    # a selector attached to an operand is not part of its number (finding 14).
    # the head has to be a name, or `0.5` against `0.500000059` reads as one
    head = a.split(".")[0]
    plain = "[" not in a and "[" not in b and head[:1].isalpha()
    if plain and head == b.split(".")[0] and a.split(".")[1:] != b.split(".")[1:]:
        return SubRole.SUFFIX
    return None


def classify_bit(before: str, after: str) -> str:
    """Which part of a composite operand a single bit controls.

    `before` and `after` are the operand text with the bit clear and set. When
    neither is a bracket expression the field is not composite and the bit is
    just part of the value.

    The comparison is per operand rather than per instruction. `LDGSTS.E [R7],
    desc[UR6][R2.64]` has two bracket expressions, and reading only the first
    one in the text attributes the second one's bits against the wrong operand,
    which is how every shared-memory address in the async copy came out
    unattributable.
    """
    if (modifier := _modifier_change(before, after)) is not None:
        return modifier

    operands_before, operands_after = _operands(before), _operands(after)
    if len(operands_before) != len(operands_after):
        return SubRole.UNKNOWN
    differing = [(a, b) for a, b in zip(operands_before, operands_after, strict=True) if a != b]
    # exactly one operand moving is what "this bit belongs to that value" looks
    # like; two or none is something this cannot read
    if len(differing) != 1:
        return SubRole.UNKNOWN

    left, right = _parts(differing[0][0]), _parts(differing[0][1])
    if left is None and right is None:
        return SubRole.WHOLE
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
    # a bit that swaps the register file is a selector, not part of the number:
    # `LDS.64 R8, [UR4]` has one at 91, and folding it in mis-encoded `[UR7]`
    if role in (SubRole.BASE, SubRole.DESCRIPTOR):
        pair = (base, r_base) if role is SubRole.BASE else (first, r_first)
        if _file(pair[0]) != _file(pair[1]):
            return SubRole.UNKNOWN
    # an absent displacement is a displacement of zero, not a different shape,
    # or every descriptor load loses its offset bits
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
