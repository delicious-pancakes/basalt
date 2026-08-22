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
    WHOLE = "whole"  # the field is not composite; the bit is the value
    UNKNOWN = "unknown"  # the effect was not readable


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


def classify_bit(before: str, after: str) -> str:
    """Which part of a composite operand a single bit controls.

    `before` and `after` are the operand text with the bit clear and set. When
    neither is a bracket expression the field is not composite and the bit is
    just part of the value.
    """
    left, right = _parts(before), _parts(after)
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
    for observation in observations:
        role = classify_bit(observation.before, observation.after)
        if role in (SubRole.WHOLE, SubRole.UNKNOWN):
            continue
        grouped.setdefault(role, []).append(observation.bit)
    return {role: tuple(sorted(bits)) for role, bits in sorted(grouped.items())}
