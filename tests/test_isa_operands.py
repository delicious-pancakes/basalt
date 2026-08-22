# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Taking a composite operand apart, on the prober's own before/after text.

The prober records what the operand text looked like with a bit clear and with
it set. That pair is enough to say which part of a bracket expression the bit
controls, and getting it wrong is not cosmetic: an assembler that writes an
offset into the bank field loads from the wrong place and disassembles as though
nothing happened.

Pure text in, a role out. No toolchain, no GPU.
"""

from __future__ import annotations

import pytest

from basalt.isa.operands import SubRole, classify_bit, subfields


class Observation:
    """The shape the prober produces, reduced to what `subfields` reads."""

    def __init__(self, bit: int, before: str, after: str) -> None:
        self.bit = bit
        self.before = before
        self.after = after


class TestConstantBank:
    @pytest.mark.parametrize(
        ("before", "after", "role"),
        [
            ("R2, c[0x0][0x380]", "R2, c[0x0][0x384]", SubRole.OFFSET),
            ("R2, c[0x0][0x380]", "R2, c[0x1][0x380]", SubRole.BANK),
            ("R2, c[0x0][R4+0x380]", "R2, c[0x0][R5+0x380]", SubRole.BASE),
        ],
    )
    def test_each_part_is_told_apart(self, before, after, role):
        assert classify_bit(before, after) == role

    def test_a_bracket_appearing_is_not_a_field(self):
        """The operand changed shape, which is a different encoding."""
        assert classify_bit("R2, R3", "R2, c[0x0][0x380]") == SubRole.UNKNOWN

    def test_two_parts_moving_at_once_is_not_attributable(self):
        assert classify_bit("R2, c[0x0][0x380]", "R2, c[0x1][0x384]") == SubRole.UNKNOWN


class TestDescriptor:
    def test_the_descriptor_register(self):
        got = classify_bit("R0, desc[UR4][R2.64]", "R0, desc[UR6][R2.64]")
        assert got == SubRole.DESCRIPTOR

    def test_the_base_register(self):
        got = classify_bit("R0, desc[UR4][R2.64]", "R0, desc[UR4][R3.64]")
        assert got == SubRole.BASE

    def test_an_absent_displacement_is_a_displacement_of_zero(self):
        """`[R2.64]` and `[R2.64+0x8]` are one operand with different offsets.

        Reading the first as having no offset field at all loses every offset
        bit on every descriptor load in the database, which is what happened.
        """
        got = classify_bit("R0, desc[UR4][R2.64]", "R0, desc[UR4][R2.64+0x8]")
        assert got == SubRole.OFFSET


class TestNonComposite:
    def test_a_plain_operand_is_the_whole_field(self):
        assert classify_bit("R2, R3, R4", "R2, R3, R5") == SubRole.WHOLE

    def test_no_bracket_on_either_side_is_never_a_subfield(self):
        assert classify_bit("R7, 0x1", "R7, 0x3") == SubRole.WHOLE


class TestModifiers:
    """A `-` is one character of text and one bit, parked far from the operand.

    Negating source 1 of `IADD` is bit 72; the register number is bits 24:31.
    The prober groups them because both change that operand's text, and telling
    them apart is the difference between assembling `R5` from a form harvested
    as `-R0` and refusing to.
    """

    @pytest.mark.parametrize(
        ("before", "after", "role"),
        [
            ("R5, R4, -R0", "R5, -R4, -R0", SubRole.NEGATE),
            ("R5, -R4, -R0", "R5, R4, -R0", SubRole.NEGATE),
            ("R13, ~R5, R9, P0", "R13, R5, R9, P0", SubRole.NOT),
            ("R0, |R3|, R9", "R0, R3, R9", SubRole.ABSOLUTE),
            ("R2, P0, R1", "R2, !P0, R1", SubRole.INVERT),
        ],
    )
    def test_each_modifier_is_told_from_the_value(self, before, after, role):
        assert classify_bit(before, after) == role

    def test_a_register_number_moving_is_not_a_modifier(self):
        assert classify_bit("R5, R4, -R0", "R5, R5, -R0") == SubRole.WHOLE

    def test_two_operands_moving_at_once_is_not_attributable(self):
        assert classify_bit("R5, R4, -R0", "R5, -R4, R0") == SubRole.UNKNOWN

    def test_a_modifier_names_the_value_bits_beside_it(self):
        """Once part of a field is a sign, the rest stops being "the field"."""
        reference = "R5, R4, -R0"
        observations = [
            Observation(24, reference, "R5, R5, -R0"),
            Observation(25, reference, "R5, R6, -R0"),
            Observation(72, reference, "R5, -R4, -R0"),
        ]
        assert subfields(observations) == {
            SubRole.NEGATE: (72,),
            SubRole.VALUE: (24, 25),
        }

    def test_a_field_with_no_modifier_is_left_whole(self):
        """Naming the value bits of a plain field would be noise, not detail."""
        observations = [
            Observation(16, "R5, R4, R0", "R4, R4, R0"),
            Observation(17, "R5, R4, R0", "R7, R4, R0"),
        ]
        assert subfields(observations) == {}

    def test_one_unreadable_bit_cancels_the_split(self):
        """Half a register number is a different register, written in silence.

        A bracket operand can lose a bit and only lose that part with it. A
        value cannot: the bits are one number, so if any of them could not be
        read the field goes back to being whole and the assembler refuses it.
        """
        reference = "R5, R4, -R0"
        observations = [
            Observation(24, reference, "R5, R5, -R0"),
            Observation(25, reference, "R5, -R6, R0"),  # two operands moved
            Observation(72, reference, "R5, -R4, -R0"),
        ]
        assert subfields(observations) == {}


class TestGrouping:
    def test_bits_are_grouped_by_what_they_control(self):
        observations = [
            Observation(38, "R2, c[0x0][0x380]", "R2, c[0x0][0x384]"),
            Observation(39, "R2, c[0x0][0x380]", "R2, c[0x0][0x388]"),
            Observation(54, "R2, c[0x0][0x380]", "R2, c[0x1][0x380]"),
        ]
        got = subfields(observations)
        assert got == {SubRole.OFFSET: (38, 39), SubRole.BANK: (54,)}

    def test_unreadable_bits_are_left_out_rather_than_guessed(self):
        observations = [
            Observation(38, "R2, c[0x0][0x380]", "R2, c[0x0][0x384]"),
            Observation(60, "R2, c[0x0][0x380]", "R2, R3"),
        ]
        got = subfields(observations)
        assert got == {SubRole.OFFSET: (38,)}

    def test_a_non_composite_field_produces_nothing(self):
        """`WHOLE` is not a sub-field; the ordinary operand path handles it."""
        assert subfields([Observation(16, "R2, R3", "R2, R4")]) == {}
