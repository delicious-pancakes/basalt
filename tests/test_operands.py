# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Register def/use extraction.

Every hazard the verifier can find depends on getting this right, and getting it
wrong is silent in both directions: a missed definition hides a real hazard, and
an invented one reports a hazard that is not there. So the cases below are drawn
from instruction text nvdisasm actually printed rather than from imagination.
"""

from __future__ import annotations

import pytest

from basalt.verify.operands import RegKind, RegRef, operand_access


def defs(mnemonic: str, operands: str) -> set[str]:
    return {str(r) for r in operand_access(mnemonic, operands).real_defs}


def uses(mnemonic: str, operands: str) -> set[str]:
    return {str(r) for r in operand_access(mnemonic, operands).real_uses}


class TestDirection:
    def test_arithmetic_defines_first_operand(self):
        assert defs("IADD", "R5, R5, 0x2a") == {"R5"}
        assert uses("IADD", "R5, R5, 0x2a") == {"R5"}

    def test_store_defines_nothing(self):
        assert defs("STG.E", "desc[UR4][R2.64], R7") == set()

    def test_store_reads_its_address_and_value(self):
        assert uses("STG.E", "desc[UR4][R2.64], R7") == {"R2", "R3", "R7", "UR4"}

    def test_exit_touches_nothing(self):
        assert defs("EXIT", "") == set()
        assert uses("EXIT", "") == set()

    def test_branch_defines_nothing(self):
        assert defs("BRA", "`(.L_x_0)") == set()


class TestWidth:
    def test_64_bit_load_defines_a_register_pair(self):
        assert defs("LDG.E.64", "R2, desc[UR4][R6.64]") == {"R2", "R3"}

    def test_128_bit_load_defines_a_quad(self):
        assert defs("LDG.E.128", "R4, desc[UR4][R2.64]") == {"R4", "R5", "R6", "R7"}

    def test_width_on_the_register_wins_over_the_mnemonic(self):
        """`R6.64` in an address is two registers even on a 32-bit load."""
        assert uses("LDG.E", "R2, desc[UR4][R6.64]") >= {"R6", "R7"}

    def test_narrow_types_stay_single_register(self):
        assert defs("LDG.E.U16", "R2, desc[UR4][R6.64]") == {"R2"}


class TestAddressing:
    def test_registers_inside_brackets_are_reads_even_in_the_first_slot(self):
        """A destination slot holding an address is still a read of that address."""
        access = operand_access("LDS.64", "[R2.64], R4")
        assert RegRef(RegKind.GENERAL, 2) in access.real_uses

    def test_uniform_registers_are_tracked_separately(self):
        assert "UR4" in uses("STG.E", "desc[UR4][R2.64], R7")


class TestPredicates:
    def test_guard_predicate_is_a_read(self):
        assert "P0" in uses("IADD", "@P0 R2, R3, R4")

    def test_negated_guard_predicate_is_a_read(self):
        assert "P1" in uses("IADD", "@!P1 R2, R3, R4")

    def test_setp_defines_a_predicate(self):
        assert defs("ISETP.GE.AND", "P0, PT, R2, R3, PT") == {"P0"}

    def test_setp_reads_its_comparands(self):
        assert uses("ISETP.GE.AND", "P0, PT, R2, R3, PT") == {"R2", "R3"}


class TestSinkRegisters:
    def test_rz_is_not_a_real_definition(self):
        assert defs("IADD", "RZ, R3, R4") == set()

    def test_rz_is_not_a_real_use(self):
        assert uses("IADD", "R2, RZ, R4") == {"R4"}

    def test_pt_is_not_a_real_predicate(self):
        assert "P7" not in defs("ISETP.GE.AND", "PT, PT, R2, R3, PT")

    def test_urz_is_not_a_real_use(self):
        assert "UR63" not in uses("IADD", "R2, R3, URZ")

    @pytest.mark.parametrize(
        ("kind", "number"),
        [(RegKind.GENERAL, 255), (RegKind.UNIFORM, 63), (RegKind.PREDICATE, 7)],
    )
    def test_sink_registers_are_identified(self, kind, number):
        assert RegRef(kind, number).is_sink

    def test_ordinary_registers_are_not_sinks(self):
        assert not RegRef(RegKind.GENERAL, 5).is_sink


class TestSplitting:
    def test_commas_inside_brackets_do_not_split_operands(self):
        access = operand_access("LDSM.16.M88.4", "R4, [R2+0x10]")
        assert "R4" in {str(r) for r in access.real_defs}

    def test_empty_operands_are_handled(self):
        access = operand_access("NOP", "")
        assert not access.defs and not access.uses


class TestGuard:
    """The guard is recorded apart from the other reads.

    Not a stylistic split: a guard gates issue and needs about two and a half
    times the lead of the same predicate read as data, so the two have to be
    told apart before either can be charged correctly.
    """

    def test_the_guard_is_identified_and_still_a_read(self):
        access = operand_access("IMAD", "@P1 R3, R0, R7, R5")
        assert access.guard == RegRef(RegKind.PREDICATE, 1)
        assert RegRef(RegKind.PREDICATE, 1) in access.real_uses

    def test_a_negated_guard_is_the_same_register(self):
        assert operand_access("BRA", "@!P0 `(.L_1)").guard == RegRef(RegKind.PREDICATE, 0)

    def test_no_guard_means_none(self):
        assert operand_access("IMAD", "R3, R0, R7, R5").guard is None

    def test_the_always_true_guard_is_not_a_dependency(self):
        """`@PT` reads as a constant, so nothing can ever be waiting on it."""
        assert operand_access("IMAD", "@PT R3, R0, R7, R5").guard is None

    def test_a_predicate_read_as_data_is_not_a_guard(self):
        """`SEL` reads P1 as an operand. Same register, different requirement."""
        access = operand_access("SEL", "R3, R0, R7, P1")
        assert access.guard is None
        assert RegRef(RegKind.PREDICATE, 1) in access.real_uses


class TestWideMultiply:
    """`IMAD.WIDE dst, a, b, c` is 64-bit in two of its four operands.

    It computes `a * b + c` where `a` and `b` are 32-bit and `dst` and `c` are
    register pairs. Nothing in the mnemonic distinguishes them beyond position,
    and the `.U32` some forms carry describes the factors rather than the
    result, so the ordinary suffix rule reads the whole instruction as 32-bit
    and loses the high half of both. That lost the dependency a 64-bit atomic
    has on the high word of its own product.
    """

    def test_the_destination_is_a_pair(self):
        access = operand_access("IMAD.WIDE.U32", "R6, R2, UR7, RZ")
        assert access.real_defs == {RegRef(RegKind.GENERAL, 6), RegRef(RegKind.GENERAL, 7)}

    def test_the_factors_are_not(self):
        access = operand_access("IMAD.WIDE.U32", "R6, R2, R3, RZ")
        assert RegRef(RegKind.GENERAL, 4) not in access.real_uses

    def test_the_addend_is_a_pair(self):
        access = operand_access("IMAD.WIDE.U32", "R8, R2, R13, R8")
        assert RegRef(RegKind.GENERAL, 9) in access.real_uses

    def test_a_plain_multiply_is_untouched(self):
        access = operand_access("IMAD", "R7, R2, UR7, RZ")
        assert access.real_defs == {RegRef(RegKind.GENERAL, 7)}


class TestMinMaxWritesARegister:
    """`IMNMX.S64 PT, PT, R4, R4, R6, !PT, !PT` computes a minimum into R4.

    Two predicate outputs come first and nothing uses them, so the register
    destination sits where every other opcode with two leading predicates keeps
    a source. `ISETP.GE.AND P0, PT, R2, R3, PT` reads R2; `IMNMX` writes R4.

    Reading it wrong is not cosmetic. The register never appears as a definition
    at all, so nothing depends on it, the checker reports no hazard and the
    scheduler leaves no gap in front of it. It was found when a scheduling
    change removed the stall before one and the GPU returned a different answer.
    """

    def test_the_register_after_the_predicates_is_written(self):
        access = operand_access("IMNMX.S64", "PT, PT, R4, R4, R6, !PT, !PT")
        assert RegRef(RegKind.GENERAL, 4) in access.real_defs

    def test_it_is_read_as_well(self):
        """`R4 = min(R4, R6)` reads what it writes."""
        access = operand_access("IMNMX.S64", "PT, PT, R4, R4, R6, !PT, !PT")
        assert RegRef(RegKind.GENERAL, 4) in access.real_uses
        assert RegRef(RegKind.GENERAL, 6) in access.real_uses

    def test_a_compare_still_keeps_its_source(self):
        """The same shape, and the register is a source. Hence the opcode list."""
        access = operand_access("ISETP.GE.AND", "P0, PT, R2, R3, PT")
        assert access.real_defs == {RegRef(RegKind.PREDICATE, 0)}
        assert RegRef(RegKind.GENERAL, 2) in access.real_uses

    def test_the_ordinary_three_operand_form_is_unchanged(self):
        access = operand_access("IMNMX", "R2, R3, R4, !PT")
        assert access.real_defs == {RegRef(RegKind.GENERAL, 2)}
