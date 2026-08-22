# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""The hazard analysis, exercised on hand-built programs.

Synthetic instructions rather than compiler output, so each test isolates one
rule. The two properties that matter most are checked here in miniature and
again against the real compiler in test_pipeline.py:

- a correctly scheduled program produces nothing, and
- an incorrectly scheduled one produces exactly the expected finding.

A checker that only satisfies the first is indistinguishable from a no-op.
"""

from __future__ import annotations

from basalt.disasm import Instruction
from basalt.encoding import NO_BARRIER, Word
from basalt.verify.hazards import (
    HazardKind,
    Severity,
    split_blocks,
    verify_program,
)
from basalt.verify.latency import (
    Confidence,
    LatencyClass,
    LatencyModel,
    LatencyRecord,
)


def instr(
    text: str,
    *,
    stall: int = 1,
    wait: int = 0,
    write_barrier: int = NO_BARRIER,
    read_barrier: int = NO_BARRIER,
    index: int = 0,
) -> Instruction:
    """Build one instruction with explicit control bits."""
    word = Word(0)
    word = word.with_field("stall", stall)
    word = word.with_field("wait_mask", wait)
    word = word.with_field("write_barrier", write_barrier)
    word = word.with_field("read_barrier", read_barrier)
    return Instruction(offset=index * 16, text=text, word=word)


def model(cycles: int = 4, confidence: Confidence = Confidence.MEASURED) -> LatencyModel:
    """A tiny model: IADD fixed, LDG variable, EXIT control."""
    return LatencyModel(
        records={
            "IADD": LatencyRecord(cycles, LatencyClass.FIXED, confidence),
            "IMAD": LatencyRecord(cycles, LatencyClass.FIXED, confidence),
            "LDG": LatencyRecord(0, LatencyClass.VARIABLE, confidence),
            "S2R": LatencyRecord(0, LatencyClass.VARIABLE, confidence),
            "STG": LatencyRecord(0, LatencyClass.CONTROL, confidence),
            "EXIT": LatencyRecord(0, LatencyClass.CONTROL, confidence),
            "BRA": LatencyRecord(0, LatencyClass.CONTROL, confidence),
        },
        sku="synthetic",
    )


class TestFixedLatency:
    def test_sufficient_stall_is_clean(self):
        prog = [
            instr("IADD R2, R3, R4", stall=4, index=0),
            instr("IADD R5, R2, R4", stall=1, index=1),
        ]
        report = verify_program(prog, model(cycles=4))
        assert report.ok
        assert report.checked_pairs == 1

    def test_insufficient_stall_is_flagged(self):
        prog = [
            instr("IADD R2, R3, R4", stall=1, index=0),
            instr("IADD R5, R2, R4", stall=1, index=1),
        ]
        report = verify_program(prog, model(cycles=4))
        assert not report.ok
        (h,) = report.hazards
        assert h.kind is HazardKind.UNDERSTALLED
        assert h.required == 4
        assert h.actual == 1
        assert h.register == "R2"

    def test_stalls_accumulate_across_intervening_instructions(self):
        """Three instructions of stall 2 cover a latency of 4."""
        prog = [
            instr("IADD R2, R3, R4", stall=2, index=0),
            instr("IADD R8, R9, R9", stall=2, index=1),
            instr("IADD R5, R2, R4", stall=1, index=2),
        ]
        assert verify_program(prog, model(cycles=4)).ok

    def test_exactly_enough_stall_is_clean(self):
        prog = [
            instr("IADD R2, R3, R4", stall=4, index=0),
            instr("IADD R5, R2, R4", stall=1, index=1),
        ]
        assert verify_program(prog, model(cycles=4)).ok

    def test_one_cycle_short_is_flagged(self):
        prog = [
            instr("IADD R2, R3, R4", stall=3, index=0),
            instr("IADD R5, R2, R4", stall=1, index=1),
        ]
        assert not verify_program(prog, model(cycles=4)).ok

    def test_unrelated_registers_are_not_dependencies(self):
        prog = [
            instr("IADD R2, R3, R4", stall=1, index=0),
            instr("IADD R5, R6, R7", stall=1, index=1),
        ]
        report = verify_program(prog, model(cycles=4))
        assert report.ok
        assert report.checked_pairs == 0


class TestVariableLatency:
    def test_consumer_waiting_on_the_barrier_is_clean(self):
        prog = [
            instr("LDG.E R2, desc[UR4][R6.64]", write_barrier=0, index=0),
            instr("IADD R5, R2, R4", wait=0b1, index=1),
        ]
        assert verify_program(prog, model()).ok

    def test_consumer_not_waiting_is_flagged(self):
        prog = [
            instr("LDG.E R2, desc[UR4][R6.64]", write_barrier=0, index=0),
            instr("IADD R5, R2, R4", wait=0b0, index=1),
        ]
        report = verify_program(prog, model())
        (h,) = report.hazards
        assert h.kind is HazardKind.BARRIER_NOT_AWAITED
        assert h.severity is Severity.ERROR

    def test_producer_signalling_nothing_is_flagged(self):
        prog = [
            instr("LDG.E R2, desc[UR4][R6.64]", write_barrier=NO_BARRIER, index=0),
            instr("IADD R5, R2, R4", wait=0b1, index=1),
        ]
        (h,) = verify_program(prog, model()).hazards
        assert h.kind is HazardKind.NO_BARRIER_SET

    def test_an_earlier_wait_satisfies_later_consumers(self):
        """Once anything waits on a scoreboard, the data is available downstream.

        Requiring every consumer to carry the wait itself is the mistake that
        made the checker fire on correct compiler output.
        """
        prog = [
            instr("LDG.E R2, desc[UR4][R6.64]", write_barrier=0, index=0),
            # stall covers this instruction's own fixed-latency result so the
            # only thing left for the assertion to catch is the barrier rule
            instr("IADD R5, R2, R4", wait=0b1, stall=4, index=1),
            instr("IADD R8, R2, R5", wait=0b0, stall=1, index=2),
        ]
        assert verify_program(prog, model()).ok

    def test_several_producers_may_share_one_scoreboard(self):
        """Scoreboards are counters, not flags; one wait covers every signal."""
        prog = [
            instr("LDG.E R2, desc[UR4][R6.64]", write_barrier=0, index=0),
            instr("LDG.E R3, desc[UR4][R6.64+0x4]", write_barrier=0, index=1),
            instr("IADD R5, R2, R3", wait=0b1, index=2),
        ]
        assert verify_program(prog, model()).ok

    def test_waiting_on_the_wrong_scoreboard_is_flagged(self):
        prog = [
            instr("LDG.E R2, desc[UR4][R6.64]", write_barrier=1, index=0),
            instr("IADD R5, R2, R4", wait=0b1, index=1),  # waits on 0, not 1
        ]
        (h,) = verify_program(prog, model()).hazards
        assert h.kind is HazardKind.BARRIER_NOT_AWAITED


class TestWriteAfterRead:
    def test_overwriting_an_operand_still_being_read_is_flagged(self):
        prog = [
            instr("STG.E desc[UR4][R2.64], R7", read_barrier=0, index=0),
            instr("IADD R7, R3, R4", wait=0b0, index=1),
        ]
        (h,) = verify_program(prog, model()).hazards
        assert h.kind is HazardKind.OVERWRITTEN_BEFORE_READ
        assert h.register == "R7"

    def test_waiting_on_the_read_barrier_is_clean(self):
        prog = [
            instr("STG.E desc[UR4][R2.64], R7", read_barrier=0, index=0),
            instr("IADD R7, R3, R4", wait=0b1, index=1),
        ]
        assert verify_program(prog, model()).ok

    def test_overwriting_an_unrelated_register_is_clean(self):
        prog = [
            instr("STG.E desc[UR4][R2.64], R7", read_barrier=0, index=0),
            instr("IADD R9, R3, R4", wait=0b0, index=1),
        ]
        assert verify_program(prog, model()).ok


class TestConfidence:
    def test_assumed_latency_downgrades_to_a_warning(self):
        """A hazard derived from a guess is a lead, not a finding."""
        prog = [
            instr("IADD R2, R3, R4", stall=1, index=0),
            instr("IADD R5, R2, R4", stall=1, index=1),
        ]
        report = verify_program(prog, model(confidence=Confidence.ASSUMED))
        (h,) = report.hazards
        assert h.severity is Severity.WARNING
        assert report.ok  # warnings alone do not fail the run

    def test_measured_latency_is_an_error(self):
        prog = [
            instr("IADD R2, R3, R4", stall=1, index=0),
            instr("IADD R5, R2, R4", stall=1, index=1),
        ]
        report = verify_program(prog, model(confidence=Confidence.MEASURED))
        assert report.hazards[0].severity is Severity.ERROR
        assert not report.ok

    def test_barrier_hazards_stay_errors_regardless_of_latency_confidence(self):
        """A missing wait is a structural error; no latency number is involved."""
        prog = [
            instr("LDG.E R2, desc[UR4][R6.64]", write_barrier=0, index=0),
            instr("IADD R5, R2, R4", wait=0b0, index=1),
        ]
        report = verify_program(prog, model(confidence=Confidence.ASSUMED))
        assert report.hazards[0].severity is Severity.ERROR


class TestBasicBlocks:
    def test_control_flow_ends_a_block(self):
        prog = [
            instr("IADD R2, R3, R4", index=0),
            instr("BRA `(.L_x_0)", index=1),
            instr("IADD R5, R2, R4", index=2),
        ]
        assert len(split_blocks(prog)) == 2

    def test_definitions_do_not_cross_a_branch(self):
        """Conservative on purpose: a linear listing is not a control-flow graph."""
        prog = [
            instr("IADD R2, R3, R4", stall=1, index=0),
            instr("BRA `(.L_x_0)", index=1),
            instr("IADD R5, R2, R4", stall=1, index=2),
        ]
        report = verify_program(prog, model(cycles=4))
        assert report.ok
        assert report.checked_pairs == 0

    def test_a_program_without_branches_is_one_block(self):
        prog = [instr("IADD R2, R3, R4", index=i) for i in range(4)]
        assert len(split_blocks(prog)) == 1

    def test_indices_in_findings_are_program_wide(self):
        """Block-local positions would make a finding impossible to locate."""
        prog = [
            instr("IADD R9, R3, R4", index=0),
            instr("BRA `(.L_x_0)", index=1),
            instr("IADD R2, R3, R4", stall=1, index=2),
            instr("IADD R5, R2, R4", stall=1, index=3),
        ]
        (h,) = verify_program(prog, model(cycles=4)).hazards
        assert (h.def_index, h.use_index) == (2, 3)


class TestReporting:
    def test_unknown_opcodes_are_recorded(self):
        prog = [instr("FROBNICATE R2, R3", index=0)]
        report = verify_program(prog, LatencyModel.assumed())
        assert "FROBNICATE" in report.unknown_opcodes

    def test_summary_mentions_counts(self):
        prog = [instr("IADD R2, R3, R4", index=0)]
        assert "1 instructions" in verify_program(prog, model()).summary()

    def test_empty_program_is_clean(self):
        report = verify_program([], model())
        assert report.ok
        assert report.instructions == 0
