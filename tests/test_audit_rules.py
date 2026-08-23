# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""The rules stage 10 added, and the reasoning each one has to keep.

Pointing the checker at machine code basalt did not compile reported 6,593
errors against a JPEG decoder that has never returned a wrong pixel. Every one
was basalt's, and the eight corrections behind finding 32 are what these hold.

None of them can be checked by running the corpus, which is the whole point: a
table mined from the code it is then checked against cannot fail. So each rule
is stated here against the property that makes it true rather than against an
output someone could regenerate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from basalt.verify.hazards import _exclusive, _requirement
from basalt.verify.latency import (
    ANTI_DEPENDENCY_CYCLES,
    DEFAULT_MODEL,
    GUARD_CYCLES,
    MEASURED_ANTI_PAIRS,
    SCOREBOARD_RESIDUE_CYCLES,
    LatencyClass,
)
from basalt.verify.observed import MIN_OBSERVATIONS, ObservedStalls, StallEvidence

ROOT = Path(__file__).resolve().parent.parent
OBSERVED = ROOT / "data" / "latency" / "observed-stalls-sm120a.json"


@pytest.fixture(scope="module")
def observed() -> ObservedStalls:
    if not OBSERVED.is_file():
        pytest.skip("no mined stall table")
    return ObservedStalls.read(OBSERVED)


class TestClassification:
    """Two opcodes the corpus could not classify, and why it could not."""

    def test_the_packed_conversions_are_not_the_conversion_pipe(self) -> None:
        # F2IP, I2FP and F2FP are their own instructions; longest-prefix would
        # hand the first two the scoreboarded entry belonging to F2I and I2F
        for opcode in ("F2IP", "I2FP", "F2FP"):
            assert DEFAULT_MODEL.lookup(opcode).kind is LatencyClass.FIXED

        for opcode in ("F2I", "I2F", "F2F"):
            assert DEFAULT_MODEL.lookup(opcode).kind is LatencyClass.VARIABLE

    def test_a_never_scoreboarded_opcode_cannot_complete_out_of_order(self) -> None:
        # ptxas signals no barrier on R2UR in 3,098 corpus instances, which is
        # the argument that reclassified VOTEU before it
        for opcode in ("R2UR", "VOTEU"):
            assert DEFAULT_MODEL.lookup(opcode).kind is LatencyClass.FIXED

    def test_every_tensor_core_instruction_is_one_family(self) -> None:
        # DMMA variable where the rest are fixed alleged 892 missing waits in
        # one cuSOLVER-Mg accumulate chain
        for opcode in ("HMMA", "IMMA", "QMMA", "OMMA", "BMMA", "DMMA"):
            assert DEFAULT_MODEL.lookup(opcode).kind is LatencyClass.FIXED


class TestUpperBounds:
    """A mined figure is an upper bound, so it may lower and never raise."""

    def test_a_requirement_never_exceeds_the_producers_own_latency(self, observed) -> None:
        record = DEFAULT_MODEL.lookup("IADD")
        required, _, _ = _requirement("IADD", "MOV", record, observed)
        assert required <= record.cycles

    def test_a_guard_never_costs_more_than_the_measured_figure(self, observed) -> None:
        record = DEFAULT_MODEL.lookup("ISETP")
        required, _, _ = _requirement("ISETP.NE.U32.AND", "BRA", record, observed, guard=True)
        assert required <= GUARD_CYCLES

    def test_thin_evidence_is_neutralised_by_the_bound_not_by_a_trust_gate(self) -> None:
        # four observations said `IADD -> MOV` needs 20 and the vendor ships it
        # at 5, but demoting thin evidence let two broken kernels through
        record = DEFAULT_MODEL.lookup("IADD")
        thin = ObservedStalls(cuda_version="", arch="sm_120")
        thin.by_pair[("IADD", "MOV")] = StallEvidence("IADD", "MOV", minimum=20, observations=4)
        required, _, grounded = _requirement("IADD", "MOV", record, thin)
        assert required == record.cycles
        assert grounded

    def test_the_constants_stay_where_they_were_measured(self) -> None:
        # each is the only figure fault injection produced for its rule, and
        # each is a ceiling on what mined evidence may claim
        assert (GUARD_CYCLES, ANTI_DEPENDENCY_CYCLES, SCOREBOARD_RESIDUE_CYCLES) == (13, 3, 2)


class TestEmitFloor:
    """Evidence lowers what basalt alleges and never what it emits."""

    def test_absorbing_a_tighter_table_keeps_the_wider_floor(self) -> None:
        base = ObservedStalls(cuda_version="", arch="sm_120")
        base.by_pair[("IMAD", "STS")] = StallEvidence("IMAD", "STS", minimum=9, observations=60)
        other = ObservedStalls(cuda_version="", arch="sm_120")
        other.by_pair[("IMAD", "STS")] = StallEvidence("IMAD", "STS", minimum=5, observations=900)

        base.absorb(other)
        evidence = base.by_pair[("IMAD", "STS")]
        assert evidence.minimum == 5
        assert evidence.emit == 9
        assert evidence.observations == 960

    def test_a_pairing_only_the_wider_table_has_carries_its_own_floor(self) -> None:
        base = ObservedStalls(cuda_version="", arch="sm_120")
        other = ObservedStalls(cuda_version="", arch="sm_120")
        other.by_pair[("SEL", "PRMT")] = StallEvidence("SEL", "PRMT", minimum=4, observations=90)

        base.absorb(other)
        assert base.by_pair[("SEL", "PRMT")].emit == 4

    def test_the_checked_in_table_never_emits_below_what_it_alleges(self, observed) -> None:
        for table in (observed.by_pair, observed.by_producer, observed.by_scoreboarded):
            for key, evidence in table.items():
                assert evidence.emit >= evidence.minimum, key


class TestSpacing:
    """Whether a missing scoreboard is a hazard is measured, not assumed."""

    def test_a_global_load_is_never_covered_by_spacing(self, observed) -> None:
        # zero across 1.3 million dependent pairs, which is what keeps the
        # missing barrier an error for this group
        for opcode in ("LDG", "LDC", "LDL", "S2R"):
            assert observed.covered_by_spacing(opcode) < MIN_OBSERVATIONS

    def test_a_shared_load_is_covered_by_spacing_constantly(self, observed) -> None:
        assert observed.covered_by_spacing("LDS") >= MIN_OBSERVATIONS

    def test_every_producer_shipped_code_uses_is_in_the_model(self, observed) -> None:
        # an opcode the model has never seen is checked against a guess, which
        # is the one gap left in a checker whose job is foreign machine code
        producers = {p.split(".")[0] for p, _ in observed.by_pair}
        producers |= {p.split(".")[0] for p, _ in observed.by_scoreboarded}
        unknown = {
            op for op in producers if DEFAULT_MODEL.lookup(op).note.startswith("opcode not in")
        }
        assert not unknown, sorted(unknown)


class TestAntiDependency:
    """Three cycles is one pairing's measurement, not a rule."""

    def test_only_the_measured_pairing_grounds_an_error(self) -> None:
        assert frozenset({("ULEA", "UMOV")}) == MEASURED_ANTI_PAIRS

    def test_the_vendor_goes_below_the_constant_constantly(self, observed) -> None:
        # 2,591 trusted pairings, which is why charging it everywhere alleged 41
        # errors against three shipped libraries
        below = sum(
            1 for e in observed.by_anti.values() if e.trusted and e.minimum < ANTI_DEPENDENCY_CYCLES
        )
        assert below > 1000


class TestReachingDefinitions:
    """A wait must not forget how a definition got here."""

    def test_a_wait_preserves_yielded_and_crossed(self) -> None:
        from basalt.verify.flow import FlowState
        from basalt.verify.operands import RegKind, RegRef

        reg = RegRef(RegKind.GENERAL, 4)
        state = FlowState()
        state.define(reg, 0, barrier=1)
        state.advance(1, yielded=True)
        crossed = state.crossing()
        crossed.satisfy(0b10)

        landed = crossed.reaching(reg)[0]
        assert landed.satisfied
        assert landed.yielded, "a wait dropped the safe-encoding flag"
        assert landed.crossed, "a wait dropped the edge it came across"


class TestExclusiveGuards:
    """A definition under `@P0` does not reach a use under `@!P0`."""

    def test_opposite_halves_of_one_predicate_never_meet(self) -> None:
        assert _exclusive("@P0 LDG.E.U8 R18, desc[UR20][R16.64]", "@!P0 PRMT R18, R19, 0x7610, R18")

    def test_the_same_half_still_meets(self) -> None:
        assert not _exclusive("@P0 IADD R4, R0, 0x1", "@P0 MOV R0, R4")

    def test_different_predicates_are_not_exclusive(self) -> None:
        assert not _exclusive("@P0 IADD R4, R0, 0x1", "@!P1 MOV R0, R4")

    def test_an_unguarded_use_always_meets(self) -> None:
        assert not _exclusive("@P0 IADD R4, R0, 0x1", "MOV R0, R4")
