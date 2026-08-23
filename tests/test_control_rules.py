# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""The rules basalt derives the last two control fields from.

Read barriers and the yield bit were the two fields basalt did not compute. It
copied the first out of the schedule it was replacing and guessed the second,
which meant it could not have scheduled a program nobody had compiled. Findings
25 to 27 replace both with rules, and these are what hold them.

Deliberately checked against the *shape* of what comes out rather than against
the vendor's choice. basalt over-approximates on purpose: a scoreboard is a
counter, so sharing one over-synchronises rather than corrupting, and "the same
barriers ptxas picked" is therefore the wrong standard. What has to be true is
that every window is covered, nothing is signalled into a void, and nothing is
renumbered that another instruction names by hand.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from basalt.encoding import NO_BARRIER
from basalt.sched.scheduler import (
    SCOREBOARD_OPERAND,
    YIELD_STALL_RANGE,
    schedule_program,
)
from basalt.verify.cfg import build_cfg
from basalt.verify.latency import DEFAULT_MODEL, LatencyClass, LatencyModel
from basalt.verify.observed import ObservedStalls
from basalt.verify.operands import operand_access

pytestmark = [pytest.mark.toolchain, pytest.mark.slow]

ROOT = Path(__file__).resolve().parent.parent
LATENCIES = ROOT / "data" / "latency" / "rtx-5070-ti.json"
OBSERVED = ROOT / "data" / "latency" / "observed-stalls-sm120a.json"

# stores have no register result and so no latency class, but their data
# register is still in flight after they issue
LATE_READING_STORES = frozenset({"STG", "STS", "STL", "ST", "RED"})


@pytest.fixture(scope="module")
def model() -> LatencyModel:
    return LatencyModel.assumed().overlay(LATENCIES) if LATENCIES.is_file() else DEFAULT_MODEL


@pytest.fixture(scope="module")
def observed() -> ObservedStalls | None:
    return ObservedStalls.read(OBSERVED) if OBSERVED.is_file() else None


@pytest.fixture(scope="module")
def scheduled(corpus_builds, model, observed):
    """Every corpus kernel rescheduled, at every level that schedules."""
    return [
        (f"{name} -O{opt}", program, schedule_program(program, model, observed=observed))
        for opt in (1, 2, 3)
        for name, (_, program) in corpus_builds.at(opt).items()
    ]


def _late_reader(opcode: str, model: LatencyModel) -> bool:
    return model.lookup(opcode).kind is LatencyClass.VARIABLE or opcode in LATE_READING_STORES


class TestTheDerivedReadBarriers:
    def test_the_corpus_needs_some(self, scheduled):
        """A rule nothing exercises is not being tested by any of this."""
        total = sum(result.read_barriers_used for _, _, result in scheduled)
        assert total > 50, f"only {total} read barriers placed; the rule is not being exercised"

    def test_every_one_is_waited_on(self, scheduled):
        """A barrier nothing waits for is a wasted number, not synchronisation."""
        stranded = []
        for name, _, result in scheduled:
            waited = 0
            for word in result.words:
                waited |= word.field("wait_mask")
            for index in sorted(result.analysed):
                barrier = result.words[index].field("read_barrier")
                if barrier != NO_BARRIER and not (waited >> barrier) & 1:
                    stranded.append(f"{name} #{index} SB{barrier}")
        assert not stranded, f"{len(stranded)} read barriers nothing waits on: {stranded[:5]}"

    def test_the_vendor_waits_on_barriers_it_signals(self, scheduled):
        """Which is why basalt is not required to avoid it.

        A wait is evaluated before the instruction issues and its barrier is
        raised at or after issue, so an instruction carrying both is reusing a
        number that has just drained rather than waiting on its own result. It
        reads wrong and is not, and basalt spent a repair loop reallocating away
        from it on the strength of a sample too small to see `ptxas` doing it
        260 times in 36,576 instructions.

        Pinned as a fact about the vendor rather than about basalt: if a future
        toolchain stops emitting the shape, that is worth re-examining rather
        than silently inheriting.
        """
        from basalt.encoding import NO_BARRIER

        found = 0
        for _, program, _ in scheduled:
            for instruction in program.instructions:
                if instruction.word is None:
                    continue
                for field in ("write_barrier", "read_barrier"):
                    barrier = instruction.word.field(field)
                    if barrier == NO_BARRIER:
                        continue
                    if (instruction.word.field("wait_mask") >> barrier) & 1:
                        found += 1
        assert found > 100, (
            f"ptxas emitted this shape only {found} times; basalt stopped avoiding it "
            f"because the vendor does it freely, and that premise needs re-checking"
        )

    def test_a_late_read_is_covered_before_its_register_moves(self, scheduled, model):
        """The rule itself, recomputed rather than asked of the scheduler.

        For every variable-latency instruction or store, walk forward to the
        first instruction that overwrites one of its sources. Something between
        the two has to wait on a barrier the reader signals, or on the reader's
        own write barrier, which cannot clear before the read has happened.
        Anything else is a read still in flight when its register moves.

        Within a block only. A window that spans an edge is covered by the same
        rule from the other side, and judging it here would need a path.
        """
        uncovered = []
        for name, program, result in scheduled:
            words = result.words
            for block in build_cfg(program).blocks:
                for reader in range(block.start, block.end):
                    instruction = program.instructions[reader]
                    if reader not in result.analysed:
                        continue
                    if not _late_reader(instruction.opcode, model):
                        continue
                    sources = operand_access(instruction.mnemonic, instruction.operands).real_uses
                    if not sources:
                        continue
                    for later in range(reader + 1, block.end):
                        following = program.instructions[later]
                        if later not in result.analysed:
                            continue
                        access = operand_access(following.mnemonic, following.operands)
                        if not access.real_defs & sources:
                            continue
                        # in order, so a barrier on any later reader before the
                        # overwrite covers this one too (finding 13)
                        mask = words[later].field("wait_mask")
                        covered = any(
                            barrier != NO_BARRIER and (mask >> barrier) & 1
                            for between in range(reader, later)
                            for barrier in (
                                words[between].field("read_barrier"),
                                words[between].field("write_barrier"),
                            )
                        )
                        if not covered:
                            uncovered.append(f"{name} #{reader} -> #{later}")
                        break
        assert not uncovered, (
            f"{len(uncovered)} reads still in flight when their register moves: {uncovered[:5]}"
        )


class TestAScoreboardNamedInAnOperandKeepsItsNumber:
    """`DEPBAR` waits by naming its scoreboard, not through the wait mask.

    So the number is not basalt's to reassign, and nothing in the control word
    would show that it had been. This is how an async copy waits for its copies
    to drain, and getting it wrong is silent (finding 27).
    """

    def test_the_signaller_keeps_the_number(self, scheduled):
        seen = 0
        broken = []
        for name, program, result in scheduled:
            for index, instruction in enumerate(program.instructions):
                if instruction.word is None:
                    continue
                for match in SCOREBOARD_OPERAND.finditer(instruction.operands):
                    wanted = int(match.group(1))
                    signaller = next(
                        (
                            earlier
                            for earlier in range(index - 1, -1, -1)
                            if (word := program.instructions[earlier].word) is not None
                            and word.field("write_barrier") == wanted
                        ),
                        None,
                    )
                    if signaller is None:
                        continue
                    seen += 1
                    got = result.words[signaller].field("write_barrier")
                    if got != wanted:
                        broken.append(f"{name} #{signaller}: SB{got} where SB{wanted} is named")
        assert seen, "no corpus kernel names a scoreboard in an operand; cp.async should"
        assert not broken, f"{len(broken)} renumbered out from under a DEPBAR: {broken[:5]}"


class TestTheYieldBit:
    """Fitted to the vendor rather than reasoned about, so this pins the fit.

    The bit does not gate correctness on this architecture, which finding 26
    establishes by inverting 680 of them and getting the vendor's answer every
    time. That makes it a free choice, and the only reason to prefer one rule
    over another is how closely it tracks what `ptxas` does.
    """

    def test_it_follows_the_stall(self, scheduled):
        from basalt.encoding import STALL_YIELD

        low, high = YIELD_STALL_RANGE
        wrong = []
        for name, _, result in scheduled:
            for index in sorted(result.analysed):
                word = result.words[index]
                stall = word.field("stall")
                want = int(low <= stall < high)
                if word.field("yield_") != want:
                    wrong.append(f"{name} #{index} stall={stall}")
        assert not wrong, f"{len(wrong)} yield bits do not follow the stall: {wrong[:5]}"
        assert low > STALL_YIELD, "the safe stall encoding must never carry the yield hint"

    def test_the_fit_still_beats_the_guess_it_replaced(self, scheduled):
        """Against the vendor's own words, not against basalt's.

        `-O0` is excluded from `scheduled` and would flatter this: it emits a
        zeroed control word, so every instruction agrees with any rule that
        declines to yield at a stall of zero.
        """
        low, high = YIELD_STALL_RANGE
        fitted = guessed = total = 0
        for _, program, _ in scheduled:
            for instruction in program.instructions:
                if instruction.word is None:
                    continue
                stall = instruction.word.field("stall")
                actual = instruction.word.field("yield_")
                total += 1
                fitted += int(low <= stall < high) == actual
                guessed += int(stall == 1) == actual
        assert total > 5000, "the corpus did not build"
        assert fitted / total > 0.9, f"the rule agrees with ptxas on only {fitted / total:.1%}"
        assert fitted > guessed, (
            f"the fitted rule ({fitted / total:.1%}) no longer beats the guess it replaced "
            f"({guessed / total:.1%}); re-fit it against a fresh corpus"
        )
