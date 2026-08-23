# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""The positive control, at full strength.

Every kernel the corpus can persuade `ptxas` to build is compiled and verified.
The vendor compiler's scheduling is the reference, so a single error on any of
them means basalt is wrong, and every finding it produces about anyone else's
code is worth nothing until that is fixed.

Checking one hand-written kernel is a smoke test. Checking the whole corpus,
across every instruction family it reaches, is what actually holds the model
honest: each of the four errors this project has made so far was found here or
by the smaller version of it, never by reasoning.

Marked `toolchain` and needs no GPU. Every class here shares one compile of the
corpus per `-O` level, and that compile is effectively the whole cost: measured
over 449 kernels on 16 cores, the three builds take 55s and the 1347 analyses
that follow take 2.1s. So the price of this file is the number of `-O` levels
swept, not the number of assertions made, which is why `OPT_LEVELS` exists.
"""

from __future__ import annotations

import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from basalt.disasm import disassemble_program
from basalt.encoding import NO_BARRIER, STALL_YIELD
from basalt.harvest.runner import _MALFORMED
from basalt.paths import ISA_DATABASE, LATENCIES, OBSERVED_STALLS
from basalt.verify.hazards import Severity, verify_program
from basalt.verify.latency import DEFAULT_MODEL, LatencyModel
from basalt.verify.observed import ObservedStalls

pytestmark = [pytest.mark.toolchain, pytest.mark.slow]

ARCH = "sm_120a"
ROOT = Path(__file__).resolve().parent.parent


# all three levels locally and nightly; CI narrows to -O3, which schedules
# hardest, because re-proving -O1 and -O2 per push finds nothing
OPT_LEVELS: tuple[int, ...] = tuple(
    int(x) for x in os.environ.get("BASALT_CORPUS_OPT", "1,2,3").split(",") if x.strip()
)


@pytest.fixture(scope="module")
def model() -> LatencyModel:
    if LATENCIES.is_file():
        return LatencyModel.assumed().overlay(LATENCIES)
    return DEFAULT_MODEL


@pytest.fixture(scope="module")
def observed() -> ObservedStalls | None:
    return ObservedStalls.read(OBSERVED_STALLS) if OBSERVED_STALLS.is_file() else None


@pytest.fixture(scope="module")
def reports(corpus_builds, model, observed):
    """Verify every kernel the vendor built, at every level that schedules.

    This checked `-O3` alone, and only the generated corpus, until widening it
    found two things at once. The mined stall table was overestimating, because
    a two-instruction kernel gives ptxas nothing to fill a gap with and it
    leaves conservative spacing: `FADD` was pinned at 5 cycles by 24 such
    observations while real code shows the vendor at 4. And a loop-carried
    dependency was being measured as a straight-line distance.
    """
    return [
        (f"{name} -O{opt}", verify_program(program, model, observed=observed))
        for opt in OPT_LEVELS
        for name, (_, program) in corpus_builds.at(opt).items()
    ]


class TestPositiveControl:
    def test_the_corpus_actually_compiled(self, reports):
        """A control over nothing proves nothing."""
        assert len(reports) > 900, f"only {len(reports)} kernel/level pairs compiled"

    def test_no_kernel_is_rejected_for_being_malformed(self, corpus_builds):
        """A corpus bug must not go on looking like a form sm_120 lacks.

        `ptxas` rejecting a snippet is an ordinary negative result, so the count
        alone says nothing. It hid every half-precision kernel emitting a load
        type PTX does not have, which cost four opcodes and eighteen mnemonics
        for as long as those kernels had existed.
        """
        corpus_builds.at(3)
        broken = {
            name: why.strip().splitlines()[0] if why.strip() else ""
            for name, why in corpus_builds.rejected.items()
            if _MALFORMED.search(why)
        }
        assert not broken, "kernels that do not compile because the PTX is wrong:\n" + "\n".join(
            f"  {name}: {why}" for name, why in list(broken.items())[:10]
        )

    def test_dependencies_were_actually_checked(self, reports):
        total = sum(report.checked_pairs for _, report in reports)
        assert total > 15000, f"only {total} dependencies checked"

    def test_no_vendor_kernel_produces_an_error(self, reports):
        """The one that matters. `ptxas` output must verify clean."""
        failures = [
            (name, h)
            for name, report in reports
            for h in report.hazards
            if h.severity is Severity.ERROR
        ]
        if failures:
            lines = [f"{name}: {h.describe()}" for name, h in failures[:20]]
            kinds = Counter(h.kind for _, h in failures)
            pytest.fail(
                f"{len(failures)} errors on vendor output, kinds {dict(kinds)}:\n"
                + "\n".join(lines)
            )

    def test_warnings_stay_rare(self, reports):
        """Warnings are tolerable; a flood of them means the model has drifted."""
        noisy = [name for name, report in reports if report.hazards]
        assert len(noisy) / len(reports) < 0.10, (
            f"{len(noisy)}/{len(reports)} kernels produced findings, "
            "which is too many to be reading real problems"
        )

    def test_cross_block_analysis_is_actually_running(self, reports):
        """A silent fallback to per-block checking would weaken every result."""
        assert any(report.cross_block for _, report in reports)

    def test_unknown_opcodes_stay_rare(self, reports):
        """An opcode with no latency entry is checked against a guess."""
        unknown: set[str] = set()
        for _, report in reports:
            unknown |= report.unknown_opcodes
        assert len(unknown) < 25, f"{len(unknown)} opcodes are not in the model: {sorted(unknown)}"


class TestSchedulerOverTheWholeCorpus:
    """basalt has to accept its own work, on every kernel, not just check others'.

    The scheduler discards every control bit `ptxas` produced and computes new
    ones. Handing the result straight back to the verifier catches the cases
    where it cannot even satisfy itself, which is cheap, needs no GPU, and is
    the half of the round trip that can run in CI.

    It is not the whole story and is not meant to be. Checker and scheduler read
    the same latency model, so a wrong entry satisfies both and only the silicon
    disagrees; that is what `scripts/roundtrip_corpus.py` is for. This is the
    floor, not the ceiling.
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def scheduled(toolchain, corpus_builds, model, observed):
        from basalt.asm.cubin import Cubin
        from basalt.sched.scheduler import schedule_program

        def one(task):
            name, opt, cubin_path, program = task
            result = schedule_program(program, model, observed=observed)
            if result.out_of_scoreboards:
                return (f"{name} -O{opt}", "out of scoreboards", result.out_of_scoreboards[0])
            cubin = Cubin.load(cubin_path)
            for slot, word in enumerate(result.words):
                if program.instructions[slot].word is not None:
                    cubin.write_word(slot, word)
            with TemporaryDirectory(prefix="basalt-sched-") as tmp:
                out = Path(tmp) / "r.cubin"
                cubin.save(out)
                # content first: this once read back an empty program and passed,
                # and a check that passes on nothing is worse than none
                written = disassemble_program(toolchain, out)
            if len(written.instructions) != len(program.instructions):
                return (
                    f"{name} -O{opt}",
                    "did not disassemble after rescheduling",
                    f"{len(written.instructions)} of {len(program.instructions)} instructions",
                )
            report = verify_program(written, model, observed=observed)
            if not report.ok:
                return (
                    f"{name} -O{opt}",
                    "rejected its own schedule",
                    report.hazards[0].describe(),
                )
            return None

        # every level that schedules: -O1 keeps a loop counter in the uniform
        # datapath that -O3 unrolls away, and two bugs lived there
        tasks = [
            (name, opt, cubin_path, program)
            for opt in (1, 2, 3)
            for name, (cubin_path, program) in corpus_builds.at(opt).items()
        ]
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
            return [r for r in pool.map(one, tasks) if r]

    def test_every_kernel_can_be_scheduled_and_verifies_clean(self, scheduled):
        if scheduled:
            lines = "\n".join(f"  {name}: {why} ({detail})" for name, why, detail in scheduled)
            pytest.fail(f"{len(scheduled)} kernels basalt could not schedule cleanly:\n{lines}")


class TestReadBarrierWindowsAreNotTightened:
    """A read barrier covers more reads than its own instruction's.

    `ptxas` puts one on the last of a run of loads and lets in-order issue plus
    the gaps it chose carry the earlier ones: by the time the last has read its
    address register, the earlier ones have too. Compress those gaps and the
    guarantee goes with them, which is `k_mma_m16n8k32_s4_s4_s32` at `-O1`,
    where `R4` was overwritten under four loads that had not finished reading it.

    What holds the window open is the rate the unit accepts work: `LDG` after
    `LDG` is 4 cycles across 1,953 observations, so four loads under one barrier
    take at least that much between them however little the dependencies ask for.
    basalt used to copy the vendor's own stalls here, which worked and could not
    have worked on a program the vendor never compiled.

    The window here is found by scanning back from the barrier to the previous
    barrier, control transfer or branch target, deliberately without asking the
    scheduler where it thinks the window is. Otherwise this would agree with the
    implementation by construction and catch nothing.

    It stops at a branch target because the vendor's spacing cannot mean
    anything across one: control can arrive there from somewhere else, so
    whatever gap the fall-through path happened to have is not a guarantee the
    vendor is relying on either. `s_loop_double` at `-O1` is the case that
    settles it, where the barrier is on a `DFMA` in a loop body and guards
    against the next iteration overwriting the operands of this one, not against
    anything in the preamble above the label.
    """

    TRANSFERS = frozenset({"BRA", "BRX", "CALL", "RET", "EXIT", "JMP", "JMX", "BSSY", "BSYNC"})

    @staticmethod
    @pytest.fixture(scope="class")
    def tightened(corpus_builds, model, observed):
        from basalt.sched.scheduler import schedule_program

        def one(task):
            name, opt, program = task
            result = schedule_program(program, model, observed=observed)
            transfers = TestReadBarrierWindowsAreNotTightened.TRANSFERS
            targets = set(getattr(program, "labels", {}).values())

            found = []
            previous = -1
            for index, instruction in enumerate(program.instructions):
                if instruction.word is None:
                    continue
                # basalt's barriers, not the vendor's: a window belongs to the
                # schedule that closes it. the rate is still recomputed here
                if result.words[index].field("read_barrier") == NO_BARRIER:
                    continue
                start = index
                while start > previous + 1 and start not in targets:
                    earlier = program.instructions[start - 1]
                    if earlier.word is None or earlier.opcode in transfers:
                        break
                    start -= 1
                for slot in range(start, index):
                    first = program.instructions[slot]
                    second = program.instructions[slot + 1]
                    if first.word is None or second.word is None or observed is None:
                        continue
                    rate = observed.issue_minimum(first.mnemonic, second.mnemonic)
                    ours = result.words[slot].field("stall")
                    if ours != STALL_YIELD and ours < rate:
                        found.append(
                            f"{name} -O{opt} slot {slot}: {ours} cycles where "
                            f"{first.mnemonic} into {second.mnemonic} needs {rate}"
                        )
                previous = index
            return found

        tasks = [
            (name, opt, program)
            for opt in (1, 2, 3)
            for name, (_, program) in corpus_builds.at(opt).items()
        ]
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
            return [line for lines in pool.map(one, tasks) for line in lines]

    def test_no_window_issues_faster_than_the_unit_accepts(self, tightened):
        if tightened:
            shown = "\n".join(f"  {line}" for line in tightened[:10])
            pytest.fail(f"{len(tightened)} read-barrier windows tightened:\n{shown}")


class TestAssemblerAgainstTheVendorsBytes:
    """Assembling a disassembled instruction must give back the same 128 bits.

    The strongest statement an assembler can make about itself, and the only one
    worth making: not that the text looks right afterwards, but that the bytes
    are the ones the vendor compiler emitted.

    Two numbers come out of this and only one of them is allowed to move. The
    share that reproduces exactly is coverage, and it goes up as the database
    learns more forms. The count that comes out *different* is a defect, and it
    is pinned at zero, because an assembler that emits a word which disassembles
    to the right text and computes something else is precisely the failure the
    rest of this repository exists to catch.
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def assembled(corpus_builds):
        from basalt.asm.assemble import Assembler, AssemblyError
        from basalt.isa.database import IsaDatabase

        database = ISA_DATABASE
        if not database.is_file():
            pytest.skip("no ISA database; run `basalt build-isa`")
        assembler = Assembler(IsaDatabase.read(database))

        def one(program):
            out = []
            for instruction in program.instructions:
                if instruction.word is None:
                    continue
                text = f"{instruction.mnemonic} {instruction.operands}".strip()
                try:
                    got = assembler.assemble(text, control=instruction.word)
                except AssemblyError:
                    out.append(("refused", text))
                    continue
                out.append(("exact" if got.value == instruction.word.value else "wrong", text))
            return out

        programs = [program for _, program in corpus_builds.at(3).values()]
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
            return [row for rows in pool.map(one, programs) for row in rows]

    def test_nothing_assembles_to_the_wrong_bytes(self, assembled):
        wrong = [text for verdict, text in assembled if verdict == "wrong"]
        assert not wrong, (
            f"{len(wrong)} instructions assembled to bytes the vendor did not emit, which is "
            f"worse than refusing them: {wrong[:5]}"
        )

    def test_coverage_does_not_regress(self, assembled):
        exact = sum(1 for verdict, _ in assembled if verdict == "exact")
        total = len(assembled)
        assert total > 5000, f"only {total} instructions seen; the corpus did not build"
        # 95.8% when written. below that on purpose, so a database that learns to
        # refuse something it was guessing at does not fail here
        assert exact / total >= 0.92, f"only {exact}/{total} ({exact / total:.1%}) reproduced"


class TestWholeProgramAssembly:
    """Assemble each kernel as a program, with its labels resolved.

    A branch cannot be assembled alone: the field holds the distance to the
    destination, so the same text encodes differently in every kernel it appears
    in. Given the whole program that distance is known, and the standard is
    unchanged: reproduce the vendor's bytes or refuse.
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def assembled(corpus_builds):
        from basalt.asm.assemble import assemble_program
        from basalt.isa.database import IsaDatabase

        database = ISA_DATABASE
        if not database.is_file():
            pytest.skip("no ISA database; run `basalt build-isa`")
        db = IsaDatabase.read(database)

        def one(program):
            result = assemble_program(program, db)
            exact = wrong = 0
            for instruction, got in zip(program.instructions, result.words, strict=True):
                if instruction.word is None or got is None:
                    continue
                if got.value == instruction.word.value:
                    exact += 1
                else:
                    wrong += 1
            return (exact, wrong, len(result.refused))

        programs = [program for _, program in corpus_builds.at(3).values()]
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
            rows = list(pool.map(one, programs))
        return tuple(sum(column) for column in zip(*rows, strict=True))

    def test_nothing_assembles_to_the_wrong_bytes(self, assembled):
        _, wrong, _ = assembled
        assert wrong == 0, f"{wrong} instructions assembled to bytes the vendor did not emit"

    def test_resolving_labels_covers_the_branches(self, assembled):
        exact, wrong, refused = assembled
        total = exact + wrong + refused
        assert total > 5000, "the corpus did not build"
        # 99.96% when written. a floor far under the real number lets coverage
        # fall a long way without anything going red
        assert exact / total >= 0.995, f"only {exact}/{total} ({exact / total:.1%}) reproduced"


class TestTheBranchFieldIsStillWhereItWasFound:
    """Re-derive the branch encoding rather than trusting the constant.

    `BRANCH_TARGET_BITS` was solved from real kernels: the label table gives the
    destination, the instruction gives its address, the word gives the bits. It
    is a measurement, and a measurement written down as a constant is exactly
    the kind of thing that goes quietly wrong when a compiler version changes.
    """

    def test_every_branch_in_the_corpus_decodes_to_its_label(self, corpus_builds):
        import re

        from basalt.asm.assemble import branch_target

        label = re.compile(r"`\(([^)]+)\)")

        def one(program):
            ok = bad = 0
            for instruction in program.instructions:
                if instruction.word is None:
                    continue
                match = label.search(instruction.operands)
                if match is None:
                    continue
                destination = program.labels.get(match.group(1))
                if destination is None:
                    continue
                if branch_target(instruction.word, instruction.offset) == destination * 16:
                    ok += 1
                else:
                    bad += 1
            return (ok, bad)

        programs = [program for _, program in corpus_builds.at(3).values()]
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
            rows = list(pool.map(one, programs))
        ok = sum(row[0] for row in rows)
        bad = sum(row[1] for row in rows)
        assert ok > 100, f"only {ok} branches found; the corpus did not build"
        assert bad == 0, f"{bad} branches did not decode to their label"


class TestWhatTheCorrectnessCosts:
    """basalt's schedules are correct and slower. Both halves are measured.

    A scheduler that only reports whether it was right is hiding the trade it
    made. basalt reaches for the safe stall encoding at every block boundary and
    declines to lean on a wait a predicated instruction carries, and those are
    not free: over the whole corpus its schedules spend about 40% more cycles
    issuing than the vendor's.

    Pinned so the number cannot drift in either direction unnoticed. Getting
    slower is a regression. Getting much faster without the round trip also
    moving is a reason to check the round trip rather than to celebrate.
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def cycles(corpus_builds, model, observed):
        from basalt.sched.scheduler import issue_cycles, schedule_program

        def one(program):
            result = schedule_program(program, model, observed=observed)
            if result.out_of_scoreboards:
                return (0, 0)
            return (
                issue_cycles([i.word for i in program.instructions], program.instructions),
                issue_cycles(result.words, program.instructions),
            )

        programs = [program for _, program in corpus_builds.at(3).values()]
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
            rows = list(pool.map(one, programs))
        return sum(r[0] for r in rows), sum(r[1] for r in rows)

    def test_the_cost_is_known_and_bounded(self, cycles):
        vendor, basalt = cycles
        assert vendor > 5000, "the corpus did not build"
        ratio = basalt / vendor
        # 1.05x when written, pinned from both sides: faster is a reason to
        # distrust the costing rather than to celebrate it (finding 12)
        assert ratio < 1.15, f"basalt's schedules cost {ratio:.2f}x the vendor's, up from 1.05x"
        assert ratio > 0.75, (
            f"basalt's schedules cost {ratio:.2f}x, which is far cheaper than the vendor's and "
            f"is the shape a costing bug takes. check the hardware round trip before believing it"
        )
