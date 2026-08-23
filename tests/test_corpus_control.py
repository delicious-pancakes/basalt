# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""The positive control, at full strength.

Every kernel the corpus can persuade `ptxas` to build is compiled and verified.
The vendor compiler's scheduling is the reference, so a single error on any of
them means basalt is wrong, and every finding it produces about anyone else's
code is worth nothing until that is fixed.

Checking one hand-written kernel is a smoke test. Checking three hundred, across
every instruction family the corpus reaches, is what actually holds the model
honest: each of the four errors this project has made so far was found here or
by the smaller version of it, never by reasoning.

Marked `toolchain`, needs no GPU, and takes a couple of minutes.
"""

from __future__ import annotations

import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from basalt.disasm import disassemble_program
from basalt.encoding import NO_BARRIER
from basalt.harvest.corpus import generate
from basalt.harvest.corpus_shapes import generate_shapes
from basalt.harvest.corpus_tensor import generate_tensor
from basalt.verify.hazards import Severity, verify_program
from basalt.verify.latency import DEFAULT_MODEL, LatencyModel
from basalt.verify.observed import ObservedStalls

pytestmark = [pytest.mark.toolchain, pytest.mark.slow]

ARCH = "sm_120a"
ROOT = Path(__file__).resolve().parent.parent
LATENCIES = ROOT / "data" / "latency" / "rtx-5070-ti.json"
OBSERVED = ROOT / "data" / "latency" / "observed-stalls-sm120a.json"


@pytest.fixture(scope="module")
def model() -> LatencyModel:
    if LATENCIES.is_file():
        return LatencyModel.assumed().overlay(LATENCIES)
    return DEFAULT_MODEL


@pytest.fixture(scope="module")
def observed() -> ObservedStalls | None:
    return ObservedStalls.read(OBSERVED) if OBSERVED.is_file() else None


@pytest.fixture(scope="module")
def reports(toolchain, model, observed):
    """Compile and verify the whole corpus once, then share the results."""
    snippets = generate() + generate_tensor()

    def one(snippet):
        with TemporaryDirectory(prefix="basalt-control-") as tmp:
            src, cubin = Path(tmp) / "k.ptx", Path(tmp) / "k.cubin"
            src.write_text(snippet.ptx)
            built = toolchain.run(
                [str(toolchain.ptxas), f"-arch={ARCH}", "-O3", "-o", str(cubin), str(src)],
                check=False,
                timeout=120.0,
            )
            if built.returncode != 0:
                return None
            program = disassemble_program(toolchain, cubin)
        return snippet.name, verify_program(program, model, observed=observed)

    workers = min(32, (os.cpu_count() or 4) * 2)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [r for r in pool.map(one, snippets) if r is not None]


class TestPositiveControl:
    def test_the_corpus_actually_compiled(self, reports):
        """A control over nothing proves nothing."""
        assert len(reports) > 200, f"only {len(reports)} kernels compiled"

    def test_dependencies_were_actually_checked(self, reports):
        total = sum(report.checked_pairs for _, report in reports)
        assert total > 1000, f"only {total} dependencies checked"

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
    def scheduled(toolchain, model, observed):
        from basalt.asm.cubin import Cubin
        from basalt.sched.scheduler import schedule_program

        def one(task):
            # Every level that schedules, not just -O3. They emit different code
            # from the same source: -O3 unrolls a loop into ordinary registers
            # where -O1 keeps its counter in uniform ones, and the uniform
            # datapath had no coverage at all until the round trip was run at
            # -O1 and found two kernels basalt scheduled wrong.
            snippet, opt = task
            with TemporaryDirectory(prefix="basalt-sched-") as tmp:
                src = Path(tmp) / "k.ptx"
                cubin_path = Path(tmp) / "k.cubin"
                src.write_text(snippet.ptx)
                built = toolchain.run(
                    [
                        str(toolchain.ptxas),
                        f"-arch={ARCH}",
                        f"-O{opt}",
                        "-o",
                        str(cubin_path),
                        str(src),
                    ],
                    check=False,
                    timeout=60.0,
                )
                if built.returncode != 0:
                    return None
                program = disassemble_program(toolchain, cubin_path)
                result = schedule_program(program, model, observed=observed)
                if result.out_of_scoreboards:
                    return (
                        f"{snippet.name} -O{opt}",
                        "out of scoreboards",
                        result.out_of_scoreboards[0],
                    )
                cubin = Cubin.load(cubin_path)
                for slot, word in enumerate(result.words):
                    if program.instructions[slot].word is not None:
                        cubin.write_word(slot, word)
                out = Path(tmp) / "r.cubin"
                cubin.save(out)
                # content first: basalt emitted a control word `nvdisasm`
                # refused for a while, and this check read back an empty program
                # and passed. a check that passes on nothing is worse than none.
                written = disassemble_program(toolchain, out)
                if len(written.instructions) != len(program.instructions):
                    return (
                        f"{snippet.name} -O{opt}",
                        "did not disassemble after rescheduling",
                        f"{len(written.instructions)} of {len(program.instructions)} instructions",
                    )
                report = verify_program(written, model, observed=observed)
                if not report.ok:
                    return (
                        f"{snippet.name} -O{opt}",
                        "rejected its own schedule",
                        report.hazards[0].describe(),
                    )
                return None

        from basalt.harvest.corpus_shapes import generate_shapes

        snippets = generate() + generate_tensor() + generate_shapes()
        tasks = [(s, opt) for s in snippets for opt in (1, 2, 3)]
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
    def tightened(toolchain, model, observed):
        from basalt.sched.scheduler import schedule_program

        def one(task):
            snippet, opt = task
            with TemporaryDirectory(prefix="basalt-rb-") as tmp:
                src, cubin_path = Path(tmp) / "k.ptx", Path(tmp) / "k.cubin"
                src.write_text(snippet.ptx)
                built = toolchain.run(
                    [
                        str(toolchain.ptxas),
                        f"-arch={ARCH}",
                        f"-O{opt}",
                        "-o",
                        str(cubin_path),
                        str(src),
                    ],
                    check=False,
                    timeout=60.0,
                )
                if built.returncode != 0:
                    return []
                program = disassemble_program(toolchain, cubin_path)
                result = schedule_program(program, model, observed=observed)
                transfers = TestReadBarrierWindowsAreNotTightened.TRANSFERS
                targets = set(getattr(program, "labels", {}).values())

                found = []
                previous = -1
                for index, instruction in enumerate(program.instructions):
                    if instruction.word is None:
                        continue
                    if instruction.word.field("read_barrier") == NO_BARRIER:
                        continue
                    start = index
                    while start > previous + 1 and start not in targets:
                        earlier = program.instructions[start - 1]
                        if earlier.word is None or earlier.opcode in transfers:
                            break
                        start -= 1
                    window = range(start, index + 1)
                    vendor = sum(program.instructions[i].word.field("stall") for i in window)
                    ours = sum(result.words[i].field("stall") for i in window)
                    if ours < vendor:
                        found.append(
                            f"{snippet.name} -O{opt} slot {index}: {ours} < {vendor} cycles"
                        )
                    previous = index
                return found

        snippets = generate() + generate_tensor() + generate_shapes()
        tasks = [(s, opt) for s in snippets for opt in (1, 2, 3)]
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
            return [line for lines in pool.map(one, tasks) for line in lines]

    def test_no_read_barrier_window_is_shorter_than_the_vendors(self, tightened):
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
    def assembled(toolchain):
        from basalt.asm.assemble import Assembler, AssemblyError
        from basalt.harvest.corpus_shapes import generate_shapes
        from basalt.isa.database import IsaDatabase

        database = ROOT / "data" / "isa" / "sm_120a.json"
        if not database.is_file():
            pytest.skip("no ISA database; run `basalt build-isa`")
        assembler = Assembler(IsaDatabase.read(database))

        def one(snippet):
            with TemporaryDirectory(prefix="basalt-asm-") as tmp:
                src = Path(tmp) / "k.ptx"
                cubin = Path(tmp) / "k.cubin"
                src.write_text(snippet.ptx)
                built = toolchain.run(
                    [str(toolchain.ptxas), f"-arch={ARCH}", "-O3", "-o", str(cubin), str(src)],
                    check=False,
                    timeout=60.0,
                )
                if built.returncode != 0:
                    return []
                out = []
                for instruction in disassemble_program(toolchain, cubin).instructions:
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

        snippets = generate() + generate_tensor() + generate_shapes()
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
            return [row for rows in pool.map(one, snippets) for row in rows]

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
        # 88.8% when this was written. the floor is deliberately below that, so
        # it catches a regression without failing on a database that legitimately
        # learned to refuse something it had been guessing at.
        assert exact / total >= 0.85, f"only {exact}/{total} ({exact / total:.1%}) reproduced"


class TestWholeProgramAssembly:
    """Assemble each kernel as a program, with its labels resolved.

    A branch cannot be assembled alone: the field holds the distance to the
    destination, so the same text encodes differently in every kernel it appears
    in. Given the whole program that distance is known, and the standard is
    unchanged: reproduce the vendor's bytes or refuse.
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def assembled(toolchain):
        from basalt.asm.assemble import assemble_program
        from basalt.harvest.corpus_shapes import generate_shapes
        from basalt.isa.database import IsaDatabase

        database = ROOT / "data" / "isa" / "sm_120a.json"
        if not database.is_file():
            pytest.skip("no ISA database; run `basalt build-isa`")
        db = IsaDatabase.read(database)

        def one(snippet):
            with TemporaryDirectory(prefix="basalt-prog-") as tmp:
                src = Path(tmp) / "k.ptx"
                cubin = Path(tmp) / "k.cubin"
                src.write_text(snippet.ptx)
                built = toolchain.run(
                    [str(toolchain.ptxas), f"-arch={ARCH}", "-O3", "-o", str(cubin), str(src)],
                    check=False,
                    timeout=60.0,
                )
                if built.returncode != 0:
                    return (0, 0, 0)
                program = disassemble_program(toolchain, cubin)
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

        snippets = generate() + generate_tensor() + generate_shapes()
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
            rows = list(pool.map(one, snippets))
        return tuple(sum(column) for column in zip(*rows, strict=True))

    def test_nothing_assembles_to_the_wrong_bytes(self, assembled):
        _, wrong, _ = assembled
        assert wrong == 0, f"{wrong} instructions assembled to bytes the vendor did not emit"

    def test_resolving_labels_covers_the_branches(self, assembled):
        exact, wrong, refused = assembled
        total = exact + wrong + refused
        assert total > 5000, "the corpus did not build"
        # 99.2% when this was written. The floor sits a point below rather than
        # four, because a floor far under the real number lets coverage fall a
        # long way without anything going red.
        assert exact / total >= 0.98, f"only {exact}/{total} ({exact / total:.1%}) reproduced"


class TestTheBranchFieldIsStillWhereItWasFound:
    """Re-derive the branch encoding rather than trusting the constant.

    `BRANCH_TARGET_BITS` was solved from real kernels: the label table gives the
    destination, the instruction gives its address, the word gives the bits. It
    is a measurement, and a measurement written down as a constant is exactly
    the kind of thing that goes quietly wrong when a compiler version changes.
    """

    def test_every_branch_in_the_corpus_decodes_to_its_label(self, toolchain):
        import re

        from basalt.asm.assemble import branch_target
        from basalt.harvest.corpus_shapes import generate_shapes

        label = re.compile(r"`\(([^)]+)\)")

        def one(snippet):
            with TemporaryDirectory(prefix="basalt-branch-") as tmp:
                src = Path(tmp) / "k.ptx"
                cubin = Path(tmp) / "k.cubin"
                src.write_text(snippet.ptx)
                built = toolchain.run(
                    [str(toolchain.ptxas), f"-arch={ARCH}", "-O3", "-o", str(cubin), str(src)],
                    check=False,
                    timeout=60.0,
                )
                if built.returncode != 0:
                    return (0, 0)
                program = disassemble_program(toolchain, cubin)
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

        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
            rows = list(pool.map(one, generate() + generate_shapes()))
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
    def cycles(toolchain, model, observed):
        from basalt.harvest.corpus_shapes import generate_shapes
        from basalt.sched.scheduler import issue_cycles, schedule_program

        def one(snippet):
            with TemporaryDirectory(prefix="basalt-cost-") as tmp:
                src = Path(tmp) / "k.ptx"
                cubin = Path(tmp) / "k.cubin"
                src.write_text(snippet.ptx)
                built = toolchain.run(
                    [str(toolchain.ptxas), f"-arch={ARCH}", "-O3", "-o", str(cubin), str(src)],
                    check=False,
                    timeout=60.0,
                )
                if built.returncode != 0:
                    return (0, 0)
                program = disassemble_program(toolchain, cubin)
                result = schedule_program(program, model, observed=observed)
                if result.out_of_scoreboards:
                    return (0, 0)
                return (
                    issue_cycles([i.word for i in program.instructions], program.instructions),
                    issue_cycles(result.words, program.instructions),
                )

        snippets = generate() + generate_tensor() + generate_shapes()
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
            rows = list(pool.map(one, snippets))
        return sum(r[0] for r in rows), sum(r[1] for r in rows)

    def test_the_cost_is_known_and_bounded(self, cycles):
        vendor, basalt = cycles
        assert vendor > 5000, "the corpus did not build"
        ratio = basalt / vendor
        # 0.90x when this was written, pinned from both sides: slower is a
        # regression, and faster is a reason to distrust the costing rather
        # than to celebrate it (finding 12)
        assert ratio < 1.15, f"basalt's schedules cost {ratio:.2f}x the vendor's, up from 0.90x"
        assert ratio > 0.75, (
            f"basalt's schedules cost {ratio:.2f}x, which is far cheaper than the vendor's and "
            f"is the shape a costing bug takes. check the hardware round trip before believing it"
        )
