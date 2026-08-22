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
from basalt.harvest.corpus import generate
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

        def one(snippet):
            with TemporaryDirectory(prefix="basalt-sched-") as tmp:
                src = Path(tmp) / "k.ptx"
                cubin_path = Path(tmp) / "k.cubin"
                src.write_text(snippet.ptx)
                built = toolchain.run(
                    [str(toolchain.ptxas), f"-arch={ARCH}", "-O3", "-o", str(cubin_path), str(src)],
                    check=False,
                    timeout=60.0,
                )
                if built.returncode != 0:
                    return None
                program = disassemble_program(toolchain, cubin_path)
                result = schedule_program(program, model, observed=observed)
                if result.out_of_scoreboards:
                    return (snippet.name, "out of scoreboards", result.out_of_scoreboards[0])
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
                        snippet.name,
                        "did not disassemble after rescheduling",
                        f"{len(written.instructions)} of {len(program.instructions)} instructions",
                    )
                report = verify_program(written, model, observed=observed)
                if not report.ok:
                    return (snippet.name, "rejected its own schedule", report.hazards[0].describe())
                return None

        snippets = generate() + generate_tensor()
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
            return [r for r in pool.map(one, snippets) if r]

    def test_every_kernel_can_be_scheduled_and_verifies_clean(self, scheduled):
        if scheduled:
            lines = "\n".join(f"  {name}: {why} ({detail})" for name, why, detail in scheduled)
            pytest.fail(f"{len(scheduled)} kernels basalt could not schedule cleanly:\n{lines}")
