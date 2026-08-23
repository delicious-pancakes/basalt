# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Rescheduling real kernels and checking the answer on hardware.

This is the closing loop. basalt measures the latencies, builds a model, assigns
the control bits from that model, checks its own work with the verifier, and then
runs the result on the GPU next to the vendor compiler's version of the same
kernel. If the two disagree, something in that chain is wrong, and no amount of
internal consistency makes up for it.

`ptxas`'s control bits are discarded entirely and recomputed; only the
instruction order and the read barriers are kept. Order, because reordering would
make it impossible to attribute a behaviour change to the control bits. Read
barriers, because basalt has no measured model for how long a late operand read
takes and inventing one would be worse than preserving something known to work.

Marked `gpu`. Nothing here is meaningful without silicon: a scheduler that
satisfies its own checker and produces wrong numbers is precisely the failure
this exists to catch, and it has caught it twice.
"""

from __future__ import annotations

import ctypes
import struct
from dataclasses import dataclass
from pathlib import Path

import pytest

from basalt.asm.cubin import Cubin
from basalt.disasm import disassemble_program
from basalt.gpu.driver import Device, cuda_available
from basalt.sched.scheduler import schedule_program
from basalt.verify.hazards import verify_program
from basalt.verify.latency import DEFAULT_MODEL, LatencyModel
from basalt.verify.observed import ObservedStalls

pytestmark = pytest.mark.gpu

ARCH = "sm_120a"
ROOT = Path(__file__).resolve().parent.parent
LATENCIES = ROOT / "data" / "latency" / "rtx-5070-ti.json"
OBSERVED = ROOT / "data" / "latency" / "observed-stalls-sm120a.json"

TEMPLATE = """.version 9.0
.target sm_120a
.address_size 64
.visible .entry k(.param .u64 pin, .param .u64 pout)
{{
  .reg .b32 %r<16>; .reg .f32 %f<16>; .reg .f64 %g<16>;
  .reg .pred %p<4>; .reg .b64 %in, %out;
  ld.param.u64 %in,[pin]; cvta.to.global.u64 %in,%in;
  ld.param.u64 %out,[pout]; cvta.to.global.u64 %out,%out;
{body}
  ret;
}}
"""


@dataclass(frozen=True)
class Case:
    name: str
    in_format: str
    inputs: tuple
    out_format: str
    body: str
    # Why this case is known not to round-trip yet. Recorded as an expected
    # failure rather than deleted: a limitation that still runs is one that
    # reports the day it is fixed, and one that is removed is one nobody
    # remembers. Strict, so an unexpected pass fails the suite as well.
    known_gap: str = ""


CASES: tuple[Case, ...] = (
    Case(
        "integer-chain",
        "<2I",
        (3, 5),
        "<I",
        """
  ld.global.u32 %r1,[%in]; ld.global.u32 %r2,[%in+4];
  mad.lo.s32 %r3,%r1,%r2,%r1; mad.lo.s32 %r4,%r3,%r2,%r3;
  mad.lo.s32 %r5,%r4,%r2,%r4; mad.lo.s32 %r6,%r5,%r2,%r5;
  st.global.u32 [%out],%r6;""",
    ),
    Case(
        "float-mix",
        "<2f",
        (1.75, 3.25),
        "<f",
        """
  ld.global.f32 %f1,[%in]; ld.global.f32 %f2,[%in+4];
  fma.rn.f32 %f3,%f1,%f2,%f1; mul.f32 %f4,%f3,%f3; add.f32 %f5,%f4,%f3;
  fma.rn.f32 %f6,%f5,%f2,%f4;
  st.global.f32 [%out],%f6;""",
    ),
    Case(
        "branch",
        "<2I",
        (9, 4),
        "<I",
        """
  ld.global.u32 %r1,[%in]; ld.global.u32 %r2,[%in+4];
  setp.gt.s32 %p1,%r1,%r2;
  @%p1 bra BIG;
  mul.lo.s32 %r3,%r1,3; bra DONE;
BIG: add.s32 %r3,%r1,%r2;
DONE: mad.lo.s32 %r4,%r3,%r2,%r3;
  st.global.u32 [%out],%r4;""",
    ),
    Case(
        "loop",
        "<2I",
        (6, 3),
        "<I",
        """
  ld.global.u32 %r1,[%in]; ld.global.u32 %r2,[%in+4];
  mov.u32 %r3,0;
L: mad.lo.s32 %r3,%r3,%r2,%r1; sub.s32 %r1,%r1,1;
  setp.gt.s32 %p1,%r1,0; @%p1 bra L;
  st.global.u32 [%out],%r3;""",
    ),
    Case(
        "double",
        "<2d",
        (1.5, 2.25),
        "<d",
        """
  ld.global.f64 %g1,[%in]; ld.global.f64 %g2,[%in+8];
  fma.rn.f64 %g3,%g1,%g2,%g1; fma.rn.f64 %g4,%g3,%g2,%g3;
  add.f64 %g5,%g4,%g3;
  st.global.f64 [%out],%g5;""",
    ),
    Case(
        "shifts-and-logic",
        "<2I",
        (0xABCD1234, 7),
        "<I",
        """
  ld.global.u32 %r1,[%in]; ld.global.u32 %r2,[%in+4];
  shl.b32 %r3,%r1,%r2; xor.b32 %r4,%r3,%r1; shr.b32 %r5,%r4,%r2;
  and.b32 %r6,%r5,%r4; popc.b32 %r7,%r6; add.s32 %r8,%r7,%r6;
  st.global.u32 [%out],%r8;""",
    ),
    Case(
        "transcendental",
        "<2f",
        (2.5, 7.0),
        "<f",
        """
  ld.global.f32 %f1,[%in]; ld.global.f32 %f2,[%in+4];
  rcp.approx.f32 %f3,%f1; sqrt.approx.f32 %f4,%f2;
  fma.rn.f32 %f5,%f3,%f4,%f1; mul.f32 %f6,%f5,%f4;
  st.global.f32 [%out],%f6;""",
    ),
)


@pytest.fixture(scope="module")
def device():
    if not cuda_available():
        pytest.skip("no CUDA device")
    with Device(0) as dev:
        yield dev


@pytest.fixture(scope="module")
def model() -> LatencyModel:
    return LatencyModel.assumed().overlay(LATENCIES) if LATENCIES.is_file() else DEFAULT_MODEL


@pytest.fixture(scope="module")
def observed() -> ObservedStalls | None:
    return ObservedStalls.read(OBSERVED) if OBSERVED.is_file() else None


def _run(dev: Device, cubin: bytes, payload: bytes, out_format: str, repeats: int) -> set:
    module = dev.load_cubin(cubin)
    fn = module.function("k")
    src, dst = dev.alloc(len(payload)), dev.alloc(16)
    dev.upload(src, payload)
    size = struct.calcsize(out_format)
    seen = set()
    for _ in range(repeats):
        dev.upload(dst, b"\0" * 16)
        dev.launch(fn, [ctypes.c_size_t(src), ctypes.c_size_t(dst)], block=(1, 1, 1))
        seen.add(struct.unpack(out_format, dev.download(dst, 16)[:size])[0])
    module.unload()
    return seen


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(c, marks=pytest.mark.xfail(strict=True, reason=c.known_gap))
        if c.known_gap
        else c
        for c in CASES
    ],
    ids=lambda c: c.name,
)
def test_rescheduled_kernel_matches_the_vendor_schedule(
    toolchain, device, model, observed, case, tmp_path
):
    """Discard ptxas's control bits, compute new ones, and compare on hardware."""
    src = tmp_path / f"{case.name}.ptx"
    original = tmp_path / f"{case.name}.cubin"
    src.write_text(TEMPLATE.format(body=case.body))
    built = toolchain.run(
        [str(toolchain.ptxas), f"-arch={ARCH}", "-O3", "-o", str(original), str(src)],
        check=False,
    )
    if built.returncode != 0:
        pytest.skip(f"ptxas rejected {case.name}: {built.stderr.strip()}")

    program = disassemble_program(toolchain, original)
    result = schedule_program(program, model, observed)
    assert not result.out_of_scoreboards, result.out_of_scoreboards

    cubin = Cubin.load(original)
    for index, word in enumerate(result.words):
        if program.instructions[index].word is not None:
            cubin.write_word(index, word)
    rescheduled = tmp_path / f"{case.name}.resched.cubin"
    cubin.save(rescheduled)

    # content is checked first: this once read back an empty program and called
    # it clean, and an assertion that passes on nothing is worse than none
    written = disassemble_program(toolchain, rescheduled)
    assert len(written.instructions) == len(program.instructions), (
        f"the rescheduled cubin disassembled to {len(written.instructions)} instructions "
        f"where the original had {len(program.instructions)}, so everything below would be "
        f"looking at nothing"
    )
    report = verify_program(written, model, observed=observed)
    assert report.ok, "\n".join(h.describe() for h in report.hazards)

    payload = struct.pack(case.in_format, *case.inputs).ljust(32, b"\0")
    reference = _run(device, original.read_bytes(), payload, case.out_format, repeats=20)
    produced = _run(device, rescheduled.read_bytes(), payload, case.out_format, repeats=30)

    assert len(reference) == 1, f"the vendor schedule was not deterministic: {reference}"
    assert len(produced) == 1, (
        f"basalt's schedule is not deterministic ({sorted(produced)}), "
        f"which means a dependency is uncovered, not merely mistimed"
    )
    assert produced == reference, (
        f"basalt scheduled {case.name} to a different answer: "
        f"{sorted(produced)} against {sorted(reference)}. {result.summary()}"
    )
