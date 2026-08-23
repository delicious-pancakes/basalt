# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Tests that need real silicon.

The one in this file that matters is `test_verdicts_match_hardware`. Everything
else basalt does is an argument that a program is safe; this is the only test
that checks the argument against what the hardware actually does, by patching a
control bit, running the kernel, and comparing the answer to a reference.

Marked `gpu`, so the rest of the suite still runs on a machine without one.
"""

from __future__ import annotations

import ctypes
import struct
from pathlib import Path

import pytest

from basalt.asm.cubin import Cubin
from basalt.disasm import disassemble_program
from basalt.encoding import STALL_YIELD, effective_stall
from basalt.gpu.driver import Device, cuda_available
from basalt.gpu.latency import ChainSpec, measure_latency
from basalt.verify.hazards import verify_program
from basalt.verify.latency import Confidence, LatencyClass, LatencyModel

pytestmark = pytest.mark.gpu

ARCH = "sm_120a"

# A dependent chain of integer multiply-adds, short enough to reason about by
# hand and long enough that a stale read changes the answer.
CHAIN_PTX = """.version 9.0
.target sm_120a
.address_size 64
.visible .entry k(.param .u64 pin, .param .u64 pout)
{
    .reg .b32 %r<24>;
    .reg .b64 %in, %out;
    ld.param.u64 %in, [pin];
    cvta.to.global.u64 %in, %in;
    ld.param.u64 %out, [pout];
    cvta.to.global.u64 %out, %out;
    ld.global.u32 %r1, [%in];
    ld.global.u32 %r2, [%in+4];
    mad.lo.s32 %r3, %r1, %r2, %r1;
    mad.lo.s32 %r4, %r3, %r2, %r3;
    mad.lo.s32 %r5, %r4, %r2, %r4;
    mad.lo.s32 %r6, %r5, %r2, %r5;
    mad.lo.s32 %r7, %r6, %r2, %r6;
    mad.lo.s32 %r8, %r7, %r2, %r7;
    st.global.u32 [%out], %r8;
    ret;
}
"""

SEED_A, SEED_B, LINKS = 3, 5, 6


def _expected() -> int:
    value = SEED_A
    for _ in range(LINKS):
        value = (value * SEED_B + value) & 0xFFFF_FFFF
    return value


@pytest.fixture(scope="module")
def device():
    if not cuda_available():
        pytest.skip("no CUDA device")
    with Device(0) as dev:
        yield dev


@pytest.fixture(scope="module")
def chain_cubin(toolchain, tmp_path_factory) -> Path:
    tmp = tmp_path_factory.mktemp("basalt-gpu")
    src, cubin = tmp / "k.ptx", tmp / "k.cubin"
    src.write_text(CHAIN_PTX)
    res = toolchain.run(
        [str(toolchain.ptxas), f"-arch={ARCH}", "-O3", "-o", str(cubin), str(src)],
        check=False,
    )
    if res.returncode != 0:
        pytest.skip(f"ptxas could not build the chain: {res.stderr.strip()}")
    return cubin


def _run(dev: Device, cubin: Path, repeats: int = 20) -> set[int]:
    module = dev.load_cubin(cubin.read_bytes())
    fn = module.function("k")
    src, dst = dev.alloc(8), dev.alloc(4)
    dev.upload(src, struct.pack("<2I", SEED_A, SEED_B))
    seen: set[int] = set()
    for _ in range(repeats):
        dev.upload(dst, b"\0" * 4)
        dev.launch(fn, [ctypes.c_size_t(src), ctypes.c_size_t(dst)], block=(1, 1, 1))
        seen.add(struct.unpack("<I", dev.download(dst, 4))[0])
    module.unload()
    return seen


class TestDevice:
    def test_reports_an_sm120_part(self, device):
        assert device.info.compute_capability[0] >= 12
        assert device.info.multiprocessors > 0

    def test_kernel_round_trip(self, device, chain_cubin):
        assert _run(device, chain_cubin) == {_expected()}


class TestStallEncoding:
    """A stall of zero is a safe encoding, not zero cycles.

    Everything downstream of this depends on it: a checker that reads 0 as zero
    cycles calls correct programs broken, and `ptxas -O0` emits nothing but
    zeroes.
    """

    def test_zero_stall_is_safe(self, device, chain_cubin, tmp_path):
        producer = _first_chain_producer(chain_cubin, device)
        patched = _patch(chain_cubin, tmp_path, producer, STALL_YIELD)
        assert _run(device, patched) == {_expected()}

    def test_short_nonzero_stall_corrupts_silently(self, device, chain_cubin, tmp_path):
        """The premise of the whole project, checked on the hardware."""
        producer = _first_chain_producer(chain_cubin, device)
        patched = _patch(chain_cubin, tmp_path, producer, 1)
        results = _run(device, patched)
        assert results != {_expected()}, "a one-cycle stall should not have been enough"
        assert len(results) == 1, "the corruption is deterministic, not a race"

    def test_effective_stall_treats_zero_as_covering(self):
        assert effective_stall(1) == 1
        assert effective_stall(15) == 15
        assert effective_stall(0) > 64


class TestVerdictsMatchHardware:
    def test_verdicts_match_hardware(self, toolchain, device, chain_cubin, tmp_path):
        """Static verdict against observed behaviour, for every stall value.

        This is the test that makes the rest of basalt worth trusting. For each
        encodable stall on a dependent producer, the checker either passes the
        program or flags it, and the hardware either computes the right answer
        or does not. Those two have to agree every time.
        """
        model = _measured_model(toolchain, device)
        producer = _first_chain_producer(chain_cubin, device)
        expected = _expected()

        disagreements = []
        for stall in range(0, 8):
            patched = _patch(chain_cubin, tmp_path, producer, stall)
            predicted_safe = verify_program(disassemble_program(toolchain, patched), model).ok
            actually_safe = _run(device, patched) == {expected}
            if predicted_safe != actually_safe:
                disagreements.append(
                    f"stall={stall}: basalt said "
                    f"{'clean' if predicted_safe else 'hazard'}, hardware was "
                    f"{'correct' if actually_safe else 'wrong'}"
                )

        assert not disagreements, "\n".join(disagreements)


class TestMeasurement:
    def test_a_measured_latency_is_a_whole_number_of_cycles(self, toolchain, device):
        spec = ChainSpec(
            opcode="IMAD",
            ptx_type="s32",
            body="    mad.lo.s32 {d}, {d}, %ra, %rb;",
            seed="    ld.global.u32 %r1, [%in];",
        )
        # the production sweep, not a truncated one: a three-point fit trips the
        # R-squared gate, and the gate is not moving to suit a test
        result = measure_latency(toolchain, device, spec)
        assert result.ok, result.rejected
        assert result.r_squared > 0.999
        assert result.integral_error < 0.2

    def test_the_chain_length_used_is_the_one_observed(self, toolchain, device):
        """Points are (observed count, cycles), never (requested count, cycles)."""
        spec = ChainSpec(
            opcode="IMAD",
            ptx_type="s32",
            body="    mad.lo.s32 {d}, {d}, %ra, %rb;",
            seed="    ld.global.u32 %r1, [%in];",
        )
        result = measure_latency(toolchain, device, spec, lengths=(64, 128))
        counts = [count for count, _ in result.points]
        assert len(set(counts)) == len(counts), "chain lengths must differ to give a slope"


# ---- helpers -----------------------------------------------------------


def _patch(cubin: Path, tmp_path: Path, index: int, stall: int) -> Path:
    cb = Cubin.load(cubin)
    cb.patch_control(index, "stall", stall)
    out = tmp_path / f"stall_{stall}.cubin"
    cb.save(out)
    return out


def _first_chain_producer(cubin: Path, device) -> int:
    """The first IMAD whose result the next IMAD consumes."""
    from basalt.toolchain import find_toolchain
    from basalt.verify.operands import operand_access

    tc = find_toolchain()
    program = disassemble_program(tc, cubin)
    imads = [i for i, ins in enumerate(program.instructions) if ins.word and ins.opcode == "IMAD"]
    for i in imads:
        produced = operand_access(
            program.instructions[i].mnemonic, program.instructions[i].operands
        ).real_defs
        for j in imads:
            if j <= i:
                continue
            used = operand_access(
                program.instructions[j].mnemonic, program.instructions[j].operands
            ).real_uses
            if produced & used:
                return i
    pytest.skip("no dependent IMAD pair in the compiled chain")
    raise AssertionError("unreachable")


def _measured_model(toolchain, device) -> LatencyModel:
    """A model whose IMAD entry is measured, so findings are errors not warnings."""
    spec = ChainSpec(
        opcode="IMAD",
        ptx_type="s32",
        body="    mad.lo.s32 {d}, {d}, %ra, %rb;",
        seed="    ld.global.u32 %r1, [%in];",
    )
    measured = measure_latency(toolchain, device, spec)
    if not measured.ok:
        pytest.skip(f"could not measure IMAD: {measured.rejected}")

    model = LatencyModel.assumed()
    records = dict(model.records)
    records["IMAD"] = type(records["IMAD"])(
        cycles=measured.rounded,
        kind=LatencyClass.FIXED,
        confidence=Confidence.MEASURED,
        note="measured for this test",
        source=device.info.name,
    )
    return LatencyModel(records=records, sku=device.info.name)
