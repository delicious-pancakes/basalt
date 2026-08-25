# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Measuring instruction latency on the actual silicon.

The verifier is only as good as its latency model, and every number in that
model starts out assumed. This is what replaces them.

The method is a dependent chain. Build a kernel in which each instance of the
instruction under test consumes the previous one's result, time the chain with
the cycle counter, and the slope of cycles against chain length is the latency.
One warp, one block, so nothing overlaps and issue latency is what dominates.

Three things keep it honest, and all three matter:

*Slope, not a single timing.* Measuring one chain conflates the instruction's
latency with the fixed cost of the clock reads and the launch. Fitting a line
across several lengths cancels every constant exactly, whatever it happens to be.

*The chain is counted, never assumed.* `ptxas` is free to fold a long run of
arithmetic into something shorter, and a measurement that silently divides by a
chain length that no longer exists is worse than no measurement. Every kernel is
disassembled and its instances of the target opcode counted, and the count that
goes into the fit is the one observed in the SASS.

*Fit quality is reported.* A clean dependent chain is almost perfectly linear.
An R-squared below the threshold means something interfered, and the result is
withheld rather than published with a caveat nobody reads.
"""

from __future__ import annotations

import ctypes
import json
import statistics
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

from ..architecture import ArchitectureError, require_architecture_match
from ..disasm import disassemble_cubin
from ..toolchain import Toolchain
from ..verify.latency import LatencyClass
from .driver import Device

__all__ = ["ChainSpec", "LatencyMeasurement", "MeasurementRun", "measure_all", "measure_latency"]

# Chain lengths to fit across. Short enough that the whole sweep is quick,
# long enough that the fixed overhead is a small part of the longest run.
DEFAULT_LENGTHS = (64, 128, 256, 512)

# Repeats per length; the minimum is taken because the cycle counter only ever
# reads high from interference, never low.
DEFAULT_REPEATS = 7

# Below this, the chain did not behave linearly and the number is not reported.
MIN_R_SQUARED = 0.999

# A dependent chain of one instruction should land on a whole cycle. Further
# than this from one means the chain is measuring something else as well.
MAX_INTEGRAL_ERROR = 0.2


@dataclass(frozen=True, slots=True)
class ChainSpec:
    """How to build a dependent chain for one opcode.

    `body` is the PTX for a single link, with `{d}` for the accumulator being
    threaded through. Operands come from memory so `ptxas` cannot constant fold
    the chain, and the operation is chosen to resist algebraic collapse.
    """

    opcode: str
    ptx_type: str
    body: str
    seed: str
    kind: LatencyClass = LatencyClass.FIXED
    note: str = ""
    # a conversion cannot chain alone, so the slope covers the round trip and
    # says so rather than halving it
    covers: tuple[str, ...] = ()

    @property
    def counted(self) -> tuple[str, ...]:
        return self.covers or (self.opcode,)

    @property
    def is_composite(self) -> bool:
        return len(self.counted) > 1


_SPECS: tuple[ChainSpec, ...] = (
    ChainSpec(
        opcode="IMAD",
        ptx_type="s32",
        body="    mad.lo.s32 {d}, {d}, %ra, %rb;",
        seed="    ld.global.u32 %r1, [%in];",
        note="integer multiply-add, the workhorse of the integer pipe",
    ),
    ChainSpec(
        opcode="IADD3",
        ptx_type="s32",
        # three-input add resists folding better than a two-input one, which
        # ptxas will happily collapse into a single multiply
        body="    add.s32 {d}, {d}, %ra;\n    add.s32 {d}, {d}, %rb;",
        seed="    ld.global.u32 %r1, [%in];",
        note="integer add",
    ),
    ChainSpec(
        opcode="FFMA",
        ptx_type="f32",
        # floating point is not associative, so the compiler cannot reassociate
        # the chain even in principle
        body="    fma.rn.f32 {d}, {d}, %fa, %fb;",
        seed="    ld.global.f32 %f1, [%in];",
        note="single precision fused multiply-add",
    ),
    ChainSpec(
        opcode="FADD",
        ptx_type="f32",
        body="    add.f32 {d}, {d}, %fa;",
        seed="    ld.global.f32 %f1, [%in];",
        note="single precision add",
    ),
    ChainSpec(
        opcode="FMUL",
        ptx_type="f32",
        body="    mul.f32 {d}, {d}, %fa;",
        seed="    ld.global.f32 %f1, [%in];",
        note="single precision multiply",
    ),
    # stated rather than left at the fixed default, or the next measurement run
    # silently overwrites a corrected class
    ChainSpec(
        opcode="DADD",
        ptx_type="f64",
        body="    add.f64 {d}, {d}, %ga;",
        seed="    ld.global.f64 %g1, [%in];",
        kind=LatencyClass.VARIABLE,
        note="fp64 add; scoreboarded, and the wait is what carries the dependency",
    ),
    ChainSpec(
        opcode="DFMA",
        ptx_type="f64",
        body="    fma.rn.f64 {d}, {d}, %ga, %gb;",
        seed="    ld.global.f64 %g1, [%in];",
        kind=LatencyClass.VARIABLE,
        note="fp64 fused multiply-add; scoreboarded, see DADD",
    ),
    ChainSpec(
        opcode="LOP3",
        ptx_type="b32",
        # an xor followed by an and collapses into a single lookup; 0x96 is
        # a^b^c, three runtime inputs with no algebraic simplification available
        body="    lop3.b32 {d}, {d}, %ra, %rb, 0x96;",
        seed="    ld.global.u32 %r1, [%in];",
        note="three-input bitwise lookup",
    ),
    ChainSpec(
        opcode="SHF",
        ptx_type="b32",
        body="    shf.l.wrap.b32 {d}, {d}, {d}, 3;",
        seed="    ld.global.u32 %r1, [%in];",
        note="funnel shift",
    ),
    ChainSpec(
        opcode="POPC",
        ptx_type="b32",
        # popc of a popc closes the loop on its own; the value converging at once
        # does not matter, the dependency is what is timed
        body="    popc.b32 {d}, {d};",
        seed="    ld.global.u32 %r1, [%in];",
        kind=LatencyClass.VARIABLE,
        note="population count",
    ),
    ChainSpec(
        # sm_120 spells the integer-to-float conversion I2FP, not I2F
        opcode="I2FP",
        # the accumulator threaded through the chain is the integer, so the
        # register class here has to be the integer one
        ptx_type="s32",
        body="    cvt.rn.f32.s32 %f2, {d};\n    cvt.rzi.s32.f32 {d}, %f2;",
        seed="    ld.global.u32 %r1, [%in];",
        covers=("I2FP", "F2I"),
        note="round trip through float. a conversion cannot feed the next link "
        "without converting back, so this slope covers the pair and is deliberately "
        "not split between them",
    ),
    ChainSpec(
        opcode="MUFU",
        ptx_type="f32",
        body="    rcp.approx.f32 {d}, {d};",
        seed="    ld.global.f32 %f1, [%in];",
        kind=LatencyClass.VARIABLE,
        note="special function unit",
    ),
)


_KERNEL = """.version 9.0
.target {arch}
.address_size 64

.visible .entry chain(.param .u64 pin, .param .u64 pout)
{{
    .reg .b32 %r<8>;
    .reg .f32 %f<8>;
    .reg .f64 %g<8>;
    .reg .b32 %ra, %rb, %t0, %t1, %t2;
    .reg .f32 %fa, %fb;
    .reg .f64 %ga, %gb;
    .reg .b64 %in, %out;

    ld.param.u64  %in, [pin];
    cvta.to.global.u64 %in, %in;
    ld.param.u64  %out, [pout];
    cvta.to.global.u64 %out, %out;

    ld.global.u32 %ra, [%in+16];
    ld.global.u32 %rb, [%in+20];
    ld.global.f32 %fa, [%in+24];
    ld.global.f32 %fb, [%in+28];
    ld.global.f64 %ga, [%in+32];
    ld.global.f64 %gb, [%in+40];
{seed}

    mov.u32 %t0, %clock;
{chain}
    mov.u32 %t1, %clock;

    sub.s32 %t2, %t1, %t0;
    st.global.u32 [%out], %t2;
{sink}
    ret;
}}
"""

_SINKS = {
    "s32": "    st.global.u32 [%out+4], %r1;",
    "b32": "    st.global.u32 [%out+4], %r1;",
    "f32": "    st.global.f32 [%out+4], %f1;",
    "f64": "    st.global.f64 [%out+8], %g1;",
}
_ACCUM = {"s32": "%r1", "b32": "%r1", "f32": "%f1", "f64": "%g1"}


@dataclass
class LatencyMeasurement:
    """One opcode's measured latency, with the evidence behind it."""

    opcode: str
    cycles: float
    kind: LatencyClass
    r_squared: float
    points: list[tuple[int, int]] = field(default_factory=list)
    note: str = ""
    rejected: str = ""
    covers: tuple[str, ...] = ()

    @property
    def is_composite(self) -> bool:
        """True when the slope covers more than one instruction.

        A composite result is real but is not one opcode's latency, so it must
        never be written into the model as though it were.
        """
        return len(self.covers) > 1

    @property
    def ok(self) -> bool:
        return not self.rejected

    @property
    def rounded(self) -> int:
        """The nearest integer, because latencies are whole cycles.

        Rounding up instead would turn a measured 4.01 into 5 and make the
        verifier reject correct compiler output, which is a worse failure than
        it sounds: the tool would be wrong precisely where it is most trusted.
        Measurements that do not land near an integer are rejected instead, by
        `integral_error`, rather than being quietly rounded into place.
        """
        return max(1, round(self.cycles))

    @property
    def integral_error(self) -> float:
        """How far the measurement sits from the nearest whole cycle."""
        return abs(self.cycles - round(self.cycles))

    def describe(self) -> str:
        if self.rejected:
            return f"{self.opcode:<8} rejected: {self.rejected}"
        label = "+".join(self.covers) if self.is_composite else self.opcode
        return (
            f"{label:<11} {self.cycles:6.2f} cycles  "
            f"(reported {self.rounded}, R2 {self.r_squared:.5f}, {len(self.points)} points)"
        )


@dataclass
class MeasurementRun:
    """A full sweep, labelled with the part it ran on."""

    sku: str
    arch: str
    multiprocessors: int
    clock_khz: int
    cuda_version: str
    # the driver reports the GPU, not the board partner. provenance rather than
    # input: these latencies are cycles, which an overclock does not move
    board: str = ""
    measurements: list[LatencyMeasurement] = field(default_factory=list)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "sku": self.sku,
            "board": self.board,
            "arch": self.arch,
            "multiprocessors": self.multiprocessors,
            "clock_khz": self.clock_khz,
            "cuda_version": self.cuda_version,
            "latencies": {
                m.opcode: {
                    "cycles": m.rounded,
                    "measured_cycles": round(m.cycles, 3),
                    "kind": m.kind.value,
                    "r_squared": round(m.r_squared, 6),
                    "note": m.note,
                }
                for m in self.measurements
                if m.ok and not m.is_composite
            },
            # measured, real, and deliberately kept out of "latencies": a pair
            # timed together is not either opcode's latency
            "composite": {
                "+".join(m.covers): {
                    "cycles": m.rounded,
                    "measured_cycles": round(m.cycles, 3),
                    "r_squared": round(m.r_squared, 6),
                    "note": m.note,
                }
                for m in self.measurements
                if m.ok and m.is_composite
            },
            "rejected": {m.opcode: m.rejected for m in self.measurements if not m.ok},
            "raw": [asdict(m) for m in self.measurements],
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _build_kernel(spec: ChainSpec, length: int, arch: str) -> str:
    accum = _ACCUM[spec.ptx_type]
    link = spec.body.replace("{d}", accum)
    chain = "\n".join(link for _ in range(length))
    return _KERNEL.format(
        arch=arch,
        seed=spec.seed,
        chain=chain,
        sink=_SINKS[spec.ptx_type],
    )


def _fit(points: list[tuple[int, int]]) -> tuple[float, float]:
    """Least-squares slope of cycles against observed chain length."""
    xs = [float(x) for x, _ in points]
    ys = [float(y) for _, y in points]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    if sxx == 0:
        return 0.0, 0.0
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys, strict=True))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot else 0.0
    return slope, r2


def measure_latency(
    tc: Toolchain,
    dev: Device,
    spec: ChainSpec,
    *,
    arch: str = "sm_120a",
    lengths: tuple[int, ...] = DEFAULT_LENGTHS,
    repeats: int = DEFAULT_REPEATS,
) -> LatencyMeasurement:
    """Measure one opcode by fitting cycles against verified chain length."""
    points: list[tuple[int, int]] = []

    with TemporaryDirectory(prefix="basalt-latency-") as tmp:
        tmpdir = Path(tmp)
        for length in lengths:
            src = tmpdir / f"chain_{length}.ptx"
            cubin = tmpdir / f"chain_{length}.cubin"
            src.write_text(_build_kernel(spec, length, arch))

            res = tc.run(
                [str(tc.ptxas), f"-arch={arch}", "-O3", "-o", str(cubin), str(src)],
                check=False,
                timeout=300.0,
            )
            if res.returncode != 0:
                first = next(
                    (ln for ln in (res.stderr + res.stdout).splitlines() if "error" in ln.lower()),
                    "ptxas rejected the chain",
                )
                return LatencyMeasurement(
                    spec.opcode, 0.0, spec.kind, 0.0, note=spec.note, rejected=first.strip()
                )

            # count what actually survived, rather than trusting the request
            listing = disassemble_cubin(tc, cubin)
            counts = {op: sum(1 for i in listing if i.opcode.upper() == op) for op in spec.counted}

            if missing := [op for op, k in counts.items() if k == 0]:
                return LatencyMeasurement(
                    spec.opcode,
                    0.0,
                    spec.kind,
                    0.0,
                    note=spec.note,
                    covers=spec.counted,
                    rejected=f"no {', '.join(missing)} survived compilation at length {length}",
                )
            # a slope is only attributable to a group if the group stays in step
            if len(set(counts.values())) != 1:
                return LatencyMeasurement(
                    spec.opcode,
                    0.0,
                    spec.kind,
                    0.0,
                    note=spec.note,
                    covers=spec.counted,
                    rejected=f"covered opcodes appear in unequal numbers: {counts}",
                )
            observed = next(iter(counts.values()))

            cycles = _run_chain(dev, cubin.read_bytes(), repeats)
            points.append((observed, cycles))

    if len({x for x, _ in points}) < 2:
        return LatencyMeasurement(
            spec.opcode,
            0.0,
            spec.kind,
            0.0,
            points=points,
            note=spec.note,
            rejected="chain length did not vary after compilation, so no slope exists",
        )

    slope, r2 = _fit(points)
    measurement = LatencyMeasurement(
        opcode=spec.opcode,
        cycles=slope,
        kind=spec.kind,
        r_squared=r2,
        points=points,
        note=spec.note,
        covers=spec.counted,
    )
    if r2 < MIN_R_SQUARED:
        measurement.rejected = f"fit is not linear enough (R2 {r2:.4f} < {MIN_R_SQUARED})"
    elif slope <= 0:
        measurement.rejected = f"non-positive slope ({slope:.3f})"
    elif measurement.integral_error > MAX_INTEGRAL_ERROR:
        measurement.rejected = (
            f"{slope:.3f} cycles is {measurement.integral_error:.3f} from a whole cycle, "
            "so the chain is probably not measuring one instruction"
        )
    return measurement


def _run_chain(dev: Device, cubin: bytes, repeats: int) -> int:
    """Launch the chain and return the smallest cycle count seen.

    The minimum rather than the mean: the counter can only read high, from
    interference the measurement is not trying to capture, so the smallest
    observation is the closest to the quantity of interest.
    """
    module = dev.load_cubin(cubin)
    fn = module.function("chain")

    # inputs are deliberately unremarkable values that will not denormal or
    # saturate over a long chain
    payload = struct.pack("<4I", 3, 5, 7, 11) + struct.pack("<2I", 3, 1)
    payload += struct.pack("<2f", 1.0000001, 0.9999999)
    payload += struct.pack("<2d", 1.0000001, 0.9999999)
    payload = payload.ljust(64, b"\0")

    src = dev.alloc(len(payload))
    dst = dev.alloc(16)
    dev.upload(src, payload)

    best = None
    for _ in range(repeats):
        dev.upload(dst, b"\0" * 16)
        dev.launch(
            fn,
            [ctypes.c_size_t(src), ctypes.c_size_t(dst)],
            grid=(1, 1, 1),
            block=(1, 1, 1),
        )
        (cycles,) = struct.unpack("<I", dev.download(dst, 16)[:4])
        best = cycles if best is None else min(best, cycles)

    module.unload()
    return int(best or 0)


def measure_all(
    tc: Toolchain,
    *,
    arch: str = "sm_120a",
    ordinal: int = 0,
    specs: tuple[ChainSpec, ...] = _SPECS,
    lengths: tuple[int, ...] = DEFAULT_LENGTHS,
    repeats: int = DEFAULT_REPEATS,
    board: str = "",
    progress: bool = True,
) -> MeasurementRun:
    """Sweep every chain spec on the device and collect the results."""
    with Device(ordinal) as dev:
        info = dev.info
        try:
            require_architecture_match(arch, info.arch, f"CUDA device {ordinal}")
        except ArchitectureError:
            # Re-raise the typed authority failure before compiling or launching
            # anything for a device that cannot execute the requested target.
            raise
        run = MeasurementRun(
            sku=info.name,
            board=board,
            arch=info.arch,
            multiprocessors=info.multiprocessors,
            clock_khz=info.clock_khz,
            cuda_version=tc.version,
        )
        if progress:
            print(f"measuring on {info.describe()}")

        for spec in specs:
            measurement = measure_latency(
                tc, dev, spec, arch=arch, lengths=lengths, repeats=repeats
            )
            run.measurements.append(measurement)
            if progress:
                print(f"  {measurement.describe()}")

    return run
