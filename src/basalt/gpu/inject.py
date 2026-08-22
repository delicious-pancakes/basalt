# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Finding the required stall by breaking the program on purpose.

Two earlier methods each answer a slightly wrong question. Timing a dependent
chain measures `max(latency, initiation interval)`, so a rate-limited unit reads
high. Mining what the compiler schedules measures what the compiler believes,
which is an upper bound on the requirement and says nothing where the compiler
was merely cautious.

This measures the requirement itself. Take a kernel with a dependent pair, set
the producer's stall to a candidate value, run it, and compare the answer to a
reference. The smallest value that still computes correctly is, by definition,
the smallest gap the hardware tolerates.

The reference comes from the safe stall encoding rather than from the compiler's
own schedule, so the comparison does not assume the compiler was right.

Two things this deliberately does not do. It does not trust a single launch: a
value is only accepted as safe if every repeat agrees, because a marginal stall
can be right most of the time. And it does not bisect, despite the temptation:
safety is not guaranteed to be monotone in the stall count, and a bisection over
a non-monotone predicate silently returns nonsense. Sixteen values is a small
enough space to walk exhaustively and be certain.
"""

from __future__ import annotations

import ctypes
import struct
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

from ..asm.cubin import Cubin
from ..disasm import disassemble_program
from ..encoding import STALL_YIELD
from ..toolchain import Toolchain
from ..verify.operands import operand_access
from .driver import Device
from .latency import ChainSpec, _build_kernel

__all__ = ["STALL_MAX", "InjectionResult", "probe_required_stall"]

# The stall field is four bits, so this is the longest gap a single instruction
# can express. A latency above it has to be covered by several instructions or
# by a scoreboard, which is exactly what ptxas does for fp64.
STALL_MAX = 15

DEFAULT_REPEATS = 12


@dataclass
class InjectionResult:
    """What the hardware tolerated for one dependent pair."""

    opcode: str
    required: int | None
    producer_index: int
    consumer_index: int
    scheduled: int
    span: int = 1
    # True when the producer signals a scoreboard the consumer waits on. The
    # stall is then irrelevant to correctness, so sweeping it measures nothing
    # about latency and the result must not be read as one.
    scoreboarded: bool = False
    # Whether the kernel can distinguish a skipped link at all, established
    # independently of the stall sweep by shortening the chain. Without this,
    # "no value produced a wrong answer" is ambiguous between "every value is
    # genuinely safe" and "this kernel cannot tell", which are opposite claims.
    sensitive: bool = True
    safe: list[int] = field(default_factory=list)
    unsafe: list[int] = field(default_factory=list)
    note: str = ""
    rejected: str = ""

    @property
    def ok(self) -> bool:
        return not self.rejected and self.required is not None

    @property
    def monotone(self) -> bool:
        """True when every value at or above the requirement is safe.

        Worth reporting rather than assuming: it is the property that would let
        a future version bisect, and a counterexample would be interesting in
        its own right.
        """
        if self.required is None:
            return True
        return all(v < self.required for v in self.unsafe)

    def describe(self) -> str:
        if self.rejected:
            return f"{self.opcode:<8} rejected: {self.rejected}"
        if self.scoreboarded:
            return (
                f"{self.opcode:<8} covered by a scoreboard, so the stall does not "
                f"carry the dependency and no requirement is measurable this way"
            )
        req = ">15" if self.required is None else str(self.required)
        flag = "" if self.monotone else "  NON-MONOTONE"
        if not self.unsafe:
            # sensitivity was established separately, so this is a real result:
            # the smallest encodable stall is genuinely enough
            flag += "  (no value was unsafe)"
        window = f", span {self.span}" if self.span > 1 else ""
        return (
            f"{self.opcode:<8} requires {req:>3} cycles  "
            f"(ptxas leaves {self.scheduled}{window}, "
            f"largest unsafe {max(self.unsafe) if self.unsafe else 'none'}){flag}"
        )


def _find_dependent_pair(program, opcode: str) -> tuple[int, int] | None:
    """The first `opcode` instruction and the next instruction consuming it.

    The consumer need not be adjacent. For a long-latency instruction ptxas pads
    the gap with NOPs at maximum stall, and those NOPs are part of what covers
    the dependency, so the span between the two is what has to be controlled.
    """
    produced_by: dict[int, set] = {}
    for i, ins in enumerate(program.instructions):
        if ins.word is None:
            continue
        if ins.opcode == opcode:
            defs = operand_access(ins.mnemonic, ins.operands).real_defs
            if defs:
                produced_by[i] = defs

    for i, produced in produced_by.items():
        for j in range(i + 1, len(program.instructions)):
            consumer = program.instructions[j]
            if consumer.word is None:
                continue
            used = operand_access(consumer.mnemonic, consumer.operands).real_uses
            if produced & used:
                return i, j
            # a later write to the same register ends the dependency
            if produced & operand_access(consumer.mnemonic, consumer.operands).real_defs:
                break
    return None


def _run(dev: Device, cubin: bytes, payload: bytes, repeats: int) -> set[bytes]:
    module = dev.load_cubin(cubin)
    fn = module.function("chain")
    src, dst = dev.alloc(len(payload)), dev.alloc(16)
    dev.upload(src, payload)
    seen: set[bytes] = set()
    for _ in range(repeats):
        dev.upload(dst, b"\0" * 16)
        dev.launch(fn, [ctypes.c_size_t(src), ctypes.c_size_t(dst)], block=(1, 1, 1))
        # byte 4 onward is the chain's result; byte 0 is the cycle count, which
        # legitimately varies between launches
        seen.add(dev.download(dst, 16)[4:])
    module.unload()
    return seen


def probe_required_stall(
    tc: Toolchain,
    dev: Device,
    spec: ChainSpec,
    *,
    arch: str = "sm_120a",
    links: int = 8,
    repeats: int = DEFAULT_REPEATS,
) -> InjectionResult:
    """Walk every encodable stall for one opcode and see what the silicon accepts."""
    # Values chosen so that skipping one link changes the answer unmistakably.
    # An operand near the identity for the operation, such as a float multiplier
    # close to 1.0, lets a stale read round back to the same result: the probe
    # then sees no difference and reports a requirement of 1 for an instruction
    # that plainly needs more. That is a statement about the test, not the
    # hardware, which is why the sensitivity check below exists as well.
    payload = (
        struct.pack("<4I", 3, 5, 7, 11)
        + struct.pack("<2I", 3, 1)
        + struct.pack("<2f", 1.75, 3.25)
        + struct.pack("<2d", 1.75, 3.25)
    ).ljust(64, b"\0")

    with TemporaryDirectory(prefix="basalt-inject-") as tmp:
        tmpdir = Path(tmp)
        src, cubin = tmpdir / "chain.ptx", tmpdir / "chain.cubin"
        src.write_text(_build_kernel(spec, links, arch))

        res = tc.run(
            [str(tc.ptxas), f"-arch={arch}", "-O3", "-o", str(cubin), str(src)],
            check=False,
            timeout=120.0,
        )
        if res.returncode != 0:
            first = next(
                (ln for ln in (res.stderr + res.stdout).splitlines() if "error" in ln.lower()),
                "ptxas rejected the chain",
            )
            return InjectionResult(spec.opcode, None, -1, -1, -1, rejected=first.strip())

        program = disassemble_program(tc, cubin)
        pair = _find_dependent_pair(program, spec.opcode)
        if pair is None:
            return InjectionResult(
                spec.opcode,
                None,
                -1,
                -1,
                -1,
                rejected=f"no adjacent dependent {spec.opcode} pair survived compilation",
            )
        producer, consumer = pair
        scheduled = sum(
            program.instructions[i].word.field("stall")
            for i in range(producer, consumer)
            if program.instructions[i].word is not None
        )

        # if the compiler covered this dependency with a scoreboard, the stall
        # is not what makes it safe, and sweeping the stall would report a
        # meaninglessly small "requirement"
        producer_word = program.instructions[producer].word
        consumer_word = program.instructions[consumer].word
        barrier = producer_word.field("write_barrier") if producer_word else 7
        waited = bool(
            consumer_word and barrier != 7 and (consumer_word.field("wait_mask") >> barrier) & 1
        )

        # Sensitivity control: a chain one link shorter must produce a
        # different answer. If it does not, this kernel cannot detect a stale
        # read and the sweep below would be measuring nothing.
        shorter = tmpdir / "shorter.cubin"
        shorter_src = tmpdir / "shorter.ptx"
        shorter_src.write_text(_build_kernel(spec, links - 1, arch))
        sensitive = True
        if (
            tc.run(
                [str(tc.ptxas), f"-arch={arch}", "-O3", "-o", str(shorter), str(shorter_src)],
                check=False,
                timeout=120.0,
            ).returncode
            == 0
        ):
            full = _run(dev, cubin.read_bytes(), payload, repeats=2)
            fewer = _run(dev, shorter.read_bytes(), payload, repeats=2)
            sensitive = full != fewer

        # the reference comes from the safe encoding, not from the compiler's
        # own schedule, so this does not assume the compiler was right
        reference_path = tmpdir / "reference.cubin"
        cb = Cubin.load(cubin)
        cb.patch_control(producer, "stall", STALL_YIELD)
        cb.save(reference_path)
        reference = _run(dev, reference_path.read_bytes(), payload, repeats)
        if len(reference) != 1:
            return InjectionResult(
                spec.opcode,
                None,
                producer,
                consumer,
                scheduled,
                rejected="the reference run was not deterministic, so nothing can be compared",
            )

        # every instruction from the producer up to the consumer contributes,
        # so the budget is spread across the whole span rather than dropped on
        # the producer alone. anything else leaves ptxas's NOP padding in place
        # and measures nothing.
        span = list(range(producer, consumer))
        ceiling = STALL_MAX * len(span)

        safe: list[int] = [STALL_YIELD]
        unsafe: list[int] = []
        sweep = () if waited else range(1, ceiling + 1)
        for total in sweep:
            patched = tmpdir / f"total_{total}.cubin"
            cb = Cubin.load(cubin)
            remaining = total
            for index in span:
                give = min(STALL_MAX, remaining)
                # 0 is the safe encoding, so a span slot with nothing left to
                # give must still be 1, not 0, or the test defeats itself
                cb.patch_control(index, "stall", max(1, give))
                remaining = max(0, remaining - give)
            cb.save(patched)
            results = _run(dev, patched.read_bytes(), payload, repeats)
            # every repeat must agree, and agree with the reference: a stall
            # that is right most of the time is not right
            if results == reference:
                safe.append(total)
                if len(safe) > 3:
                    # the requirement is found; a few more confirm monotonicity
                    break
            else:
                unsafe.append(total)

    positive = [s for s in safe if s != STALL_YIELD]
    required = min(positive) if positive else None
    return InjectionResult(
        opcode=spec.opcode,
        required=required,
        producer_index=producer,
        consumer_index=consumer,
        scheduled=scheduled,
        span=max(1, consumer - producer),
        scoreboarded=waited,
        safe=safe,
        unsafe=unsafe,
        note=spec.note,
        sensitive=sensitive,
        rejected=_rejection(required, waited, ceiling, sensitive),
    )


def _rejection(
    required: int | None,
    scoreboarded: bool,
    ceiling: int,
    sensitive: bool,
) -> str:
    """Why no requirement was produced, distinguishing the interesting cases."""
    if not sensitive:
        # shortening the chain did not change the answer, so this kernel cannot
        # detect a skipped link and nothing it reports means anything
        return "the chain cannot detect a skipped link, so it establishes no requirement"
    if scoreboarded:
        # not a failure: the dependency really is covered, just not by the field
        # being swept
        return ""
    if required is None:
        return f"no accumulated stall up to {ceiling} was sufficient"
    return ""
