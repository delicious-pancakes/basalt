# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Learning the required stall from what the vendor compiler actually schedules.

Timing a dependent chain measures how long a link takes, which is not quite the
question a checker asks. A chain measures `max(latency, initiation interval)`,
and for a rate-limited unit those are different numbers: the double-precision
and conversion pipes on a consumer part are narrow enough that the chain is
bounded by how fast the unit accepts work rather than by how long a result takes
to appear.

The measured conversion round trip is 24 cycles. `ptxas` schedules the same
producer and consumer 6 cycles apart. Both are correct answers to different
questions, and only the second one is what the hardware requires.

So the requirement is mined rather than assumed. Across a large corpus of
compiled kernels, for every dependent pair the compiler emitted, record the
stall it left between them. The smallest gap it ever leaves for a given pair is
its belief about the minimum safe distance, and since the compiler is the
reference the checker is already validated against, that belief is the best
available statement of the requirement.

What this is and is not:

- It is an empirical lower bound on the required stall, from the only source
  that definitively knows it.
- It is not a proof. The compiler could be conservative, in which case the true
  requirement is lower and basalt is merely strict. If the compiler were ever
  wrong in the other direction the positive control would already be failing.
- It only covers pairs the compiler actually emitted. Coverage is reported, and
  pairs with too few observations are not trusted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..disasm import Program
from ..encoding import effective_stall
from .cfg import build_cfg
from .operands import operand_access

__all__ = ["ObservedStalls", "StallEvidence", "mine_program"]

# A pair seen only once could be a coincidence of scheduling rather than a
# statement about the requirement. Below this, the observation is kept but not
# treated as authoritative.
MIN_OBSERVATIONS = 3


@dataclass
class StallEvidence:
    """What the compiler was seen to do for one producer/consumer pairing."""

    producer: str
    consumer: str
    minimum: int
    observations: int = 0
    samples: list[str] = field(default_factory=list)

    @property
    def trusted(self) -> bool:
        return self.observations >= MIN_OBSERVATIONS

    def observe(self, stall: int, sample: str) -> None:
        if stall < self.minimum:
            self.minimum = stall
            # keep the example that produced the tightest gap, since that is
            # the one a reader will want to check by hand
            self.samples = [sample]
        elif stall == self.minimum and len(self.samples) < 3:
            self.samples.append(sample)
        self.observations += 1


@dataclass
class ObservedStalls:
    """Minimum stall the compiler leaves between each dependent pair."""

    cuda_version: str
    arch: str
    by_pair: dict[tuple[str, str], StallEvidence] = field(default_factory=dict)
    by_producer: dict[str, StallEvidence] = field(default_factory=dict)
    kernels: int = 0

    def observe(self, producer: str, consumer: str, stall: int, sample: str) -> None:
        key = (producer, consumer)
        if key not in self.by_pair:
            self.by_pair[key] = StallEvidence(producer, consumer, minimum=stall)
        self.by_pair[key].observe(stall, sample)

        # the same evidence collapsed over consumers, which is what a
        # single-number-per-opcode latency model needs
        if producer not in self.by_producer:
            self.by_producer[producer] = StallEvidence(producer, "*", minimum=stall)
        self.by_producer[producer].observe(stall, sample)

    def requirement(self, producer: str, consumer: str | None = None) -> StallEvidence | None:
        """The tightest trusted gap seen for a pairing, or for the producer alone."""
        pair = self.by_pair.get((producer, consumer)) if consumer is not None else None
        if pair is not None and pair.trusted:
            return pair
        evidence = self.by_producer.get(producer)
        return evidence if evidence and evidence.trusted else None

    def summary(self) -> str:
        trusted = sum(1 for e in self.by_producer.values() if e.trusted)
        return (
            f"{self.kernels} kernels, {len(self.by_pair)} distinct pairs, "
            f"{trusted}/{len(self.by_producer)} producers with enough observations"
        )

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "cuda_version": self.cuda_version,
                    "arch": self.arch,
                    "kernels": self.kernels,
                    "min_observations": MIN_OBSERVATIONS,
                    "note": (
                        "minimum stall ptxas was observed to leave between a dependent "
                        "producer and consumer. an empirical lower bound on the required "
                        "distance, not a proof of it."
                    ),
                    "by_producer": {
                        name: {
                            "min_stall": e.minimum,
                            "observations": e.observations,
                            "trusted": e.trusted,
                            "samples": e.samples,
                        }
                        for name, e in sorted(self.by_producer.items())
                    },
                    "by_pair": {
                        f"{p}->{c}": {
                            "min_stall": e.minimum,
                            "observations": e.observations,
                            "trusted": e.trusted,
                        }
                        for (p, c), e in sorted(self.by_pair.items())
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )

    @classmethod
    def read(cls, path: Path) -> ObservedStalls:
        raw = json.loads(path.read_text())
        out = cls(cuda_version=raw.get("cuda_version", ""), arch=raw.get("arch", ""))
        out.kernels = raw.get("kernels", 0)
        for name, entry in raw.get("by_producer", {}).items():
            ev = StallEvidence(name, "*", minimum=entry["min_stall"])
            ev.observations = entry["observations"]
            ev.samples = entry.get("samples", [])
            out.by_producer[name] = ev
        for key, entry in raw.get("by_pair", {}).items():
            producer, _, consumer = key.partition("->")
            ev = StallEvidence(producer, consumer, minimum=entry["min_stall"])
            ev.observations = entry["observations"]
            out.by_pair[(producer, consumer)] = ev
        return out


def mine_program(program: Program, into: ObservedStalls) -> None:
    """Record every dependent pair the compiler emitted in one kernel.

    Only pairs inside a block are used. A gap that spans a branch depends on
    which path was taken, so it says nothing definite about the requirement.
    """
    if not program.instructions:
        return
    into.kernels += 1

    cfg = build_cfg(program)
    for block in cfg.blocks:
        # last writer of each register, and the stall accumulated since
        last_def: dict[object, tuple[int, str]] = {}
        elapsed: dict[object, int] = {}

        for index in range(block.start, block.end):
            instr = program.instructions[index]
            if instr.word is None:
                continue
            access = operand_access(instr.mnemonic, instr.operands)

            for reg in access.real_uses:
                if (previous := last_def.get(reg)) is None:
                    continue
                producer_index, producer_opcode = previous
                word = program.instructions[producer_index].word
                if word is None:
                    continue
                # a scoreboarded producer is covered by the wait, not the stall,
                # so its gap carries no information about a latency requirement
                if word.field("write_barrier") != 7:
                    continue
                into.observe(
                    producer_opcode,
                    instr.opcode,
                    elapsed.get(reg, 0),
                    f"{program.instructions[producer_index].text} -> {instr.text}",
                )

            stall = effective_stall(instr.word.field("stall"))
            for key in list(elapsed):
                elapsed[key] += stall

            for reg in access.real_defs:
                last_def[reg] = (index, instr.opcode)
                elapsed[reg] = stall


def mine_corpus(
    tc,
    *,
    arch: str = "sm_120a",
    # -O3 only, deliberately. At -O0 ptxas does not run its scheduling pass at
    # all: every control word comes out zeroed, no stalls and no scoreboards,
    # and the code still runs correctly because a zero stall is the safe
    # encoding. Mining that would record a requirement of zero for everything.
    opt_levels: tuple[int, ...] = (3,),
    jobs: int | None = None,
    progress: bool = True,
) -> ObservedStalls:
    """Compile the whole generated corpus and mine every kernel it produces.

    Deliberately reuses the harvest corpus rather than a bespoke one: the point
    is to observe the compiler across as many instruction pairings as it can be
    provoked into emitting, which is the same thing the corpus was built for.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from ..disasm import disassemble_program
    from ..harvest.corpus import generate as generate_scalar
    from ..harvest.corpus_tensor import generate_tensor

    snippets = generate_scalar() + generate_tensor()
    tasks = [(s, o) for s in snippets for o in opt_levels]
    if progress:
        print(f"mining {len(tasks)} kernels for scheduling decisions")

    def compile_one(task) -> Program | None:
        snippet, opt = task
        with TemporaryDirectory(prefix="basalt-mine-") as tmp:
            src = Path(tmp) / "k.ptx"
            cubin = Path(tmp) / "k.cubin"
            src.write_text(snippet.ptx)
            res = tc.run(
                [str(tc.ptxas), f"-arch={arch}", f"-O{opt}", "-o", str(cubin), str(src)],
                check=False,
                timeout=60.0,
            )
            if res.returncode != 0 or not cubin.exists():
                return None
            return disassemble_program(tc, cubin)

    out = ObservedStalls(cuda_version=tc.version, arch=arch)
    workers = jobs or min(32, (os.cpu_count() or 4) * 2)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for program in pool.map(compile_one, tasks):
            done += 1
            if program is not None:
                mine_program(program, out)
            if progress and done % 200 == 0:
                print(f"  {done}/{len(tasks)} kernels, {len(out.by_pair)} pairs")
    return out
