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
- It is not a proof. The compiler is conservative when a kernel gives it nothing
  to fill a gap with, and a corpus of two-instruction kernels reads high for
  exactly that reason: `FADD` mined at 5 cycles from 24 such observations and at
  4 from 90 once the corpus had workloads in it, where 4 is what fault injection
  had measured all along (finding 21).
- So the observation count is the evidence, and a thin one is not evidence at
  all. `CS2R` had three observations, said sixteen, and measures four. Pairs
  below the threshold are recorded and not trusted, and coverage is reported.
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
MIN_OBSERVATIONS = 8


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
    # what a scoreboarded producer still owes its consumer. keyed on the full
    # mnemonic and the consumer, because collapsing either errs toward corruption
    by_scoreboarded: dict[tuple[str, str], StallEvidence] = field(default_factory=dict)
    kernels: int = 0

    def observe(self, producer: str, consumer: str, stall: int, sample: str) -> None:
        key = (producer, consumer)
        if key not in self.by_pair:
            self.by_pair[key] = StallEvidence(producer, consumer, minimum=stall)
        self.by_pair[key].observe(stall, sample)

        # collapsed over consumers, for a per-opcode model. guards collapse
        # separately under `@OPCODE`, or one drags the other down to its minimum
        collapsed = self._collapse_key(producer, consumer)
        if collapsed not in self.by_producer:
            self.by_producer[collapsed] = StallEvidence(producer, "*", minimum=stall)
        self.by_producer[collapsed].observe(stall, sample)

    def observe_scoreboarded(self, mnemonic: str, consumer: str, cycles: int, sample: str) -> None:
        """Record the gap between a waited-on scoreboarded producer and its use.

        Takes the producer's full mnemonic, modifiers and all, because the
        modifier is what decides the number: `I2F.RP` needs one cycle and every
        other `I2F` needs two.
        """
        # keyed on the producer's full mnemonic: the modifier decides the number
        key = (mnemonic, consumer)
        if key not in self.by_scoreboarded:
            self.by_scoreboarded[key] = StallEvidence(mnemonic, consumer, minimum=cycles)
        self.by_scoreboarded[key].observe(cycles, sample)

    def scoreboarded_minimum(
        self, mnemonic: str, consumer: str | None = None
    ) -> StallEvidence | None:
        """Cycles a waited-on scoreboarded producer still needs before its use.

        A wait covers the long, variable part of a result. It does not cover the
        whole of it, and how much is left over depends on the exact form and on
        what reads it.

        Keyed on both ends, because both decide the number.

        The producer's exact form matters: `I2F.RP` needs 1 where every other
        `I2F` needs 2, `MUFU.RCP64H` needs 1 where the rest of `MUFU` needs 2,
        and `ATOMG`'s float and min/max forms need 4 where the plain one needs
        1. So does the consumer: `MUFU.SQRT` into an `FMUL` is not the same
        question as `FLO.U32` into a `SHFL` twenty-eight cycles later.

        Failing an exact pairing, the answer is the smallest gap anything in the
        same family was ever given, first across this form's other consumers and
        then across the whole opcode. The tightest gap the compiler ever leaves
        is the closest thing to a statement of a requirement; a wide one usually
        means it had other work to fit in, and reading a requirement off it
        produces confident nonsense.

        No `trusted` gate. A single observation is thin evidence for a lower
        bound in general, but for an exact pairing it is the only evidence there
        is, and declining to use it left `I2F.F64` and `FLO.U32` a cycle short of
        what the hardware needs.
        """
        if consumer is not None and (exact := self.by_scoreboarded.get((mnemonic, consumer))):
            return exact
        same_form = [e for (m, _), e in self.by_scoreboarded.items() if m == mnemonic]
        if same_form:
            return min(same_form, key=lambda e: e.minimum)
        bare = mnemonic.split(".")[0]
        family = [e for (m, _), e in self.by_scoreboarded.items() if m.split(".")[0] == bare]
        return min(family, key=lambda e: e.minimum) if family else None

    @staticmethod
    def _collapse_key(producer: str, consumer: str) -> str:
        """Where a pairing's evidence is collapsed to, per producer.

        Guard and non-guard evidence never share a bucket.
        """
        return f"@{producer}" if consumer.startswith("@") else producer

    def requirement(self, producer: str, consumer: str | None = None) -> StallEvidence | None:
        """The tightest trusted gap seen for a pairing, or for the producer alone.

        `producer` is a full mnemonic. The modifier decides the number here just
        as it does for a scoreboarded producer: `IMAD.HI.U32` is a wide multiply
        and is never scheduled closer than 5 cycles to its consumer, while plain
        `IMAD` is scheduled at 3. Collapsing them onto `IMAD` takes the smaller,
        which is how basalt emitted an integer divide that computed 10 where the
        vendor computed 8.

        Four places are tried, most specific first: this exact form and consumer,
        this form against any consumer, the bare opcode and this consumer, the
        bare opcode against any consumer. The bare-opcode steps are what keep a
        form the compiler never emitted checkable at all.

        A guard consumer is spelled `@OPCODE` and only ever matches guard
        evidence, at every step.
        """
        bare = producer.split(".")[0]
        if consumer is not None:
            # an exact pairing however thin, because falling through to the bare
            # opcode answers with a different instruction's requirement
            if (pair := self.by_pair.get((producer, consumer))) is not None:
                return pair
            if (pair := self.by_pair.get((bare, consumer))) is not None and pair.trusted:
                return pair
        for name in (producer, bare):
            key = self._collapse_key(name, consumer) if consumer is not None else name
            evidence = self.by_producer.get(key)
            if evidence is not None and evidence.trusted:
                return evidence
        return None

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
                    "by_scoreboarded": {
                        f"{m}=>{c}": {
                            "min_stall": e.minimum,
                            "observations": e.observations,
                            "trusted": e.trusted,
                            "samples": e.samples,
                        }
                        for (m, c), e in sorted(self.by_scoreboarded.items())
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
        for key, entry in raw.get("by_scoreboarded", {}).items():
            mnemonic, _, consumer = key.partition("=>")
            ev = StallEvidence(mnemonic, consumer, minimum=entry["min_stall"])
            ev.observations = entry["observations"]
            ev.samples = entry.get("samples", [])
            out.by_scoreboarded[(mnemonic, consumer)] = ev
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
        # per definition, not per scoreboard: a counter means a later producer on
        # the same index does not undo an earlier wait
        signalled: dict[int, int] = {}
        satisfied: set[int] = set()

        for index in range(block.start, block.end):
            instr = program.instructions[index]
            if instr.word is None:
                continue
            access = operand_access(instr.mnemonic, instr.operands)

            # a wait takes effect before this instruction reads its operands
            wait_mask = instr.word.field("wait_mask")
            satisfied |= {producer for producer, sb in signalled.items() if (wait_mask >> sb) & 1}

            # only what the instruction demonstrably reads: this records a minimum,
            # so pairing a call with everything live logs the nearest register
            for reg in access.real_uses:
                if (previous := last_def.get(reg)) is None:
                    continue
                # a guard is resolved at issue rather than at operand read, so
                # it needs a different amount of lead and is mined separately.
                # `@IMAD` and `IMAD` are two requirements, not one.
                consumer_key = ("@" if reg == access.guard else "") + instr.opcode
                producer_index, producer_mnemonic = previous
                word = program.instructions[producer_index].word
                if word is None:
                    continue
                sample = f"{program.instructions[producer_index].text} -> {instr.text}"
                barrier = word.field("write_barrier")
                if barrier != 7:
                    # its own keyspace, and only where the barrier was already
                    # waited on. any instruction's wait counts, as in the checker
                    if producer_index in satisfied:
                        into.observe_scoreboarded(
                            producer_mnemonic, consumer_key, elapsed.get(reg, 0), sample
                        )
                    continue
                into.observe(producer_mnemonic, consumer_key, elapsed.get(reg, 0), sample)

            stall = effective_stall(instr.word.field("stall"))
            for key in list(elapsed):
                elapsed[key] += stall

            if (mine := instr.word.field("write_barrier")) != 7:
                signalled[index] = mine
                satisfied.discard(index)

            for reg in access.real_defs:
                last_def[reg] = (index, instr.mnemonic)
                elapsed[reg] = stall


def mine_corpus(
    tc,
    *,
    arch: str = "sm_120a",
    # every level that schedules. -O0 zeroes the control word, and -O1 keeps a
    # loop counter in the uniform datapath that -O3 unrolls away
    opt_levels: tuple[int, ...] = (1, 2, 3),
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
    from ..harvest.corpus_shapes import generate_shapes
    from ..harvest.corpus_tensor import generate_tensor

    # the shape kernels are mined too: an unmined pairing falls back to a number
    # collapsed over other consumers, which takes whichever tolerates least
    snippets = generate_scalar() + generate_tensor() + generate_shapes()
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
