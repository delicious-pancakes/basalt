#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Reschedule the whole corpus from scratch and check it on the GPU.

The strongest control basalt has. For every kernel the corpus generates:
compile it with `ptxas`, throw away every control bit it produced, compute new
ones, write them back, and run both versions on the card with identical input.
The rescheduled kernel has to produce the same bytes.

This is the only control that is genuinely independent. The checker and the
scheduler share a latency model, so they share its blind spots and agree with
each other while both being wrong; only the silicon has no stake in it. Running
it over the whole corpus rather than a handful of hand-written kernels is what
turned a scheduler that looked finished into one with forty known-wrong kernels,
and every model correction since came out of it.

    python scripts/roundtrip_corpus.py
    python scripts/roundtrip_corpus.py --report out.json
    python scripts/roundtrip_corpus.py --only k_sqrt_approx_f32 k_brev_b32

**Run every optimisation level before believing a scheduler change**, not just the
default. They are not interchangeable: `-O3` unrolls a loop into ordinary
registers where `-O1` keeps its counter in uniform ones, so the uniform datapath
had no coverage at all until `-O1` was run, and it immediately found two kernels
basalt scheduled wrong. Both had passed everything else.

Needs an sm_120 card and a toolchain; see the README for both.

Three things this has to get right to mean anything:

*A kernel the vendor's own cubin cannot run here says nothing about basalt.*
Shared and local memory kernels need launch configuration this runner does not
provide. The vendor side runs first and a failure there is reported against the
harness, not the schedule.

*One input is one chance to notice.* A stale read only changes the answer when
the stale value and the fresh one differ, so a schedule can be wrong and a single
pattern of bytes can fail to show it. Every kernel runs against four patterns
chosen to disagree with each other everywhere, and their outputs are compared as
one result.

*A kernel whose output is not stable cannot be compared.* Several have 32 threads
storing to one address, so the winner is arbitrary. Others read shared or local
memory this runner never initialises, so they are stable until something else has
used the card and not afterwards. The vendor is therefore run again at the end,
after basalt has had it, and a kernel that no longer agrees with itself is
excluded rather than judged. Without that check every `LDSM` and `MOVMATRIX`
kernel reads as a basalt failure.

*A faulted CUDA context stays faulted, and the primary context is process-wide.*
So the work runs in a child process that exits on the first error, and the parent
restarts it after the kernel that died. Without that, one bad kernel makes every
later one look broken, which is exactly the false trail this script was first
read to produce.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import _repo

_repo.use_repo_source()

ROOT = _repo.ROOT

ARCH = "sm_120a"
# Several inputs rather than one, and the same ones for both schedules.
#
# A stale read only changes the answer when the stale value and the fresh one
# differ, so a single pattern of bytes is a single chance to notice. These four
# are chosen to disagree with each other everywhere: an odd stride, a different
# odd stride, a run of alternating extremes, and a quadratic.
PATTERNS: tuple[bytes, ...] = (
    bytes((i * 37 + 11) & 0xFF for i in range(256)),
    bytes((i * 211 + 173) & 0xFF for i in range(256)),
    bytes(0xFF if i % 3 else 0x00 for i in range(256)),
    bytes((i * i + 7) & 0xFF for i in range(256)),
)
BUFFER = 256
REPEATS = 5
THREADS = 32

# How many patterns the worker actually uses. It runs as a separate process, so
# `--patterns` reaches it through the environment rather than an argument.
#
# All four is the default and what any published number should come from.
# Fewer exists because the full sweep across three optimisation levels takes the
# better part of an hour and a contributor bisecting one kernel does not need
# all of it. Lowering it makes the run faster and the result weaker, in that
# order.
PATTERNS_ENV = "BASALT_PATTERNS"


@dataclass(frozen=True)
class Verdict:
    name: str
    outcome: str
    detail: str = ""


ORDER = (
    "match",
    "MISMATCH",
    "basalt-nondeterministic",
    "basalt-faulted",
    "vendor-unstable",
    "not-reproducible",
    "unrunnable",
    "unschedulable",
)


# --------------------------------------------------------------------- build


def build(work: Path, only: set[str] | None, opt: int) -> list[dict]:
    """Compile every corpus kernel and reschedule it, writing both cubins."""
    from basalt.asm.cubin import Cubin
    from basalt.disasm import disassemble_program
    from basalt.harvest.corpus import generate as generate_scalar
    from basalt.harvest.corpus_shapes import generate_shapes
    from basalt.harvest.corpus_tensor import generate_tensor
    from basalt.sched.scheduler import schedule_program
    from basalt.toolchain import find_toolchain
    from basalt.verify.latency import DEFAULT_MODEL, LatencyModel
    from basalt.verify.observed import ObservedStalls

    tc = find_toolchain()
    latencies = ROOT / "data" / "latency" / "rtx-5070-ti.json"
    observed_path = ROOT / "data" / "latency" / "observed-stalls-sm120a.json"
    model = LatencyModel.assumed().overlay(latencies) if latencies.is_file() else DEFAULT_MODEL
    observed = ObservedStalls.read(observed_path) if observed_path.is_file() else None

    # the shape kernels are included here and not in the harvest: they exist to
    # give the scheduler loops, barriers and branches to get wrong, which the
    # generated corpus deliberately does not have
    snippets = generate_scalar() + generate_tensor() + generate_shapes()
    if only:
        snippets = [s for s in snippets if s.name in only]
    work.mkdir(parents=True, exist_ok=True)

    def one(numbered):
        index, snippet = numbered
        src = work / f"{index:04d}.ptx"
        src.write_text(snippet.ptx)
        vendor = work / f"{index:04d}.v.cubin"
        built = tc.run(
            [str(tc.ptxas), f"-arch={ARCH}", f"-O{opt}", "-o", str(vendor), str(src)],
            check=False,
            timeout=60.0,
        )
        if built.returncode != 0:
            return None
        try:
            program = disassemble_program(tc, vendor)
            result = schedule_program(program, model, observed=observed)
        except Exception as exc:
            return {"i": index, "name": snippet.name, "skip": f"{type(exc).__name__}: {exc}"[:120]}
        if result.out_of_scoreboards:
            return {"i": index, "name": snippet.name, "skip": result.out_of_scoreboards[0][:120]}

        cubin = Cubin.load(vendor)
        for slot, word in enumerate(result.words):
            if program.instructions[slot].word is not None:
                cubin.write_word(slot, word)
        cubin.save(work / f"{index:04d}.b.cubin")
        entry = next(
            line.split("(")[0].split()[-1] for line in snippet.ptx.splitlines() if ".entry" in line
        )
        return {"i": index, "name": snippet.name, "entry": entry}

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        built = [r for r in pool.map(one, enumerate(snippets)) if r]
    (work / "manifest.json").write_text(json.dumps(built))
    return built


# -------------------------------------------------------------------- worker


def worker(work: Path, start: int) -> None:
    """Run kernels from `start` onward, one result line each, then exit on error."""
    from basalt.gpu.driver import Device

    manifest = json.loads((work / "manifest.json").read_text())

    def emit(index, outcome, name, detail=""):
        print(f"{index}\t{outcome}\t{name}\t{detail}", flush=True)

    patterns = PATTERNS[: int(os.environ.get(PATTERNS_ENV, len(PATTERNS)))]

    with Device(0) as device:
        source = device.alloc(BUFFER)
        destination = device.alloc(BUFFER)

        def run(path, entry):
            module = device.load_cubin(Path(path).read_bytes())
            function = module.function(entry)
            seen = []
            for _ in range(REPEATS):
                # every pattern joined into one result, so a disagreement on any
                # of them is a disagreement
                outputs = []
                for pattern in patterns:
                    device.upload(source, pattern)
                    device.upload(destination, b"\0" * BUFFER)
                    device.launch(
                        function,
                        [ctypes.c_size_t(source), ctypes.c_size_t(destination)],
                        block=(THREADS, 1, 1),
                    )
                    outputs.append(device.download(destination, BUFFER))
                seen.append(b"".join(outputs))
            module.unload()
            return seen

        for entry in manifest[start:]:
            index, name = entry["i"], entry["name"]
            if "skip" in entry:
                emit(index, "unschedulable", name, entry["skip"])
                continue
            try:
                vendor = run(work / f"{index:04d}.v.cubin", entry["entry"])
            except Exception as exc:
                emit(index, "unrunnable", name, type(exc).__name__)
                sys.stdout.flush()
                os._exit(1)
            if len(set(vendor)) != 1:
                emit(index, "vendor-unstable", name)
                continue
            try:
                ours = run(work / f"{index:04d}.b.cubin", entry["entry"])
            except Exception as exc:
                emit(index, "basalt-faulted", name, type(exc).__name__)
                sys.stdout.flush()
                os._exit(1)
            # Run the vendor again, after basalt has had the card, and before
            # judging basalt at all. A kernel that reads uninitialised shared or
            # local memory is stable while nothing else has touched it and not
            # stable once something has, so its first result is not ground truth
            # and its instability is not basalt's. Every `LDSM` and `MOVMATRIX`
            # kernel here is in that position and each looked like a basalt
            # failure until this check existed. Checked before basalt's own
            # determinism, because otherwise these land in that bucket instead.
            again = run(work / f"{index:04d}.v.cubin", entry["entry"])
            if again[0] != vendor[0]:
                emit(index, "not-reproducible", name)
                continue
            if len(set(ours)) != 1:
                emit(index, "basalt-nondeterministic", name)
                continue
            emit(index, "match" if ours[0] == vendor[0] else "MISMATCH", name)


# -------------------------------------------------------------------- driver


def drive(work: Path, manifest: list[dict]) -> list[Verdict]:
    """Run the worker repeatedly, stepping past whichever kernel killed it."""
    verdicts: dict[int, Verdict] = {}
    position = 0
    restarts = 0
    limit = len(manifest) + 8

    while position < len(manifest) and restarts < limit:
        restarts += 1
        finished = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                str(position),
                "--work",
                str(work),
            ],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        lines = [line for line in finished.stdout.splitlines() if "\t" in line]
        for line in lines:
            index, outcome, name, *rest = line.split("\t")
            verdicts[int(index)] = Verdict(name, outcome, rest[0] if rest else "")
        reached = max((int(line.split("\t")[0]) for line in lines), default=None)
        if reached is None:
            # died before reporting anything at all, so step over this one
            entry = manifest[position]
            verdicts[entry["i"]] = Verdict(entry["name"], "unrunnable", "no output")
            position += 1
        else:
            position = next(k for k, e in enumerate(manifest) if e["i"] == reached) + 1

    return [verdicts[k] for k in sorted(verdicts)]


# Outcomes that count as basalt having been given a fair chance. A fault on
# basalt's side belongs here and not among the exclusions: the vendor's kernel
# ran, basalt's crashed the context, and calling that "not comparable" would
# quietly move a failure out of the denominator.
COMPARABLE = ("match", "MISMATCH", "basalt-nondeterministic", "basalt-faulted")


def report(verdicts: list[Verdict]) -> int:
    counts = Counter(v.outcome for v in verdicts)
    comparable = sum(counts[outcome] for outcome in COMPARABLE)
    print(f"\n{_repo.provenance()}")
    print(f"\n{len(verdicts)} corpus kernels rescheduled and run\n")
    for outcome in ORDER:
        if counts[outcome]:
            print(f"  {outcome:24} {counts[outcome]:4}")
    if comparable:
        print(f"\n  {counts['match']} of {comparable} comparable kernels match the vendor exactly")

    for outcome in ("MISMATCH", "basalt-nondeterministic", "basalt-faulted"):
        named = [v.name for v in verdicts if v.outcome == outcome]
        if named:
            print(f"\n{outcome} ({len(named)}):")
            for name in named:
                print(f"  {name}")
    failures = comparable - counts["match"]
    return 0 if failures == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--work", type=Path, default=None, help="scratch directory for cubins")
    parser.add_argument("--report", type=Path, default=None, help="write the verdicts as JSON")
    parser.add_argument("--only", nargs="*", default=None, help="restrict to these kernel names")
    parser.add_argument(
        "--patterns",
        type=int,
        default=len(PATTERNS),
        choices=range(1, len(PATTERNS) + 1),
        help="how many input patterns to run each kernel against (default all of them). "
        "fewer is faster and weaker: a stale read only changes the answer when the stale "
        "value and the fresh one differ, so each pattern dropped is a chance to miss one",
    )
    parser.add_argument(
        "--opt",
        type=int,
        default=3,
        choices=(1, 2, 3),
        help="ptxas optimisation level to reschedule (default 3). -O0 is not offered: it "
        "emits a zeroed control word, so there is no schedule to replace",
    )
    parser.add_argument("--worker", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    work = args.work or Path(os.environ.get("TMP", "/tmp")) / f"basalt-roundtrip-O{args.opt}"
    # the worker is a separate process, so this rides across in the environment
    os.environ[PATTERNS_ENV] = str(args.patterns)

    if args.worker is not None:
        worker(work, args.worker)
        return 0

    manifest = build(work, set(args.only) if args.only else None, args.opt)
    if not manifest:
        raise SystemExit("nothing built; is the toolchain on PATH or BASALT_CUDA_BIN?")
    print(f"built {len(manifest)} vendor and rescheduled cubin pairs")

    verdicts = drive(work, manifest)
    if args.report:
        args.report.write_text(
            json.dumps(
                [{"name": v.name, "outcome": v.outcome, "detail": v.detail} for v in verdicts],
                indent=2,
            )
        )
        print(f"wrote {args.report}")
    return report(verdicts)


if __name__ == "__main__":
    sys.exit(main())
