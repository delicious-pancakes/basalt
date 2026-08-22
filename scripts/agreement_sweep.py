#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Does the checker agree with the silicon about what is broken?

The positive control says basalt stays quiet on correct code. The negative
control says it complains about one deliberately broken instruction. Both are
run on a handful of kernels, and neither asks the question that matters most:

    when basalt says a schedule is unsafe, is it?

This asks it across the corpus. For every kernel, take the vendor's own working
schedule, shorten one stall on a real dependency, and collect two independent
verdicts: what basalt says statically, and what the GPU computes. Then count the
four cases.

| | GPU agrees with the reference | GPU computes something else |
| :--- | :--- | :--- |
| basalt: clean | agreed safe | **missed**, the bad one |
| basalt: hazard | over-strict | agreed broken |

The bottom-left cell is a false alarm and costs credibility. The top-right cell
is a silent miss and costs correctness, which is the whole product, so it is the
number to drive to zero.

Neither is free of judgement. A kernel can be understalled and still return the
right answer, because a stale read only changes the result when the stale value
and the fresh one differ, so an "over-strict" verdict is not automatically wrong.
That is why the two are counted separately and named rather than folded into an
accuracy percentage.

    python scripts/agreement_sweep.py
    python scripts/agreement_sweep.py --limit 40 --report sweep.json

Needs an sm_120 card and a toolchain.
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

# The same four input patterns the round trip uses, taken from the one place
# they are defined. A stale read only changes the answer when the stale value
# and the fresh one differ, so a single pattern is a single chance to notice and
# an "over-strict" verdict reached on one is mostly a claim about that pattern.
from roundtrip_corpus import PATTERNS

_repo.use_repo_source()

ROOT = _repo.ROOT

ARCH = "sm_120a"
BUFFER = 256
REPEATS = 5
THREADS = 32
# a broken kernel can hang the card, and an unbounded worker outlives its run
WORKER_TIMEOUT = 300.0


@dataclass(frozen=True)
class Case:
    name: str
    verdict: str  # what basalt said: "hazard" or "clean"
    behaviour: str  # what the GPU did: "same", "different", "faulted", "unstable"

    @property
    def cell(self) -> str:
        if self.behaviour not in ("same", "different"):
            return self.behaviour
        if self.verdict == "hazard":
            return "agreed broken" if self.behaviour == "different" else "over-strict"
        return "MISSED" if self.behaviour == "different" else "agreed safe"


def _shortenable(program, model, observed):
    """A dependency in this kernel whose stall can be cut, and by how much.

    Picks a fixed-latency producer whose consumer is covered only by elapsed
    cycles. A scoreboarded one is not a candidate: the wait covers it whatever
    the stall says, so shortening proves nothing about either the checker or the
    hardware.
    """
    from basalt.encoding import NO_BARRIER
    from basalt.verify.cfg import build_cfg
    from basalt.verify.latency import LatencyClass
    from basalt.verify.operands import operand_access

    for block in build_cfg(program).blocks:
        last: dict = {}
        for index in range(block.start, block.end):
            instruction = program.instructions[index]
            if instruction.word is None:
                continue
            access = operand_access(instruction.mnemonic, instruction.operands)
            for register in access.real_uses:
                producer = last.get(register)
                if producer is None:
                    continue
                word = program.instructions[producer].word
                if word is None or word.field("write_barrier") != NO_BARRIER:
                    continue
                if model.lookup(program.instructions[producer].opcode).kind is not (
                    LatencyClass.FIXED
                ):
                    continue
                if word.field("stall") >= 2 and index == producer + 1:
                    return producer, word.field("stall")
            for register in access.real_defs:
                last[register] = index
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--limit", type=int, default=0, help="stop after this many kernels")
    parser.add_argument("--report", type=Path, default=None, help="write the cases as JSON")
    parser.add_argument("--work", type=Path, default=None, help="scratch directory")
    parser.add_argument("--worker", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    work = args.work or Path(os.environ.get("TMP", "/tmp")) / "basalt-agreement"
    if args.worker is not None:
        return _worker(work, args.worker)

    cases = _build(work, args.limit)
    if not cases:
        raise SystemExit("nothing built; is the toolchain on PATH or BASALT_CUDA_BIN?")
    print(f"built {len(cases)} broken kernels beside their references")

    results = _drive(work, cases)
    counts = Counter(case.cell for case in results)
    print(f"\n{_repo.provenance()}")
    print(f"\n{len(results)} kernels, one dependency shortened in each\n")
    for cell in ("agreed broken", "agreed safe", "over-strict", "MISSED"):
        if counts[cell]:
            print(f"  {cell:16} {counts[cell]:4}")
    for cell in ("faulted", "unstable", "unrunnable"):
        if counts[cell]:
            print(f"  {cell:16} {counts[cell]:4}   (excluded)")

    missed = [case.name for case in results if case.cell == "MISSED"]
    if missed:
        print(f"\nMISSED ({len(missed)}): basalt called these clean and the GPU disagreed")
        for name in missed[:20]:
            print(f"  {name}")

    if args.report:
        args.report.write_text(
            json.dumps(
                [{"name": c.name, "verdict": c.verdict, "behaviour": c.behaviour} for c in results],
                indent=2,
            )
        )
        print(f"\nwrote {args.report}")
    return 1 if missed else 0


def _build(work: Path, limit: int) -> list[dict]:
    from basalt.asm.cubin import Cubin
    from basalt.disasm import disassemble_program
    from basalt.harvest.corpus import generate as generate_scalar
    from basalt.harvest.corpus_shapes import generate_shapes
    from basalt.toolchain import find_toolchain
    from basalt.verify.hazards import Severity, verify_program
    from basalt.verify.latency import DEFAULT_MODEL, LatencyModel
    from basalt.verify.observed import ObservedStalls

    tc = find_toolchain()
    latencies = ROOT / "data" / "latency" / "rtx-5070-ti.json"
    observed_path = ROOT / "data" / "latency" / "observed-stalls-sm120a.json"
    model = LatencyModel.assumed().overlay(latencies) if latencies.is_file() else DEFAULT_MODEL
    observed = ObservedStalls.read(observed_path) if observed_path.is_file() else None

    snippets = generate_scalar() + generate_shapes()
    if limit:
        snippets = snippets[:limit]
    work.mkdir(parents=True, exist_ok=True)

    def one(numbered):
        index, snippet = numbered
        src = work / f"{index:04d}.ptx"
        src.write_text(snippet.ptx)
        good = work / f"{index:04d}.good.cubin"
        built = tc.run(
            [str(tc.ptxas), f"-arch={ARCH}", "-O3", "-o", str(good), str(src)],
            check=False,
            timeout=60.0,
        )
        if built.returncode != 0:
            return None
        try:
            program = disassemble_program(tc, good)
        except Exception:
            return None
        target = _shortenable(program, model, observed)
        if target is None:
            return None
        producer, stall = target

        cubin = Cubin.load(good)
        cubin.patch_control(producer, "stall", 1)
        bad = work / f"{index:04d}.bad.cubin"
        cubin.save(bad)
        try:
            report = verify_program(disassemble_program(tc, bad), model, observed=observed)
        except Exception:
            return None
        hazard = any(h.severity is Severity.ERROR for h in report.hazards)
        entry = next(
            line.split("(")[0].split()[-1] for line in snippet.ptx.splitlines() if ".entry" in line
        )
        return {
            "i": index,
            "name": f"{snippet.name} #{producer} {stall}->1",
            "entry": entry,
            "verdict": "hazard" if hazard else "clean",
        }

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        built = [r for r in pool.map(one, enumerate(snippets)) if r]
    (work / "manifest.json").write_text(json.dumps(built))
    return built


def _worker(work: Path, start: int) -> int:
    from basalt.gpu.driver import Device

    manifest = json.loads((work / "manifest.json").read_text())
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
                for pattern in PATTERNS:
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
            index = entry["i"]
            try:
                good = run(work / f"{index:04d}.good.cubin", entry["entry"])
            except Exception:
                print(f"{index}\tunrunnable", flush=True)
                sys.stdout.flush()
                os._exit(1)
            if len(set(good)) != 1:
                print(f"{index}\tunstable", flush=True)
                continue
            try:
                bad = run(work / f"{index:04d}.bad.cubin", entry["entry"])
            except Exception:
                print(f"{index}\tfaulted", flush=True)
                sys.stdout.flush()
                os._exit(1)
            if len(set(bad)) != 1:
                print(f"{index}\tunstable", flush=True)
                continue
            print(f"{index}\t{'same' if bad[0] == good[0] else 'different'}", flush=True)
    return 0


def _drive(work: Path, manifest: list[dict]) -> list[Case]:
    by_index = {entry["i"]: entry for entry in manifest}
    behaviour: dict[int, str] = {}
    position = 0
    restarts = 0
    while position < len(manifest) and restarts < len(manifest) + 8:
        restarts += 1
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            str(position),
            "--work",
            str(work),
        ]
        # A deliberately broken kernel can hang the card rather than fault, and
        # without a bound the worker outlives the run that started it and goes
        # on competing with everything measured afterwards.
        try:
            output = subprocess.run(
                command, capture_output=True, text=True, cwd=str(ROOT), timeout=WORKER_TIMEOUT
            ).stdout
        except subprocess.TimeoutExpired as expired:
            raw = expired.stdout or ""
            output = raw.decode(errors="replace") if isinstance(raw, bytes) else raw
        lines = [line for line in output.splitlines() if "\t" in line]
        for line in lines:
            index, outcome = line.split("\t")[:2]
            behaviour[int(index)] = outcome
        reached = max((int(line.split("\t")[0]) for line in lines), default=None)
        if reached is None:
            behaviour[manifest[position]["i"]] = "unrunnable"
            position += 1
        else:
            position = next(k for k, e in enumerate(manifest) if e["i"] == reached) + 1

    return [
        Case(by_index[i]["name"], by_index[i]["verdict"], outcome)
        for i, outcome in sorted(behaviour.items())
    ]


if __name__ == "__main__":
    sys.exit(main())
