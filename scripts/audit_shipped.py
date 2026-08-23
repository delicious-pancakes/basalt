#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Point the checker at machine code basalt did not produce.

Every other control here runs on kernels the corpus compiled. That proves the
checker agrees with `ptxas` on `ptxas` output, which is necessary and is not the
question anyone holding a cubin actually has. This runs it against shipped
production libraries: `cublasLt`, `cusolver`, `cusparse`, `npp` and the rest,
whose sm_120 kernels were scheduled by a compiler basalt has never seen the
output of, at optimisation settings and instruction mixes the corpus does not
reach.

The result is a measurement whichever way it comes out. Silence over millions of
shipped instructions is a much stronger statement about the checker than silence
over the corpus, because these kernels were not written to exercise it. A hazard
is either a real one in shipped code or a hole in the model, and both are worth
knowing.

    python scripts/audit_shipped.py --libs <dir of dlls or .so files>
    python scripts/audit_shipped.py --cubins <dir of extracted cubins>
    python scripts/audit_shipped.py --cubins <dir> --report audit.json

Needs no GPU. The libraries are not vendored and never enter the repository;
`python scripts/fetch_toolchain.py --libs` fetches them from the same pinned
redistributable the compiler comes from and records which build each one was.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import _repo

_repo.use_repo_source()

ROOT = _repo.ROOT
DEFAULT_LATENCIES = ROOT / "data" / "latency" / "rtx-5070-ti.json"
DEFAULT_OBSERVED = ROOT / "data" / "latency" / "observed-stalls-sm120a.json"


@dataclass
class Tally:
    """What one worker saw, in a shape that adds up across workers."""

    cubins: int = 0
    kernels: int = 0
    instructions: int = 0
    pairs: int = 0
    errors: int = 0
    warnings: int = 0
    unknown: Counter = field(default_factory=Counter)
    # a pattern is (kind, producing opcode, consuming opcode): millions of
    # findings from one modelling hole are one finding, and the count says so
    patterns: Counter = field(default_factory=Counter)
    findings: list[dict] = field(default_factory=list)

    def merge(self, other: Tally) -> None:
        self.cubins += other.cubins
        self.kernels += other.kernels
        self.instructions += other.instructions
        self.pairs += other.pairs
        self.errors += other.errors
        self.warnings += other.warnings
        self.unknown.update(other.unknown)
        self.patterns.update(other.patterns)
        self.findings.extend(other.findings)


def _opcode(text: str) -> str:
    """The bare opcode, skipping the guard predicate printed before it."""
    parts = text.split()
    if parts and parts[0].startswith("@"):
        parts = parts[1:]
    return parts[0].split(".")[0] if parts else "?"


_STATE: dict = {}


def _setup() -> None:
    """One toolchain and one model per worker process, not per cubin."""
    from basalt.toolchain import find_toolchain
    from basalt.verify.latency import LatencyModel
    from basalt.verify.observed import ObservedStalls

    model = LatencyModel.assumed()
    if DEFAULT_LATENCIES.is_file():
        model = model.overlay(DEFAULT_LATENCIES)
    _STATE["tc"] = find_toolchain()
    _STATE["model"] = model
    _STATE["observed"] = (
        ObservedStalls.read(DEFAULT_OBSERVED) if DEFAULT_OBSERVED.is_file() else None
    )


def _audit(path: Path) -> Tally:
    from basalt.disasm import disassemble_kernels
    from basalt.verify.hazards import Severity, verify_program

    if not _STATE:
        _setup()

    tally = Tally(cubins=1)
    for name, program in disassemble_kernels(_STATE["tc"], path).items():
        report = verify_program(program, _STATE["model"], observed=_STATE["observed"])
        tally.kernels += 1
        tally.instructions += report.instructions
        tally.pairs += report.checked_pairs
        tally.unknown.update(report.unknown_opcodes)
        for hazard in report.hazards:
            if hazard.severity is Severity.ERROR:
                tally.errors += 1
            elif hazard.severity is Severity.WARNING:
                tally.warnings += 1
            else:
                continue
            producer = _opcode(hazard.def_text)
            consumer = _opcode(hazard.use_text)
            key = (str(hazard.kind), producer, consumer)
            tally.patterns[key] += 1
            # one example per pattern, because the second one teaches nothing
            if tally.patterns[key] > 1:
                continue
            tally.findings.append(
                {
                    "library": path.parent.name,
                    "cubin": path.name,
                    "kernel": name,
                    "severity": str(hazard.severity),
                    "kind": str(hazard.kind),
                    "register": hazard.register,
                    "required": hazard.required,
                    "actual": hazard.actual,
                    "def": hazard.def_text,
                    "use": hazard.use_text,
                    "detail": hazard.detail,
                }
            )
    return tally


def _extract(libs: Path, dest: Path) -> list[Path]:
    """Pull every sm_120 ELF out of each host library, one directory per library."""
    from basalt.toolchain import find_toolchain

    cuobjdump = find_toolchain().bin_dir / "cuobjdump.exe"
    if not cuobjdump.is_file():
        cuobjdump = cuobjdump.with_suffix("")
    if not cuobjdump.is_file():
        raise SystemExit("cuobjdump not found beside ptxas; run scripts/fetch_toolchain.py")

    found: list[Path] = []
    for lib in sorted([*libs.rglob("*.dll"), *libs.rglob("*.so*")]):
        out = dest / lib.stem
        out.mkdir(parents=True, exist_ok=True)
        # filtered: cuBLASLt carries every architecture back to Volta, and
        # unpacking those to delete them costs tens of GB
        subprocess.run(
            [str(cuobjdump), "-xelf", "sm_120", str(lib.resolve())],
            cwd=out,
            capture_output=True,
            check=False,
        )
        cubins = sorted(out.glob("*sm_120*.cubin"))
        print(f"  {lib.name}: {len(cubins)} sm_120 cubins", flush=True)
        found.extend(cubins)
    return found


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--libs", type=Path, help="directory of host libraries to extract from")
    parser.add_argument("--cubins", type=Path, help="directory of already extracted cubins")
    parser.add_argument("--report", type=Path, help="write the full result as JSON")
    parser.add_argument("--limit", type=int, default=0, help="stop after this many cubins")
    parser.add_argument("--workers", type=int, default=0, help="processes, default one per core")
    args = parser.parse_args()

    if args.cubins:
        targets = sorted(args.cubins.rglob("*.cubin"))
    elif args.libs:
        targets = _extract(args.libs, args.libs.parent / "cubins")
    else:
        raise SystemExit("pass --libs or --cubins")
    if not targets:
        raise SystemExit("no sm_120 cubins found")
    if args.limit:
        targets = targets[: args.limit]

    print(f"{_repo.provenance()}\n")
    print(f"auditing {len(targets)} cubins from {len({t.parent.name for t in targets})} libraries")

    total = Tally()
    started = time.perf_counter()
    workers = args.workers or None
    with ProcessPoolExecutor(max_workers=workers, initializer=_setup) as pool:
        for done, tally in enumerate(pool.map(_audit, targets, chunksize=4), start=1):
            total.merge(tally)
            if done % 100 == 0:
                print(
                    f"  {done}/{len(targets)} cubins, {total.kernels} kernels, "
                    f"{total.errors} errors",
                    flush=True,
                )
    elapsed = time.perf_counter() - started

    print(
        f"\n{total.kernels} kernels, {total.instructions} instructions, "
        f"{total.pairs} dependencies checked in {elapsed / 60:.1f} min"
    )
    print(f"errors: {total.errors}")
    print(f"warnings: {total.warnings}")
    if total.unknown:
        top = ", ".join(f"{op}" for op, _ in total.unknown.most_common(12))
        print(f"opcodes not in the latency model: {len(total.unknown)} ({top})")

    print(f"\ndistinct (hazard, producer, consumer) patterns: {len(total.patterns)}")
    for (kind, producer, consumer), count in total.patterns.most_common(30):
        print(f"  {count:8}  {kind:24} {producer} -> {consumer}")

    for finding in total.findings[:20]:
        print(f"\n  {finding['severity']} {finding['kind']} {finding['register']}")
        print(f"    {finding['library']}/{finding['cubin']}  {finding['kernel'][:70]}")
        print(f"    def  {finding['def']}")
        print(f"    use  {finding['use']}")

    if args.report:
        args.report.write_text(
            json.dumps(
                {
                    "cubins": total.cubins,
                    "kernels": total.kernels,
                    "instructions": total.instructions,
                    "dependencies": total.pairs,
                    "errors": total.errors,
                    "warnings": total.warnings,
                    "unknown_opcodes": dict(total.unknown.most_common()),
                    "patterns": [
                        {"kind": k, "producer": p, "consumer": c, "count": n}
                        for (k, p, c), n in total.patterns.most_common()
                    ],
                    "findings": total.findings,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
