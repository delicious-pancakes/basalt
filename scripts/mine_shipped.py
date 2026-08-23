#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Mine the stall requirement from shipped machine code instead of from a corpus.

basalt's requirement table is the tightest gap `ptxas` was ever seen to leave
between a dependent pair. Finding 21 established what that number is worth: it
is an upper bound on the requirement, and a narrow corpus reads high because a
kernel with two instructions of body gives the compiler nothing to fill a gap
with. Widening the corpus moved `FADD` from 5 to 4 and `ULEA` from 9 to 7, and
in both cases the wider number was the one fault injection had measured.

There is a much wider corpus available, and it is the one stage 10 audits. Every
sm_120 kernel in cuBLAS, cuSOLVER, cuSPARSE and NPP is real code under real
register pressure, scheduled by the same compiler, and there are millions of
dependent pairs in it rather than tens of thousands.

Mined from the same thing every other number here is mined from: observable
output. No NVIDIA code, table or header enters the repository, only the gaps
their compiler left, exactly as `mine-stalls` records the gaps it leaves on
basalt's own kernels.

    python scripts/mine_shipped.py --cubins <dir> --out <table.json>
    python scripts/mine_shipped.py --cubins <dir> --exclude nvjpeg64_13,curand64_10

`--exclude` is the point rather than a convenience. A table mined from the same
code it is then checked against cannot fail, which is the flaw stage 10 found in
the corpus-mined one, so the libraries the audit reports on are held out here.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import _repo

_repo.use_repo_source()

ROOT = _repo.ROOT
DEFAULT_BASE = ROOT / "data" / "latency" / "observed-stalls-sm120a.json"

_STATE: dict = {}


def _setup() -> None:
    from basalt.toolchain import find_toolchain

    _STATE["tc"] = find_toolchain()


def _mine(paths: list[Path]):
    from basalt.disasm import disassemble_kernels
    from basalt.verify.observed import ObservedStalls, mine_program

    if not _STATE:
        _setup()
    tc = _STATE["tc"]
    out = ObservedStalls(cuda_version=tc.version, arch="sm_120")
    for path in paths:
        try:
            kernels = disassemble_kernels(tc, path)
        # one unreadable cubin is not a reason to abandon the other 2,472
        except Exception:
            continue
        for program in kernels.values():
            mine_program(program, out)
    return out


def _chunks(items: list[Path], size: int) -> list[list[Path]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cubins", type=Path, required=True, help="directory of sm_120 cubins")
    parser.add_argument("--out", type=Path, required=True, help="where to write the merged table")
    parser.add_argument(
        "--base",
        type=Path,
        default=DEFAULT_BASE,
        help="table to fold into, default the corpus-mined one",
    )
    parser.add_argument("--no-base", action="store_true", help="mine from the shipped code alone")
    parser.add_argument(
        "--exclude", default="", help="comma separated library directories to hold out"
    )
    parser.add_argument(
        "--per-library",
        type=int,
        default=0,
        help="cubins to take from each library, default all",
    )
    parser.add_argument("--workers", type=int, default=0, help="processes, default one per core")
    args = parser.parse_args()

    from basalt.verify.observed import ObservedStalls

    held = {name for name in args.exclude.split(",") if name}
    targets = [p for p in sorted(args.cubins.rglob("*.cubin")) if p.parent.name not in held]
    if args.per_library:
        # the pairings saturate long before the cubins do, and one library of
        # tensor-core GEMM variants is 1,601 near-copies of the same schedule
        seen: dict[str, int] = {}
        capped = []
        for path in targets:
            n = seen.get(path.parent.name, 0)
            if n >= args.per_library:
                continue
            seen[path.parent.name] = n + 1
            capped.append(path)
        targets = capped
    if not targets:
        raise SystemExit("no cubins to mine")

    print(f"{_repo.provenance()}\n")
    libraries = sorted({p.parent.name for p in targets})
    print(f"mining {len(targets)} cubins from {len(libraries)} libraries: {', '.join(libraries)}")
    if held:
        print(f"holding out {', '.join(sorted(held))}")

    total = ObservedStalls(cuda_version="", arch="sm_120")
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers or None, initializer=_setup) as pool:
        for done, part in enumerate(pool.map(_mine, _chunks(targets, 8)), start=1):
            if not total.cuda_version:
                total.cuda_version = part.cuda_version
            total.absorb(part)
            if done % 25 == 0:
                print(f"  {done * 8}/{len(targets)} cubins, {total.kernels} kernels", flush=True)
    elapsed = time.perf_counter() - started

    print(f"\nshipped only: {total.summary()}")
    print(f"mined in {elapsed / 60:.1f} min")

    if not args.no_base:
        base = ObservedStalls.read(args.base)
        print(f"corpus table: {base.summary()}")
        tightened = sum(
            1
            for key, ev in total.by_pair.items()
            if key in base.by_pair and ev.minimum < base.by_pair[key].minimum
        )
        added = sum(1 for key in total.by_pair if key not in base.by_pair)
        print(f"\n{tightened} pairings tightened, {added} pairings the corpus never emitted")
        base.absorb(total)
        total = base

    total.write(args.out)
    print(f"\nmerged: {total.summary()}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
