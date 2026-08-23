#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Run every control basalt has, in order, and print what each one answered.

The README quotes about a dozen numbers. Each has a command behind it, and until
now reproducing the set meant reading the README and running them by hand in the
right order, which is the sort of thing that works once for the person who wrote
it. This is the one command.

    python scripts/verify_all.py              # everything the machine can run
    python scripts/verify_all.py --no-gpu     # skip the four that need silicon
    python scripts/verify_all.py --quick      # skip the rebuilds as well

Nothing here is new work. It shells out to the same scripts CI runs, in the same
order, and reports the exit code and the line each one calls its answer. A step
that needs hardware this machine does not have is reported as skipped rather than
quietly passing, because a control that did not run is not a control that passed.
The audit needs the shipped libraries fetched first, and reports what it could
not find rather than passing on an empty directory.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass

import _repo

_repo.use_repo_source()

ROOT = _repo.ROOT


@dataclass(frozen=True)
class Step:
    """One control, and how to read its answer out of the noise."""

    name: str
    command: list[str]
    # first line matching any of these is the answer worth printing
    answer: tuple[str, ...] = ()
    needs_gpu: bool = False
    slow: bool = False
    # a path that has to exist first, so an unfetched input reads as skipped
    needs_path: str = ""


PY = sys.executable

# held out of the audit because the requirement was mined from them, and a table
# checked against the code it came from cannot fail (finding 32)
MINED_LIBRARIES = (
    "cublas64_13",
    "cublasLt64_13",
    "cusolver64_12",
    "cusparse64_12",
    "nppial64_13",
    "nppicc64_13",
    "nppidei64_13",
    "nppif64_13",
    "nppig64_13",
    "nppim64_13",
    "nppist64_13",
    "nppitc64_13",
    "npps64_13",
)

STEPS: tuple[Step, ...] = (
    Step("lint", [PY, "-m", "ruff", "check", "."]),
    Step("format", [PY, "-m", "ruff", "format", "--check", "."], ("files already formatted",)),
    Step("types, linux", [PY, "-m", "mypy", "--platform", "linux"], ("Success",)),
    Step("types, win32", [PY, "-m", "mypy", "--platform", "win32"], ("Success",)),
    Step("both oracles", [PY, "-m", "basalt.cli", "doctor"], ("ptxas", "ok")),
    Step(
        "ISA rebuild has not drifted",
        [PY, "scripts/rebuild_and_compare.py"],
        ("no drift", "forms"),
        slow=True,
    ),
    Step(
        "measured fields still behave",
        [PY, "-m", "basalt.cli", "validate-isa"],
        ("controllable",),
        slow=True,
    ),
    *(
        Step(
            f"assembler at -O{opt}",
            [PY, "scripts/assembler_coverage.py", "--opt", str(opt)],
            ("bit-identical",),
            slow=True,
        )
        for opt in (0, 1, 2, 3)
    ),
    Step("assembler under mutation", [PY, "scripts/fuzz_assembler.py"], ("decoded back", "wrong")),
    Step(
        "one schedule per family target",
        [PY, "scripts/across_the_family.py"],
        ("property of the architecture", "different schedule"),
        slow=True,
    ),
    *(
        Step(
            f"round trip on the card at -O{opt}",
            [PY, "scripts/roundtrip_corpus.py", "--opt", str(opt)],
            ("comparable kernels match",),
            needs_gpu=True,
            slow=True,
        )
        for opt in (1, 2, 3)
    ),
    Step(
        "the checker against the silicon",
        [PY, "scripts/agreement_sweep.py"],
        ("MISSED", "one dependency shortened"),
        needs_gpu=True,
        slow=True,
    ),
    Step(
        "shipped libraries audit",
        [
            PY,
            "scripts/audit_shipped.py",
            "--cubins",
            "third_party/cuda/13.3.1/cubins",
            "--exclude",
            ",".join(MINED_LIBRARIES),
        ],
        ("errors:",),
        slow=True,
        needs_path="third_party/cuda/13.3.1/cubins",
    ),
    Step("the suite", [PY, "-m", "pytest", "-q"], ("passed", "failed", "error"), slow=True),
)


def _answer(step: Step, output: str) -> str:
    for line in output.splitlines():
        if any(needle in line for needle in step.answer):
            return line.strip()
    tail = [ln.strip() for ln in output.splitlines() if ln.strip()]
    return tail[-1][:96] if tail else ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--no-gpu", action="store_true", help="skip the steps needing silicon")
    parser.add_argument("--quick", action="store_true", help="skip the slow rebuilds too")
    args = parser.parse_args()

    from basalt.gpu.driver import cuda_available

    have_gpu = not args.no_gpu and cuda_available()
    print(f"{_repo.provenance()}\n")
    if not have_gpu:
        print("no GPU in play, so the steps needing one are skipped rather than passed\n")

    width = max(len(s.name) for s in STEPS)
    failures: list[str] = []
    skipped = 0
    started = time.perf_counter()

    for step in STEPS:
        if step.needs_gpu and not have_gpu:
            print(f"  {step.name:<{width}}  skipped, needs a GPU")
            skipped += 1
            continue
        if step.needs_path and not (ROOT / step.needs_path).exists():
            print(f"  {step.name:<{width}}  skipped, run fetch_toolchain.py --libs first")
            skipped += 1
            continue
        if step.slow and args.quick:
            print(f"  {step.name:<{width}}  skipped, --quick")
            skipped += 1
            continue
        began = time.perf_counter()
        run = subprocess.run(step.command, cwd=str(ROOT), capture_output=True, text=True)
        took = time.perf_counter() - began
        mark = "ok  " if run.returncode == 0 else "FAIL"
        print(f"  {step.name:<{width}}  {mark} {took:6.1f}s  {_answer(step, run.stdout)}")
        if run.returncode != 0:
            failures.append(step.name)
            for line in (run.stdout + run.stderr).splitlines()[-6:]:
                print(f"       {line[:110]}")

    elapsed = time.perf_counter() - started
    print(f"\n{len(STEPS) - skipped - len(failures)} controls passed in {elapsed / 60:.1f} min")
    if skipped:
        print(f"{skipped} skipped")
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
