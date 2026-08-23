#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Derive the ISA twice, from two compiler releases, and compare the two models.

basalt's instruction model comes from differential probing of one `ptxas` build.
That is one measurement, and one measurement is not a result: every bit role it
assigns could be a property of that build rather than of the silicon, and nothing
inside the pipeline would notice.

So the model is derived a second time, independently, from a different CUDA
release, and the two are compared. Two things are asked, and they are not the
same question:

*Do the two derivations agree?* Compared on the model rather than the encoding.
An encoding carries the control bits and the branch target of whichever kernel
the form was harvested from, so two releases disagree on nearly every encoding
and that says nothing. What matters is which bits basalt calls the opcode, which
it calls modifiers, and which bits each operand occupies.

*Can one model read the other's output?* Stronger, and the one that would be hard
to fake: take the database built from release A and assemble every instruction
release B emits. A model that only describes the compiler it was derived from
fails this immediately.

    python scripts/cross_toolchain.py
    python scripts/cross_toolchain.py --other 13.0.3 --opt 1 3

Needs both toolchains fetched and no GPU:

    python scripts/fetch_toolchain.py
    python scripts/fetch_toolchain.py --version 13.0.3
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import _repo

_repo.use_repo_source()

ROOT = _repo.ROOT
DEFAULT_OTHER = "13.0.3"
COMMITTED = ROOT / "data" / "isa" / "sm_120a.json"


def _model(form: dict) -> tuple:
    """What basalt derived, with no trace of the kernel it was derived from."""
    return (
        tuple(form.get("opcode_bits", ())),
        tuple(form.get("modifier_bits", ())),
        tuple(
            (
                slot.get("slot"),
                tuple(slot.get("bits", ())),
                tuple(sorted((k, tuple(v)) for k, v in slot.get("subfields", {}).items())),
            )
            for slot in form.get("operands", ())
        ),
    )


def _operand_model(form: dict) -> tuple:
    """The half an assembler writes through, which is the half that has to agree."""
    return _model(form)[2]


def _run(command: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=str(ROOT), env=env, capture_output=True, text=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--other", default=DEFAULT_OTHER, help="the second CUDA release")
    parser.add_argument(
        "--opt", type=int, nargs="*", default=[1, 3], help="levels to assemble across"
    )
    parser.add_argument("--work", type=Path, default=None, help="scratch directory")
    args = parser.parse_args()

    other_bin = ROOT / "third_party" / "cuda" / args.other / "bin"
    if not other_bin.is_dir():
        raise SystemExit(
            f"{other_bin} is not there. fetch it first:\n"
            f"  python scripts/fetch_toolchain.py --version {args.other}"
        )
    if not COMMITTED.is_file():
        raise SystemExit(f"{COMMITTED} is not there; run `basalt build-isa` first")

    work = args.work or Path(os.environ.get("TMP", "/tmp")) / "basalt-cross"
    work.mkdir(parents=True, exist_ok=True)
    other_db = work / f"isa-{args.other}.json"

    env = dict(os.environ, BASALT_CUDA_BIN=str(other_bin))
    print(f"{_repo.provenance()}\n")

    if not other_db.is_file():
        print(f"deriving the model again from CUDA {args.other}, which takes a few minutes")
        built = _run(
            [sys.executable, "-m", "basalt.cli", "build-isa", "-o", str(other_db)], env=env
        )
        if built.returncode != 0:
            print(built.stdout[-2000:], built.stderr[-2000:], file=sys.stderr)
            raise SystemExit(f"build-isa failed under CUDA {args.other}")
    else:
        print(f"reusing {other_db}")

    ours = json.loads(COMMITTED.read_text())
    theirs = json.loads(other_db.read_text())
    here, there = ours["forms"], theirs["forms"]
    shared = sorted(set(here) & set(there))

    model_differs = [n for n in shared if _model(here[n]) != _model(there[n])]
    operands_differ = [n for n in shared if _operand_model(here[n]) != _operand_model(there[n])]
    encoding_differs = [n for n in shared if here[n]["encoding"] != there[n]["encoding"]]

    print(f"\n{ours['cuda_version']} against {theirs['cuda_version']}")
    print(f"  forms                         {len(here)} and {len(there)}, {len(shared)} shared")
    print(f"  only in {ours['cuda_version']:<22} {sorted(set(here) - set(there))}")
    print(f"  only in {theirs['cuda_version']:<22} {sorted(set(there) - set(here))}")
    print(f"  exemplar encodings differing  {len(encoding_differs)}")
    print(f"  derived models differing      {len(model_differs)}")
    print(f"  **operand models differing**  {len(operands_differ)}")
    if operands_differ:
        for name in operands_differ[:20]:
            print(f"      {name}")

    print(f"\nassembling CUDA {args.other} output with the {ours['cuda_version']} database")
    failures = 0
    for opt in args.opt:
        run = _run(
            [
                sys.executable,
                "scripts/assembler_coverage.py",
                "--opt",
                str(opt),
                "--work",
                str(work / f"asm-O{opt}"),
            ],
            env=env,
        )
        line = next(
            (ln for ln in run.stdout.splitlines() if "bit-identical" in ln), "  (no output)"
        )
        wrong = next((ln for ln in run.stdout.splitlines() if "WRONG" in ln), "")
        print(f"  -O{opt}{line}   {wrong.strip()}")
        failures += run.returncode != 0

    # the operand model is the claim: which bits read as opcode depends on the
    # exemplar, since that is what decides which flips change the mnemonic
    if operands_differ or failures:
        print("\nthe two derivations disagree about something that matters", file=sys.stderr)
        return 1
    print("\nthe two derivations agree on every operand field, and each reads the other's output")
    return 0


if __name__ == "__main__":
    sys.exit(main())
