# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Reassemble every corpus kernel and compare the bytes against the vendor's.

The assembler's headline number is "N of M instructions bit-identical, none
wrong", and a headline number nobody can reproduce is an assertion. This is the
command that produces it.

Each kernel is compiled with `ptxas`, disassembled, and handed back to basalt's
assembler as a whole program. Whole program rather than instruction by
instruction, because a branch encodes the distance to its destination: the same
text is a different word in every kernel it appears in, so a branch can only be
assembled in the company of the thing it jumps to.

Three outcomes, and only one of them is allowed to be non-zero without comment:

  exact     basalt produced the vendor's word, all 128 bits
  refused   basalt declined, naming the field it could not encode
  WRONG     basalt produced a different word and did not say so

Refusing is a limit. Being wrong is a bug, and on this architecture a quiet
one, which is why this exits non-zero the moment the wrong count leaves zero.
No GPU is needed. `ptxas` and `nvdisasm` are enough.

    python scripts/assembler_coverage.py
    python scripts/assembler_coverage.py --opt 1 --show-refusals
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import _repo

_repo.use_repo_source()

ROOT = _repo.ROOT
ARCH = "sm_120a"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--opt", type=int, default=3, choices=(0, 1, 2, 3), help="ptxas -O level")
    parser.add_argument("--work", type=Path, default=None, help="scratch directory for cubins")
    parser.add_argument("--show-refusals", action="store_true", help="list the reasons, by count")
    parser.add_argument("--only", default=None, help="comma separated kernel names")
    parser.add_argument("--db", type=Path, default=None, help="ISA database to assemble against")
    args = parser.parse_args()

    from basalt.asm.assemble import assemble_program
    from basalt.disasm import disassemble_program
    from basalt.harvest.corpus import generate as generate_scalar
    from basalt.harvest.corpus_shapes import generate_shapes
    from basalt.harvest.corpus_tensor import generate_tensor
    from basalt.isa.database import IsaDatabase
    from basalt.toolchain import find_toolchain

    database_path = args.db or ROOT / "data" / "isa" / f"{ARCH}.json"
    if not database_path.is_file():
        print(f"error: {database_path} does not exist; run `basalt build-isa`", file=sys.stderr)
        return 2

    tc = find_toolchain()
    database = IsaDatabase.read(database_path)

    work = args.work or Path(__file__).resolve().parent.parent / ".work" / "assembler-coverage"
    work.mkdir(parents=True, exist_ok=True)

    snippets = generate_scalar() + generate_tensor() + generate_shapes()
    if args.only:
        wanted = {name.strip() for name in args.only.split(",")}
        snippets = [s for s in snippets if s.name in wanted]

    # every kernel is independent, and the work is two subprocess calls apiece,
    # so this is one core's worth of waiting per kernel unless it is spread out
    def one(numbered: tuple[int, object]):
        index, snippet = numbered
        src = work / f"{index:04d}.ptx"
        src.write_text(snippet.ptx)
        cubin = work / f"{index:04d}.cubin"
        built = tc.run(
            [str(tc.ptxas), f"-arch={ARCH}", f"-O{args.opt}", "-o", str(cubin), str(src)],
            check=False,
            timeout=60.0,
        )
        if built.returncode != 0:
            return None

        program = disassemble_program(tc, cubin)
        result = assemble_program(program, database)

        counted = Counter()
        mismatched = 0
        for instruction, produced in zip(program.instructions, result.words, strict=True):
            if instruction.word is None:
                continue
            if produced is None:
                counted["refused"] += 1
            elif produced.value == instruction.word.value:
                counted["exact"] += 1
            else:
                counted["wrong"] += 1
                mismatched += 1
        # the field name is the useful part; the operand text is not
        why = Counter(reason.split(":")[0][:70] for _, _, reason in result.refused)
        return counted, why, (snippet.name, mismatched) if mismatched else None

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        outcomes = [r for r in pool.map(one, enumerate(snippets)) if r is not None]

    kernels = len(outcomes)
    totals: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    wrong_kernels: list[tuple[str, int]] = []
    for counted, why, bad in outcomes:
        totals.update(counted)
        reasons.update(why)
        if bad:
            wrong_kernels.append(bad)
    exact, refused, wrong = totals["exact"], totals["refused"], totals["wrong"]

    considered = exact + refused + wrong
    print(f"\n{_repo.provenance()}")
    print(f"\n{kernels} corpus kernels assembled at ptxas -O{args.opt}\n")
    print(f"  exact      {exact:6}")
    print(f"  refused    {refused:6}")
    print(f"  WRONG      {wrong:6}")
    if considered:
        print(f"\n  {exact} of {considered} instructions bit-identical to the vendor")

    if args.show_refusals and reasons:
        print("\nwhy the rest were refused:")
        for reason, count in reasons.most_common(15):
            print(f"  {count:5}  {reason}")

    if wrong_kernels:
        print("\nWRONG, which is never acceptable:")
        for name, count in wrong_kernels:
            print(f"  {name}: {count}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
