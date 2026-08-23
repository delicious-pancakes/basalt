#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Does `ptxas` schedule differently for different parts in the sm_120 family?

The largest caveat on everything basalt measures is that it was measured on one
card. Buying more cards is the obvious answer and a bad one, because the question
underneath is not "what does this part do" but "is the required schedule a
property of the part or of the architecture", and that one can be asked without
any card at all.

`ptxas` targets six members of this family: `sm_120`, `sm_120a`, `sm_120f`, and
the same three for `sm_121`, which is a different chip rather than a different
board. If the scheduling requirements varied across them, the compiler would have
to emit different control words, because it is the compiler that carries them:
the hardware checks nothing. So compile the whole corpus for every target and
compare the words.

    python scripts/across_the_family.py
    python scripts/across_the_family.py --opt 1 2 3

Needs a toolchain and no GPU. A difference here would be the more interesting
result and is worth stating either way, which is why this reports rather than
asserts.
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import _repo

_repo.use_repo_source()

# `a` is architecture-specific and `f` is family-specific; the bare name is the
# portable subset, and it declines the tensor forms the other two accept
FAMILY = ("sm_120a", "sm_120", "sm_120f", "sm_121a", "sm_121", "sm_121f")
REFERENCE = "sm_120a"


def _build(task):
    from basalt.disasm import disassemble_program
    from basalt.toolchain import find_toolchain

    snippet, arch, opt = task
    tc = find_toolchain()
    with TemporaryDirectory(prefix="basalt-family-") as tmp:
        source = Path(tmp) / "k.ptx"
        # the corpus names one target in its own text, and ptxas refuses a
        # `-arch` that disagrees with it
        source.write_text(snippet.ptx.replace(f".target {REFERENCE}", f".target {arch}"))
        cubin = Path(tmp) / "k.cubin"
        built = tc.run(
            [str(tc.ptxas), f"-arch={arch}", f"-O{opt}", "-o", str(cubin), str(source)],
            check=False,
            timeout=60.0,
        )
        if built.returncode != 0:
            return (snippet.name, arch, opt, None)
        program = disassemble_program(tc, cubin)
    return (
        snippet.name,
        arch,
        opt,
        tuple((i.text, i.word.value if i.word else None) for i in program.instructions),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--opt", type=int, nargs="*", default=[1, 3], help="optimisation levels")
    parser.add_argument("--arch", nargs="*", default=list(FAMILY), help="targets to compare")
    args = parser.parse_args()

    from basalt.harvest.corpus import generate as generate_scalar
    from basalt.harvest.corpus_shapes import generate_shapes
    from basalt.harvest.corpus_tensor import generate_tensor

    snippets = generate_scalar() + generate_tensor() + generate_shapes()
    tasks = [(s, a, o) for s in snippets for a in args.arch for o in args.opt]
    print(f"{_repo.provenance()}\n")
    print(
        f"compiling {len(snippets)} kernels for {len(args.arch)} targets at {len(args.opt)} levels"
    )

    with ThreadPoolExecutor(max_workers=(os.cpu_count() or 4) * 2) as pool:
        rows = list(pool.map(_build, tasks))
    got = {(name, arch, opt): words for name, arch, opt, words in rows}

    built = collections.Counter(a for _, a, _, w in rows if w is not None)
    print(f"\nkernels each target builds, of {len(snippets)} at {len(args.opt)} levels:")
    for arch in args.arch:
        print(f"  {arch:9} {built[arch]}")

    print(f"\nagainst {REFERENCE}, over every kernel both targets build:")
    disagreements = 0
    for arch in args.arch:
        if arch == REFERENCE:
            continue
        same = other_code = other_control = 0
        examples: list[str] = []
        for snippet in snippets:
            for opt in args.opt:
                base = got.get((snippet.name, REFERENCE, opt))
                mine = got.get((snippet.name, arch, opt))
                if base is None or mine is None:
                    continue
                if base == mine:
                    same += 1
                elif [t for t, _ in base] != [t for t, _ in mine]:
                    other_code += 1
                    if len(examples) < 3:
                        examples.append(f"{snippet.name} -O{opt}: different instructions")
                else:
                    other_control += 1
                    if len(examples) < 3:
                        pairs = enumerate(zip(base, mine, strict=True))
                        at = next(i for i, (x, y) in pairs if x != y)
                        examples.append(f"{snippet.name} -O{opt} #{at}: {base[at][0][:40]}")
        print(
            f"  {arch:9} identical {same:5}   different code {other_code:4}   "
            f"same code, different control {other_control:4}"
        )
        for line in examples:
            print(f"      {line}")
        disagreements += other_control

    if disagreements:
        print(
            f"\n{disagreements} kernels compile to the same instructions and a different "
            f"schedule. that is a per-part scheduling requirement, and it is a finding."
        )
        return 1
    print(
        "\nevery target that builds a kernel builds it to the same bytes, control words "
        "included:\nthe schedule is a property of the architecture rather than of the part"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
