#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Recompute every corpus-derived figure the documentation quotes.

The findings are full of counts: how many read barriers `ptxas` emits, how often
it waits on a barrier it signals, how many anti-dependencies it leaves under
three cycles. Each was measured once and then written down, and a number written
down is a number that rots the next time the corpus grows. Several already had:
"334 read barriers" became 341, "354 branches" became 517, and one claim was not
stale but wrong, which is worse and is why this exists.

    python scripts/corpus_figures.py
    python scripts/corpus_figures.py --opt 3

Needs a toolchain and no GPU. Nothing here reads the documentation, so it cannot
tell you the docs are right; it tells you what the answer is today, in one place,
so checking them is a minute's work rather than an afternoon's.
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

ROOT = _repo.ROOT
ARCH = "sm_120a"
LEVELS = (1, 2, 3)

# stores have no register result, so no latency class, but their data register is
# still in flight after they issue
LATE_READING_STORES = frozenset({"STG", "STS", "STL", "ST", "RED"})
FP64 = frozenset({"DADD", "DMUL", "DFMA", "DSETP", "DMNMX", "DMMA"})


def _compile(tc, snippet, opt):
    from basalt.disasm import disassemble_program

    with TemporaryDirectory(prefix="basalt-figures-") as tmp:
        src = Path(tmp) / "k.ptx"
        src.write_text(snippet.ptx)
        cubin = Path(tmp) / "k.cubin"
        built = tc.run(
            [str(tc.ptxas), f"-arch={ARCH}", f"-O{opt}", "-o", str(cubin), str(src)],
            check=False,
            timeout=60.0,
        )
        if built.returncode != 0:
            return None
        return disassemble_program(tc, cubin)


def _late(mnemonic, model):
    from basalt.verify.latency import LatencyClass

    opcode = mnemonic.split(".")[0]
    return model.lookup(opcode).kind is LatencyClass.VARIABLE or opcode in LATE_READING_STORES


def _one(program, model, observed) -> collections.Counter:
    """Every figure this reports, for one compiled kernel."""
    import re

    from basalt.asm.assemble import branch_target
    from basalt.encoding import NO_BARRIER, effective_stall
    from basalt.sched.scheduler import SCOREBOARD_OPERAND, YIELD_STALL_RANGE
    from basalt.verify.cfg import build_cfg

    label = re.compile(r"`\(([^)]+)\)")
    counts: collections.Counter = collections.Counter()
    words = [i.word for i in program.instructions]

    for instruction in program.instructions:
        word = instruction.word
        if word is None:
            continue
        counts["instructions"] += 1
        if word.field("read_barrier") != NO_BARRIER:
            counts["read barriers"] += 1
            where = "a store" if _store(instruction) else "a variable-latency instruction"
            counts[f"read barriers on {where}"] += 1
        for field in ("write_barrier", "read_barrier"):
            barrier = word.field(field)
            if barrier != NO_BARRIER and (word.field("wait_mask") >> barrier) & 1:
                counts[f"waits on the {field} it signals"] += 1
        if word.field("stall") == 0:
            counts[f"stall of zero, yield {'set' if word.field('yield_') else 'clear'}"] += 1
        if word.field("stall") == 1:
            counts[f"stall of one, yield {'set' if word.field('yield_') else 'clear'}"] += 1
        # the two rules finding 26 compares, scored against the vendor's own bit
        low, high = YIELD_STALL_RANGE
        stall = word.field("stall")
        counts["yield: the fitted rule agrees"] += int(low <= stall < high) == word.field("yield_")
        counts["yield: the guess it replaced agrees"] += int(stall == 1) == word.field("yield_")
        if instruction.opcode in FP64:
            has = word.field("write_barrier") != NO_BARRIER
            counts["fp64 with a write barrier" if has else "fp64 without one"] += 1
        if SCOREBOARD_OPERAND.search(instruction.operands):
            counts["instructions naming a scoreboard in an operand"] += 1
        match = label.search(instruction.operands)
        if match is not None and (destination := program.labels.get(match.group(1))) is not None:
            correct = branch_target(word, instruction.offset) == destination * 16
            counts["branches decoding to their label" if correct else "branches wrong"] += 1

    for block in build_cfg(program).blocks:
        _read_barrier_shapes(program, words, block, model, counts)
        _short_anti_dependencies(program, words, block, observed, counts, effective_stall)
    return counts


def _store(instruction) -> bool:
    return instruction.opcode in LATE_READING_STORES


def _read_barrier_shapes(program, words, block, model, counts) -> None:
    """How the vendor covers a read whose register is overwritten later."""
    from basalt.encoding import NO_BARRIER, effective_stall
    from basalt.verify.operands import operand_access

    for reader in range(block.start, block.end):
        instruction = program.instructions[reader]
        if instruction.word is None or not _late(instruction.mnemonic, model):
            continue
        sources = operand_access(instruction.mnemonic, instruction.operands).real_uses
        if not sources:
            continue
        write = instruction.word.field("write_barrier")
        read = instruction.word.field("read_barrier")
        covered = False
        gap = 0
        overwrite = None
        for later in range(reader + 1, block.end):
            following = program.instructions[later]
            if following.word is None:
                continue
            gap += effective_stall(program.instructions[later - 1].word.field("stall"))
            if write != NO_BARRIER and (following.word.field("wait_mask") >> write) & 1:
                covered = True
            access = operand_access(following.mnemonic, following.operands)
            if access.real_defs & sources:
                overwrite = (later, gap)
                break
        if overwrite is None:
            # not in this block. a loop-carried overwrite is the case read
            # barriers exist for, so ask whether anything writes it at all
            elsewhere = any(
                other.word is not None
                and index != reader
                and operand_access(other.mnemonic, other.operands).real_defs & sources
                for index, other in enumerate(program.instructions)
            )
            where = "in another block" if elsewhere else "anywhere"
            kind = "barrier" if read != NO_BARRIER else "none"
            counts[f"a source is overwritten {where}, {kind}"] += 1
            continue
        if read != NO_BARRIER:
            counts["a source is overwritten later, barrier"] += 1
            continue
        counts["a source is overwritten later, none"] += 1
        if covered:
            counts["  covered by a wait on its own write barrier"] += 1
            continue
        mask = words[overwrite[0]].field("wait_mask")
        inorder = any(
            words[k] is not None
            and words[k].field("read_barrier") != NO_BARRIER
            and (mask >> words[k].field("read_barrier")) & 1
            for k in range(reader, overwrite[0])
        )
        how = "a later reader's barrier" if inorder else "distance alone"
        counts[f"  covered by {how}"] += 1
        if not inorder:
            counts["distance total"] += overwrite[1]
            counts["distance count"] += 1


def _short_anti_dependencies(program, words, block, observed, counts, effective_stall) -> None:
    """Reads the vendor overwrites sooner than basalt's model would allow."""
    from basalt.verify.observed import anti_dependency_cycles
    from basalt.verify.operands import operand_access

    last_read: dict = {}
    elapsed: dict = {}
    for index in range(block.start, block.end):
        instruction = program.instructions[index]
        if instruction.word is None:
            continue
        access = operand_access(instruction.mnemonic, instruction.operands)
        for register in access.real_defs:
            previous = last_read.get(register)
            if previous is None or previous[0] == index:
                continue
            reader = program.instructions[previous[0]]
            if reader.word is None or reader.word.field("read_barrier") != 0b111:
                continue
            needed = anti_dependency_cycles(previous[1], instruction.opcode, observed)
            if elapsed.get(register, 0) < needed:
                counts["anti-dependencies shorter than the model allows"] += 1
        stall = effective_stall(instruction.word.field("stall"))
        for key in list(elapsed):
            elapsed[key] += stall
        for register in access.real_defs:
            last_read.pop(register, None)
            elapsed.pop(register, None)
        for register in access.real_uses:
            last_read[register] = (index, instruction.mnemonic)
            elapsed[register] = stall


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--opt",
        type=int,
        nargs="*",
        default=list(LEVELS),
        help="optimisation levels to include (default every level that schedules)",
    )
    args = parser.parse_args()

    from basalt.harvest.corpus import generate as generate_scalar
    from basalt.harvest.corpus_shapes import generate_shapes
    from basalt.harvest.corpus_tensor import generate_tensor
    from basalt.toolchain import find_toolchain
    from basalt.verify.latency import DEFAULT_MODEL, LatencyModel
    from basalt.verify.observed import ObservedStalls

    tc = find_toolchain()
    latencies = ROOT / "data" / "latency" / "rtx-5070-ti.json"
    observed_path = ROOT / "data" / "latency" / "observed-stalls-sm120a.json"
    model = LatencyModel.assumed().overlay(latencies) if latencies.is_file() else DEFAULT_MODEL
    observed = ObservedStalls.read(observed_path) if observed_path.is_file() else None

    snippets = generate_scalar() + generate_tensor() + generate_shapes()
    tasks = [(s, o) for s in snippets for o in args.opt]
    print(f"{_repo.provenance()}\n")
    print(f"compiling {len(snippets)} kernels at -O{', -O'.join(str(o) for o in args.opt)}")

    def run(task):
        snippet, opt = task
        program = _compile(tc, snippet, opt)
        return _one(program, model, observed) if program is not None else collections.Counter()

    with ThreadPoolExecutor(max_workers=(os.cpu_count() or 4) * 2) as pool:
        totals: collections.Counter = sum(pool.map(run, tasks), collections.Counter())

    distance = totals.pop("distance total", 0)
    seen = totals.pop("distance count", 0)
    print(f"\n{totals['instructions']} instructions\n")
    for key in sorted(totals):
        if key == "instructions":
            continue
        print(f"  {key:58} {totals[key]:6}")
    if seen:
        print(f"\n  mean distance where distance is the only cover: {distance / seen:.0f} cycles")
    return 0


if __name__ == "__main__":
    sys.exit(main())
