#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Reschedule one kernel and show which control fields moved, in seconds.

`scripts/roundtrip_corpus.py` runs the whole corpus and takes minutes, which is
right for a control and wrong for a debugging loop. Nearly every scheduler defect
found so far was exposed by one kernel at one optimisation level, and the corpus
runner reports only that the bytes disagreed, never which field caused it.

    python scripts/probe_kernel.py s_tile_matmul
    python scripts/probe_kernel.py s_tile_matmul --opt 1 --all
    python scripts/probe_kernel.py s_ --run

Compiling and rescheduling needs no GPU, so the field diff and the shape check
print on any machine. `--run` adds the round trip on the card and needs an
sm_120 part. Nothing here rebuilds the ISA database or the mined stall table,
because a scheduler change cannot move either; `docs/METHOD.md` says which half
of the pipeline feeds them.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
from pathlib import Path

import _repo

_repo.use_repo_source()

ROOT = _repo.ROOT
ARCH = "sm_120a"
BUFFER = 256
THREADS = 32

# the same four the corpus runner uses, so a verdict here means what it means there
PATTERNS: tuple[bytes, ...] = (
    bytes((i * 37 + 11) & 0xFF for i in range(256)),
    bytes((i * 211 + 173) & 0xFF for i in range(256)),
    bytes(0xFF if i % 3 else 0x00 for i in range(256)),
    bytes((i * i + 7) & 0xFF for i in range(256)),
)


def _snippets():
    from basalt.harvest.corpus import generate as generate_scalar
    from basalt.harvest.corpus_shapes import generate_shapes
    from basalt.harvest.corpus_tensor import generate_tensor

    return generate_scalar() + generate_tensor() + generate_shapes()


def _reschedule(tc, snippet, opt: int, work: Path):
    """Compile one snippet and reschedule it, returning both and the vendor cubin."""
    from basalt.disasm import disassemble_program
    from basalt.sched.scheduler import schedule_program
    from basalt.verify.latency import DEFAULT_MODEL, LatencyModel
    from basalt.verify.observed import ObservedStalls

    latencies = ROOT / "data" / "latency" / "rtx-5070-ti.json"
    observed_path = ROOT / "data" / "latency" / "observed-stalls-sm120a.json"
    model = LatencyModel.assumed().overlay(latencies) if latencies.is_file() else DEFAULT_MODEL
    observed = ObservedStalls.read(observed_path) if observed_path.is_file() else None

    work.mkdir(parents=True, exist_ok=True)
    src = work / f"{snippet.name}.ptx"
    src.write_text(snippet.ptx)
    vendor = work / f"{snippet.name}.O{opt}.v.cubin"
    built = tc.run(
        [str(tc.ptxas), f"-arch={ARCH}", f"-O{opt}", "-o", str(vendor), str(src)],
        check=False,
        timeout=60.0,
    )
    if built.returncode != 0:
        raise SystemExit(f"{snippet.name}: ptxas -O{opt} refused it\n{built.stderr}")
    program = disassemble_program(tc, vendor)
    return program, schedule_program(program, model, observed=observed), vendor


def _shapes(program, words) -> list[str]:
    """Control-word pairings ptxas never emits, found in a rescheduled program.

    A cheap standing check rather than a latency question. Each of these is wrong
    on its face, so finding one needs no card and no reference output, and the
    self-wait line is what finally located the scoreboard defect in finding 24.
    """
    from basalt.encoding import NO_BARRIER
    from basalt.sched.scheduler import _SCOREBOARD_OPERAND

    signalled = 0
    waited = 0
    for index, word in enumerate(words):
        if word is None:
            continue
        for name in ("write_barrier", "read_barrier"):
            barrier = word.field(name)
            if barrier != NO_BARRIER:
                signalled |= 1 << barrier
        waited |= word.field("wait_mask")
        # `DEPBAR.LE SB0, 0x0` waits by naming the scoreboard in its operand
        instruction = program.instructions[index]
        if instruction.word is not None:
            for match in _SCOREBOARD_OPERAND.finditer(instruction.operands):
                waited |= 1 << int(match.group(1))

    notes = []
    for index, word in enumerate(words):
        if word is None or program.instructions[index].word is None:
            continue
        barrier = word.field("write_barrier")
        if barrier != NO_BARRIER and (word.field("wait_mask") >> barrier) & 1:
            name = program.instructions[index].mnemonic
            notes.append(f"#{index} {name}: waits on SB{barrier}, the scoreboard it signals")
    for sb in range(6):
        if (waited >> sb) & 1 and not (signalled >> sb) & 1:
            notes.append(f"SB{sb}: waited on, and nothing signals it")
        if (signalled >> sb) & 1 and not (waited >> sb) & 1:
            notes.append(f"SB{sb}: signalled, and nothing waits on it")
    return notes


def _render(word) -> str:
    from basalt.encoding import STALL_YIELD

    stall = word.field("stall")
    shown = "safe" if stall == STALL_YIELD else str(stall)
    return (
        f"stall={shown:>4} wait={word.field('wait_mask'):06b} "
        f"wb={word.field('write_barrier')} rb={word.field('read_barrier')}"
    )


def _diff(program, words, show_all: bool) -> int:
    """Print the control fields that changed, returning how many words moved."""
    from basalt.encoding import CONTROL_FIELDS

    names = [f.name for f in CONTROL_FIELDS]
    moved = 0
    for index, instruction in enumerate(program.instructions):
        before = instruction.word
        after = words[index] if index < len(words) else None
        if before is None or after is None:
            continue
        changes = [n for n in names if before.field(n) != after.field(n)]
        if not changes and not show_all:
            continue
        moved += bool(changes)
        marker = "  " if changes else "= "
        text = f"{instruction.mnemonic} {instruction.operands}".strip()
        print(f"{marker}#{index:<4} {text[:56]}")
        print(f"      vendor  {_render(before)}")
        print(f"      basalt  {_render(after)}   {' '.join(changes)}")
    return moved


def _run(vendor: Path, mine: Path, entry: str) -> str:
    """Run both cubins against every pattern and say whether they agreed."""
    from basalt.gpu.driver import Device

    with Device(0) as device:
        source = device.alloc(BUFFER)
        destination = device.alloc(BUFFER)

        def execute(path: Path) -> bytes:
            module = device.load_cubin(path.read_bytes())
            function = module.function(entry)
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
            module.unload()
            return b"".join(outputs)

        try:
            theirs = execute(vendor)
        except Exception as exc:
            return f"unrunnable: the vendor's own cubin failed here ({type(exc).__name__})"
        try:
            ours = execute(mine)
        except Exception as exc:
            return f"basalt-faulted: {type(exc).__name__}"
        # stable until something else has used the card, for a kernel reading
        # memory this harness never initialises
        if execute(vendor) != theirs:
            return "not-reproducible: the vendor disagreed with itself"
    return "match" if ours == theirs else "MISMATCH"


def _entry_of(snippet) -> str:
    return next(
        line.split("(")[0].split()[-1] for line in snippet.ptx.splitlines() if ".entry" in line
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("names", nargs="*", help="corpus kernel names, or a prefix of one")
    parser.add_argument("--opt", type=int, default=3, choices=(1, 2, 3), help="ptxas -O level")
    parser.add_argument("--all", action="store_true", help="print unchanged instructions too")
    parser.add_argument("--run", action="store_true", help="also run both on the GPU")
    parser.add_argument("--work", type=Path, default=None, help="scratch directory")
    parser.add_argument("--list", action="store_true", help="print every corpus kernel name")
    args = parser.parse_args()

    from basalt.toolchain import find_toolchain

    snippets = _snippets()
    if args.list:
        for snippet in snippets:
            print(snippet.name)
        return 0
    if not args.names:
        parser.error("name at least one kernel, or pass --list")

    wanted = [s for s in snippets if any(s.name.startswith(n) for n in args.names)]
    if not wanted:
        raise SystemExit(f"no corpus kernel matches {args.names}; --list shows them all")

    work = args.work or Path(os.environ.get("TMP", "/tmp")) / "basalt-probe"
    tc = find_toolchain()
    failures = 0

    for snippet in wanted:
        program, result, vendor = _reschedule(tc, snippet, args.opt, work)
        print(f"\n=== {snippet.name} -O{args.opt}: {result.summary()}")
        moved = _diff(program, result.words, args.all)
        print(f"  {moved} of {len(program.instructions)} control words changed")

        for note in _shapes(program, result.words):
            print(f"  ! {note}")
            failures += 1
        for note in result.unplaceable:
            print(f"  ! unplaceable {note}")

        if args.run:
            from basalt.asm.cubin import Cubin

            cubin = Cubin.load(vendor)
            for slot, word in enumerate(result.words):
                if program.instructions[slot].word is not None:
                    cubin.write_word(slot, word)
            mine = vendor.with_suffix(".b.cubin")
            cubin.save(mine)
            verdict = _run(vendor, mine, _entry_of(snippet))
            print(f"  {verdict}")
            failures += verdict.startswith(("MISMATCH", "basalt-faulted"))

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
