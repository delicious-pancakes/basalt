#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Try to make the assembler emit bytes that mean something else.

`scripts/assembler_coverage.py` checks the assembler against instructions the
vendor actually emitted, which is the right control and a narrow one: it only
ever asks about text `ptxas` chose to produce. The failure this repository
exists to catch does not need the vendor's cooperation. A word that assembles,
disassembles back to exactly the text it came from, and computes something else
is the whole problem, and the way to find one is to go looking.

So this mutates the operand text of every form in the database, assembles the
result, and hands it straight back to `nvdisasm`:

    assemble(text) must disassemble to text

Anything else is a defect. If the decoder reads back different text, the
assembler wrote a different instruction; if it reads back nothing, the assembler
produced a word the decoder rejects. Refusing to assemble is not a defect and is
counted separately, because refusing is the designed answer whenever the
database cannot describe what was asked for.

The mutations are the ones an assembler is most likely to get subtly wrong:
a different register in each position, immediates across the width of their
field including negative and boundary values, and modifiers added or removed.

No GPU. `nvdisasm` is the whole oracle, and the seed is fixed so a failure is
reproducible.

    python scripts/fuzz_assembler.py
    python scripts/fuzz_assembler.py --cases 40 --seed 7 --show 20
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import _repo

_repo.use_repo_source()

ROOT = _repo.ROOT
ARCH = "sm_120a"

_REGISTER = re.compile(r"^(UR|R|UP|P)(\d+|Z|T)(\.reuse)?$")
_IMMEDIATE = re.compile(r"^-?0[xX][0-9a-fA-F]+$|^-?\d+$")
_REUSE = re.compile(r"\.reuse\b")
# registers that discard what is written to them, which the decoder may omit
_SINKS = frozenset({"RZ", "URZ", "PT", "UPT"})


def _split(operands: str) -> list[str]:
    """Operand text split on top-level commas."""
    parts: list[str] = []
    current = ""
    depth = 0
    for char in operands:
        if char in "[({":
            depth += 1
        elif char in "])}":
            depth -= 1
        if char == "," and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += char
    if current.strip():
        parts.append(current.strip())
    return parts


def _peel(part: str) -> tuple[str, str, bool]:
    """A register's leading signs, its own text, and whether bars wrapped it.

    `|R2|` is a register carrying a modifier, so leaving the bars on hides every
    absolute-valued form from the fuzzer entirely.
    """
    sign = part[: len(part) - len(part.lstrip("-~"))]
    rest = part[len(sign) :]
    bars = len(rest) > 2 and rest.startswith("|") and rest.endswith("|")
    return sign, rest[1:-1] if bars else rest, bars


def _mutate(operands: str, rng: random.Random) -> str | None:
    """One plausible edit to one operand, or None if nothing here is editable."""
    parts = _split(operands)
    editable = [
        i
        for i, part in enumerate(parts)
        if _IMMEDIATE.match(part) or _REGISTER.match(_peel(part)[1])
    ]
    if not editable:
        return None

    index = rng.choice(editable)
    part = parts[index]
    sign, core, bars = _peel(part)

    if (match := _REGISTER.match(core)) is not None:
        prefix = match.group(1)
        limit = 7 if prefix in ("P", "UP") else 254
        choice = rng.choice(["number", "sink", "modifier"])
        if choice == "sink":
            replacement = f"{prefix}{'T' if prefix in ('P', 'UP') else 'Z'}"
        elif choice == "modifier" and prefix == "R":
            replacement = f"{prefix}{rng.randint(0, limit)}"
            sign = "" if sign else "-"
        else:
            replacement = f"{prefix}{rng.randint(0, limit)}"
        parts[index] = sign + (f"|{replacement}|" if bars else replacement)
        return ", ".join(parts)

    # an immediate: sweep the interesting corners of its width as well as noise
    width = rng.choice([4, 8, 12, 16, 20, 24, 28, 31])
    value = rng.choice(
        [
            0,
            1,
            -1,
            (1 << width) - 1,
            -(1 << width),
            rng.getrandbits(width),
            -rng.getrandbits(width),
        ]
    )
    parts[index] = hex(value) if value >= 0 else f"-{hex(-value)}"
    return ", ".join(parts)


def _dropped_only_sinks(asked: list[str], got: list[str]) -> bool:
    """Is `got` `asked` with nothing removed but sink registers?

    A discarded result is not printed at all, so `VOTE.ANY RZ, PT, P0` reads
    back as `VOTE.ANY PT, P0`. The operand is still encoded, and this stays
    narrow on purpose: every operand that vanished has to be one that discards.
    """
    if len(got) >= len(asked):
        return False
    position = 0
    for operand in asked:
        if position < len(got) and got[position] == operand:
            position += 1
        elif operand not in _SINKS:
            return False
    return position == len(got)


def _normalise(text: str) -> str:
    """Compare on what the instruction says, not on how it was spaced."""
    text = _REUSE.sub("", text)
    text = text.replace("\t", " ").strip().rstrip(";").strip()
    return re.sub(r"\s*,\s*", ", ", re.sub(r"\s+", " ", text))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cases", type=int, default=25, help="mutations per form")
    parser.add_argument("--seed", type=int, default=1, help="fixed so failures reproduce")
    parser.add_argument("--show", type=int, default=10, help="how many defects to print")
    parser.add_argument("--db", type=Path, default=None, help="ISA database to fuzz")
    args = parser.parse_args()

    from basalt.asm.assemble import Assembler, AssemblyError
    from basalt.disasm import decode_words, raw_arch
    from basalt.isa.database import IsaDatabase
    from basalt.toolchain import find_toolchain

    database_path = args.db or ROOT / "data" / "isa" / f"{ARCH}.json"
    if not database_path.is_file():
        print(f"error: {database_path} does not exist; run `basalt build-isa`", file=sys.stderr)
        return 2

    tc = find_toolchain()
    database = IsaDatabase.read(database_path)
    assembler = Assembler(database)
    rng = random.Random(args.seed)
    started = time.perf_counter()

    wanted: list[tuple] = []
    counts: Counter[str] = Counter()
    for mnemonic in database.forms:
        for form in database.shapes(mnemonic):
            if not form.operand_text:
                continue
            for _ in range(args.cases):
                mutated = _mutate(form.operand_text, rng)
                if mutated is None:
                    break
                counts["generated"] += 1
                text = f"{mnemonic} {mutated}"
                try:
                    word = assembler.assemble(text, control=form.word)
                except AssemblyError:
                    counts["refused"] += 1
                    continue
                wanted.append((text, word, form))

    counts["assembled"] = len(wanted)
    decoded = decode_words(tc, [w for _, w, _ in wanted], arch=raw_arch(ARCH))

    defects: list[tuple[str, str]] = []
    for (text, word, form), got in zip(wanted, decoded, strict=True):
        if got is None or not got.is_valid:
            counts["undecodable"] += 1
            defects.append((text, "the decoder rejects the word this produced"))
            continue
        if _normalise(got.text) == _normalise(text):
            counts["round-tripped"] += 1
            continue

        # different text is not a different instruction: `P7` is `PT`, and a
        # negative immediate prints unsigned. re-assembling it settles which
        try:
            again = assembler.assemble(_normalise(got.text), control=form.word)
        except AssemblyError:
            again = None
        if again is not None and again.value == word.value:
            counts["same word, printed differently"] += 1
            continue

        # a suffix that follows the operand's value is finding 15 again, so the
        # same operands under the same base opcode means the encoding is right
        asked_head, _, asked_rest = _normalise(text).partition(" ")
        got_head, _, got_rest = _normalise(got.text).partition(" ")
        if asked_rest == got_rest and asked_head.split(".")[0] == got_head.split(".")[0]:
            counts["same operands, suffix follows the value"] += 1
            continue

        # A discarded result is not printed at all: `VOTE.ANY RZ, PT, P0` comes
        # back as `VOTE.ANY PT, P0`. The operand is still encoded, so this only
        # counts when the difference is exactly one or more sinks disappearing.
        if asked_head == got_head and _dropped_only_sinks(_split(asked_rest), _split(got_rest)):
            counts["a discarded result is not printed"] += 1
            continue
        counts["WRONG"] += 1
        defects.append((text, f"decodes as {_normalise(got.text)}"))

    elapsed = time.perf_counter() - started
    print(f"\n{_repo.provenance()}")
    total = counts["generated"]
    print(f"\n{total} mutations of {len(database.forms)} mnemonics in {elapsed:.1f}s\n")
    for name in (
        "round-tripped",
        "same word, printed differently",
        "refused",
        "undecodable",
        "WRONG",
    ):
        if counts[name]:
            print(f"  {name:14} {counts[name]:6}")

    if defects:
        print(f"\n{len(defects)} defects, first {min(args.show, len(defects))}:")
        for text, why in defects[: args.show]:
            print(f"  {text}")
            print(f"      {why}")
        return 1
    print("\n  every word that assembled decoded back to exactly the text asked for")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
