# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Building the ISA database end to end.

harvest -> pick one representative encoding per form -> probe its bits -> record.

Probing is the expensive half, roughly one nvdisasm invocation per form, so the
representative is chosen deliberately rather than taken at random: prefer an
encoding whose operands are distinct registers, because a form printed as
`IADD R5, R5, 0x2a` cannot distinguish the destination field from the source
field when both happen to hold 5. Picking a degenerate representative is a quiet
way to produce a field map that looks fine and is wrong.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..disasm import raw_arch
from ..encoding import Word
from ..harvest.corpus import generate as generate_scalar
from ..harvest.corpus_shapes import generate_shapes
from ..harvest.corpus_tensor import generate_tensor
from ..harvest.runner import HarvestResult, Observation, harvest
from ..probe.fields import BitRole, probe_word
from ..toolchain import Toolchain
from .database import InstructionForm, IsaDatabase, OperandField
from .operands import subfields

__all__ = ["build_database", "collect_representatives"]

_GUARD_PREFIX = re.compile(r"^@!?U?P(\d+|T)\s+")
_KIND = re.compile(r"^-?(UR|R|UP|P)(\d+|Z|T)\b")
_RNUM = re.compile(r"\bR(\d+)\b")


def _degeneracy(obs: Observation) -> int:
    """Lower is a better probe subject.

    Counts repeated register numbers in the operand text. Distinct operands make
    every field independently observable; repeated ones alias fields together.
    """
    nums = _RNUM.findall(obs.operands)
    return len(nums) - len(set(nums))


def operand_shape(operands: str) -> tuple[str, ...]:
    """The kinds of an instruction's operands, ignoring their values.

    One mnemonic can cover several genuinely different encodings. `IADD R2, R3,
    0x4` and `IADD R2, R3, R4` differ in bits outside every operand field, so a
    database holding one of them can describe the other only by accident. The
    shape is what tells them apart.

    The guard is deliberately not part of it. `@P0 IADD` and `IADD` are the same
    encoding with one fixed field set differently, so counting the guard would
    split every predicated form into a duplicate of itself.
    """
    # the guard rides on the first operand, so `@P0 R2, R3` splits as
    # ("@P0 R2", "R3"); skipping that part would drop a real operand
    operands = _GUARD_PREFIX.sub("", operands.strip(), count=1)

    kinds: list[str] = []
    depth = 0
    current = ""
    parts: list[str] = []
    for char in operands:
        if char in "[({":
            depth += 1
        elif char in "])}":
            depth -= 1
        if char == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += char
    if current.strip():
        parts.append(current)

    for part in parts:
        text = part.strip()
        if text.startswith("desc["):
            kinds.append("descriptor")
        elif text.startswith(("c[", "cx[")):
            kinds.append("constant")
        elif text.startswith("`"):
            kinds.append("label")
        elif (match := _KIND.match(text)) is not None:
            kinds.append({"R": "reg", "UR": "ureg", "P": "pred", "UP": "upred"}[match.group(1)])
        else:
            kinds.append("immediate")
    return tuple(kinds)


_FLOAT_TEXT = re.compile(r"^[+-]?(?:\d+\.\d|\d+(?:\.\d+)?e[+-]?\d+|INF|QNAN|NAN)", re.IGNORECASE)


def _reveals_float(observation) -> bool:
    """Did this bit show the operand printing as a float rather than a number?"""
    index = observation.changed_operand_index
    if index is None:
        return False
    for text in (observation.before, observation.after):
        parts = [p.strip() for p in text.split(",")]
        if index < len(parts) and _FLOAT_TEXT.match(parts[index]):
            return True
    return False


def collect_representatives(
    result: HarvestResult,
) -> dict[tuple[str, tuple[str, ...]], Observation]:
    """One encoding per mnemonic *and operand shape*, chosen to probe well.

    Keyed on the shape as well as the mnemonic because they are different
    encodings. Probing only the first shape seen leaves an assembler that has to
    refuse every other one, which was 600 of the 8,560 instructions in the
    corpus before this existed.
    """
    buckets: dict[tuple[str, tuple[str, ...]], list[Observation]] = defaultdict(list)
    for o in result.observations:
        buckets[(o.mnemonic, operand_shape(o.operands))].append(o)

    chosen: dict[tuple[str, tuple[str, ...]], Observation] = {}
    for key, obs in buckets.items():
        # prefer distinct operands, then a longer operand list (more slots to
        # attribute), then a stable tiebreak so rebuilds are deterministic
        chosen[key] = min(
            obs,
            key=lambda o: (_degeneracy(o), -len(o.operands.split(",")), o.encoding),
        )
    return chosen


def build_database(
    tc: Toolchain,
    *,
    arch: str = "sm_120a",
    include_tensor: bool = True,
    harvest_out: Path | None = None,
    progress: bool = True,
) -> tuple[IsaDatabase, HarvestResult]:
    # the control-flow kernels are harvested too: they emit predicated forms and
    # immediate positions the straight-line corpus never reaches
    snippets = generate_scalar() + generate_shapes()
    if include_tensor:
        snippets = snippets + generate_tensor()

    if progress:
        print(f"corpus: {len(snippets)} kernels")

    result = harvest(tc, snippets, arch=arch, progress=progress)
    if progress:
        print(f"harvest: {result.summary()}")
    if harvest_out is not None:
        result.write(harvest_out)

    reps = collect_representatives(result)
    if progress:
        print(f"probing {len(reps)} distinct forms")

    db = IsaDatabase(arch=arch, cuda_version=tc.version)
    raw = raw_arch(arch)

    def _probe_one(item) -> InstructionForm | None:
        (mnemonic, _shape), obs = item
        fmap = probe_word(tc, Word(int(obs.encoding, 16)), arch=raw)
        if fmap is None:
            return None

        operands = []
        for slot, bits in fmap.operand_fields().items():
            mine = [o for o in fmap.observations if o.bit in bits and o.role is BitRole.OPERAND]
            # the sample decides what an assembler thinks this field holds, so
            # take one that shows a float where any bit reveals one: `FFMA`'s
            # immediate prints as `1` and flipping a high bit keeps it looking
            # like an integer, which is how a float field is mistaken for a
            # plain value and given an integer to encode
            sample = next((o for o in mine if _reveals_float(o)), mine[0] if mine else None)
            operands.append(
                OperandField(
                    slot=slot,
                    bits=tuple(bits),
                    example_before=sample.before if sample else "",
                    example_after=sample.after if sample else "",
                    # the prober already recorded what each bit did to the text,
                    # so taking a composite operand apart costs nothing extra
                    subfields=subfields(mine),
                )
            )

        return InstructionForm(
            mnemonic=mnemonic,
            opcode=obs.opcode,
            modifiers=obs.modifiers,
            encoding=obs.encoding,
            payload=obs.payload,
            operand_text=obs.operands,
            operands=operands,
            opcode_bits=tuple(fmap.bits_with_role(BitRole.OPCODE)),
            modifier_bits=tuple(fmap.bits_with_role(BitRole.MODIFIER)),
            inert_bits=tuple(fmap.bits_with_role(BitRole.INERT)),
            invalid_bits=tuple(fmap.bits_with_role(BitRole.INVALID)),
            source_label=obs.source_label,
            source_family=obs.source_family,
        )

    # every probe is an independent chain of subprocess calls, so this scales
    # with cores rather than with the ISA
    workers = min(32, (os.cpu_count() or 4) * 2)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for form in pool.map(_probe_one, sorted(reps.items())):
            done += 1
            if form is not None:
                db.add(form)
            if progress and done % 50 == 0:
                print(f"  probed {done}/{len(reps)}")

    if progress:
        cov = db.coverage()
        print("database: " + ", ".join(f"{k.replace('_', ' ')}={v}" for k, v in cov.items()))
    return db, result
