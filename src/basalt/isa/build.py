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
from ..harvest.corpus_tensor import generate_tensor
from ..harvest.runner import HarvestResult, Observation, harvest
from ..probe.fields import BitRole, probe_word
from ..toolchain import Toolchain
from .database import InstructionForm, IsaDatabase, OperandField

__all__ = ["build_database", "collect_representatives"]

_RNUM = re.compile(r"\bR(\d+)\b")


def _degeneracy(obs: Observation) -> int:
    """Lower is a better probe subject.

    Counts repeated register numbers in the operand text. Distinct operands make
    every field independently observable; repeated ones alias fields together.
    """
    nums = _RNUM.findall(obs.operands)
    return len(nums) - len(set(nums))


def collect_representatives(result: HarvestResult) -> dict[str, Observation]:
    """One encoding per mnemonic, chosen to make probing informative."""
    buckets: dict[str, list[Observation]] = defaultdict(list)
    for o in result.observations:
        buckets[o.mnemonic].append(o)

    chosen: dict[str, Observation] = {}
    for mnemonic, obs in buckets.items():
        # prefer distinct operands, then a longer operand list (more slots to
        # attribute), then a stable tiebreak so rebuilds are deterministic
        chosen[mnemonic] = min(
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
    snippets = generate_scalar()
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

    def _probe_one(item: tuple[str, Observation]) -> InstructionForm | None:
        mnemonic, obs = item
        fmap = probe_word(tc, Word(int(obs.encoding, 16)), arch=raw)
        if fmap is None:
            return None

        operands = []
        for slot, bits in fmap.operand_fields().items():
            sample = next(
                (o for o in fmap.observations if o.bit in bits and o.role is BitRole.OPERAND),
                None,
            )
            operands.append(
                OperandField(
                    slot=slot,
                    bits=tuple(bits),
                    example_before=sample.before if sample else "",
                    example_after=sample.after if sample else "",
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
