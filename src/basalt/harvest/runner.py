# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Running the corpus through ptxas and collecting what comes out.

Each snippet is assembled at several optimisation levels. That is not
redundancy: ptxas picks different instruction forms and different scheduling
under -O0 than under -O3, and the harvester wants both. A form we never asked
for is a hole in the database, and holes are what make an ISA table quietly
wrong rather than loudly wrong.

Failures are recorded, not raised. A snippet ptxas rejects is a negative result
worth keeping, because it tells us a PTX form does not lower to this target.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from ..disasm import Instruction, disassemble_cubin
from ..encoding import Word
from ..toolchain import Toolchain
from .corpus import Snippet

__all__ = ["Observation", "HarvestResult", "harvest", "OPT_LEVELS"]

# -O0 keeps the shape of the PTX and exposes plain forms; -O3 is what real code
# gets and exposes fused and rescheduled forms. Both are worth having.
OPT_LEVELS: tuple[int, ...] = (0, 3)


@dataclass(frozen=True, slots=True)
class Observation:
    """One SASS instruction seen in the wild, with what produced it."""

    mnemonic: str
    opcode: str
    modifiers: tuple[str, ...]
    operands: str
    encoding: str          # 32 hex chars, high half first
    payload: str           # encoding with control bits zeroed
    control: dict[str, int]
    source_kernel: str
    source_label: str
    source_family: str
    opt_level: int

    @classmethod
    def build(cls, instr: Instruction, snip: Snippet, opt: int) -> "Observation":
        word = instr.word
        assert word is not None
        return cls(
            mnemonic=instr.mnemonic,
            opcode=instr.opcode,
            modifiers=instr.modifiers,
            operands=instr.operands,
            encoding=str(word),
            payload=f"{word.payload:032x}",
            control=word.control,
            source_kernel=snip.name,
            source_label=snip.label,
            source_family=snip.family,
            opt_level=opt,
        )


@dataclass
class HarvestResult:
    """Everything one harvest run produced, plus its provenance."""

    cuda_version: str
    arch: str
    generated_utc: str
    observations: list[Observation] = field(default_factory=list)
    rejected: dict[str, str] = field(default_factory=dict)

    @property
    def opcodes(self) -> set[str]:
        return {o.opcode for o in self.observations}

    @property
    def mnemonics(self) -> set[str]:
        return {o.mnemonic for o in self.observations}

    def summary(self) -> str:
        return (
            f"{len(self.observations)} observations, "
            f"{len(self.mnemonics)} distinct mnemonics, "
            f"{len(self.opcodes)} distinct opcodes, "
            f"{len(self.rejected)} snippets rejected"
        )

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cuda_version": self.cuda_version,
            "arch": self.arch,
            "generated_utc": self.generated_utc,
            "counts": {
                "observations": len(self.observations),
                "mnemonics": len(self.mnemonics),
                "opcodes": len(self.opcodes),
                "rejected": len(self.rejected),
            },
            "observations": [asdict(o) for o in self.observations],
            "rejected": self.rejected,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _one(tc: Toolchain, snip: Snippet, arch: str, opt: int) -> tuple[list[Observation], str | None]:
    with TemporaryDirectory(prefix="basalt-harvest-") as tmp:
        src = Path(tmp) / f"{snip.name}.ptx"
        cubin = Path(tmp) / f"{snip.name}.cubin"
        src.write_text(snip.ptx)

        res = tc.run(
            [str(tc.ptxas), f"-arch={arch}", f"-O{opt}", "-o", str(cubin), str(src)],
            check=False,
            timeout=60.0,
        )
        if res.returncode != 0 or not cubin.exists():
            # ptxas splits diagnostics across both streams depending on the kind
            # of failure, so scan them together and keep the first line that
            # actually names an error rather than echoing the source
            blob = res.stderr + "\n" + res.stdout
            reason = next(
                (
                    ln.split("error", 1)[-1].lstrip(" :\t")
                    for ln in blob.splitlines()
                    if "error" in ln.lower()
                ),
                next((ln.strip() for ln in blob.splitlines() if ln.strip()), "unknown"),
            )
            return [], reason.strip()

        instrs = disassemble_cubin(tc, cubin)

    obs = [Observation.build(i, snip, opt) for i in instrs if i.is_valid and i.word is not None]
    return obs, None


def harvest(
    tc: Toolchain,
    snippets: list[Snippet],
    *,
    arch: str = "sm_120a",
    opt_levels: tuple[int, ...] = OPT_LEVELS,
    jobs: int | None = None,
    progress: bool = True,
) -> HarvestResult:
    """Assemble every snippet at every opt level and collect the SASS."""
    result = HarvestResult(
        cuda_version=tc.version,
        arch=arch,
        generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )

    tasks = [(s, o) for s in snippets for o in opt_levels]
    # subprocess spawn dominates, so oversubscribe the core count a little
    workers = jobs or min(32, (os.cpu_count() or 4) * 2)
    done = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one, tc, s, arch, o): (s, o) for s, o in tasks}
        for fut in as_completed(futures):
            snip, opt = futures[fut]
            obs, reason = fut.result()
            result.observations.extend(obs)
            if reason is not None:
                result.rejected[f"{snip.label}@O{opt}"] = reason
            done += 1
            if progress and done % 100 == 0:
                print(f"  {done}/{len(tasks)} kernels, {len(result.observations)} instructions")

    result.observations.sort(key=lambda o: (o.opcode, o.mnemonic, o.encoding))
    return result


def distinct_by_payload(observations: list[Observation]) -> dict[str, Observation]:
    """Collapse observations that differ only in scheduling.

    Two encodings with the same payload are the same instruction issued under
    different control bits, which is one instruction for ISA purposes and two
    for scheduling purposes. The ISA database wants the former.
    """
    out: dict[str, Observation] = {}
    for o in observations:
        out.setdefault(o.payload, o)
    return out
