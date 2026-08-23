# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""The instruction database: what basalt knows about sm_120.

A database entry is one *form*, meaning one mnemonic with its modifier suffixes
resolved, such as `IMAD.WIDE.U32`. Each form carries a real encoding observed
from the vendor compiler, the bit runs that were measured to control each
operand slot, and the bits that were measured to do nothing.

Two properties are load-bearing and both are checked when the database is built:

*Every entry is grounded in an encoding that actually assembled.* Nothing here
is inferred from documentation or guessed from a neighbouring architecture.

*Every entry records how it was produced.* The ptxas build, the arch, and the
PTX that caused it, because encodings are not promised to be stable across
compiler releases and a database that cannot be re-derived is a liability.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ..encoding import CONTROL_FIELDS, Word

__all__ = ["InstructionForm", "IsaDatabase", "OperandField"]

SCHEMA_VERSION = 2


def _runs(bits: list[int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for b in sorted(bits):
        if out and b == out[-1][0] + out[-1][1]:
            lo, width = out[-1]
            out[-1] = (lo, width + 1)
        else:
            out.append((b, 1))
    return out


@dataclass(frozen=True, slots=True)
class OperandField:
    """Bits that were observed to move one operand slot."""

    slot: int
    bits: tuple[int, ...]
    example_before: str = ""
    example_after: str = ""
    # which bits inside a composite operand carry the bank, offset and base.
    # empty when it is not composite, or when the prober could not read it
    subfields: dict[str, tuple[int, ...]] = field(default_factory=dict)

    @property
    def runs(self) -> list[tuple[int, int]]:
        return _runs(list(self.bits))

    @property
    def width(self) -> int:
        return len(self.bits)

    def describe(self) -> str:
        return ",".join(f"{lo}:{lo + w - 1}" if w > 1 else str(lo) for lo, w in self.runs)


@dataclass
class InstructionForm:
    """One mnemonic, one grounded encoding, and its measured field layout."""

    mnemonic: str
    opcode: str
    modifiers: tuple[str, ...]
    encoding: str  # 32 hex chars, high half first
    payload: str  # control bits zeroed
    operand_text: str
    operands: list[OperandField] = field(default_factory=list)
    opcode_bits: tuple[int, ...] = ()
    modifier_bits: tuple[int, ...] = ()
    inert_bits: tuple[int, ...] = ()
    invalid_bits: tuple[int, ...] = ()
    source_label: str = ""
    source_family: str = ""

    @property
    def word(self) -> Word:
        return Word(int(self.encoding, 16))

    @property
    def operand_count(self) -> int:
        return len(self.operands)

    def describe(self) -> str:
        slots = " ".join(f"[{o.slot}]{o.describe()}" for o in self.operands)
        return f"{self.mnemonic:<32} {slots}"


@dataclass
class IsaDatabase:
    """A queryable, serialisable set of instruction forms."""

    arch: str
    cuda_version: str
    generated_utc: str = ""
    forms: dict[str, InstructionForm] = field(default_factory=dict)
    # the other operand shapes of a mnemonic already in `forms`, kept beside the
    # canonical one so existing lookups keep working
    variants: dict[str, list[InstructionForm]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.generated_utc:
            self.generated_utc = datetime.now(UTC).isoformat(timespec="seconds")

    # ---- queries -------------------------------------------------------

    def __len__(self) -> int:
        return len(self.forms)

    def __contains__(self, mnemonic: str) -> bool:
        return mnemonic in self.forms

    def get(self, mnemonic: str) -> InstructionForm | None:
        return self.forms.get(mnemonic)

    def by_opcode(self, opcode: str) -> list[InstructionForm]:
        return sorted(
            (f for f in self.forms.values() if f.opcode == opcode),
            key=lambda f: f.mnemonic,
        )

    @property
    def opcodes(self) -> list[str]:
        return sorted({f.opcode for f in self.forms.values()})

    def add(self, form: InstructionForm) -> None:
        """Record a form, keeping the most informative one as canonical.

        The canonical form is the one with the most attributed operand slots,
        since that is the one worth showing when someone asks about a mnemonic.
        The rest become variants, which is what makes assembling the other
        shapes possible at all.
        """
        existing = self.forms.get(form.mnemonic)
        if existing is None:
            self.forms[form.mnemonic] = form
            return
        if any(form.encoding == other.encoding for other in self.shapes(form.mnemonic)):
            # the identical word, harvested twice
            return
        if len(form.operands) > len(existing.operands):
            self.forms[form.mnemonic] = form
            self.variants.setdefault(form.mnemonic, []).append(existing)
        else:
            self.variants.setdefault(form.mnemonic, []).append(form)

    def shapes(self, mnemonic: str) -> list[InstructionForm]:
        """Every recorded encoding of a mnemonic, canonical first."""
        canonical = self.forms.get(mnemonic)
        if canonical is None:
            return []
        return [canonical, *self.variants.get(mnemonic, [])]

    def coverage(self) -> dict[str, int]:
        """Numbers the README and CI report rather than assert in prose."""
        with_ops = [f for f in self.forms.values() if f.operands]
        return {
            "forms": len(self.forms),
            "opcodes": len(self.opcodes),
            "forms_with_operand_map": len(with_ops),
            "tensor_forms": len([f for f in self.forms.values() if "MMA" in f.opcode]),
            "uniform_forms": len([f for f in self.forms.values() if f.opcode.startswith("U")]),
        }

    # ---- persistence ---------------------------------------------------

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "arch": self.arch,
            "cuda_version": self.cuda_version,
            "generated_utc": self.generated_utc,
            "control_fields": [
                {"name": f.name, "lo": f.lo, "width": f.width, "note": f.note}
                for f in CONTROL_FIELDS
            ],
            "coverage": self.coverage(),
            "forms": {name: _form_payload(form) for name, form in sorted(self.forms.items())},
            "variants": {
                name: [_form_payload(v) for v in forms]
                for name, forms in sorted(self.variants.items())
            },
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    @classmethod
    def read(cls, path: Path) -> IsaDatabase:
        raw = json.loads(path.read_text())
        if raw.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"{path} is schema v{raw.get('schema_version')}, this build expects "
                f"v{SCHEMA_VERSION}; rebuild it with `basalt build-isa`"
            )
        db = cls(
            arch=raw["arch"],
            cuda_version=raw["cuda_version"],
            generated_utc=raw["generated_utc"],
        )
        for name, f in raw["forms"].items():
            db.forms[name] = _read_form(f)
        for name, forms in (raw.get("variants") or {}).items():
            db.variants[name] = [_read_form(v) for v in forms]
        return db


def _form_payload(form: InstructionForm) -> dict:
    """One form as plain JSON. Shared so variants serialise identically."""
    return {
        **asdict(form),
        "modifiers": list(form.modifiers),
        "opcode_bits": list(form.opcode_bits),
        "modifier_bits": list(form.modifier_bits),
        "inert_bits": list(form.inert_bits),
        "invalid_bits": list(form.invalid_bits),
        "operands": [
            {
                **asdict(o),
                "bits": list(o.bits),
                "subfields": {k: list(v) for k, v in sorted(o.subfields.items())},
            }
            for o in form.operands
        ],
    }


def _read_form(f: dict) -> InstructionForm:
    return InstructionForm(
        mnemonic=f["mnemonic"],
        opcode=f["opcode"],
        modifiers=tuple(f["modifiers"]),
        encoding=f["encoding"],
        payload=f["payload"],
        operand_text=f["operand_text"],
        operands=[
            OperandField(
                slot=o["slot"],
                bits=tuple(o["bits"]),
                example_before=o.get("example_before", ""),
                example_after=o.get("example_after", ""),
                subfields={k: tuple(v) for k, v in (o.get("subfields") or {}).items()},
            )
            for o in f["operands"]
        ],
        opcode_bits=tuple(f["opcode_bits"]),
        modifier_bits=tuple(f["modifier_bits"]),
        inert_bits=tuple(f["inert_bits"]),
        invalid_bits=tuple(f["invalid_bits"]),
        source_label=f.get("source_label", ""),
        source_family=f.get("source_family", ""),
    )
