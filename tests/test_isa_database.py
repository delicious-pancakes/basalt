# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Invariants the committed instruction database has to hold.

The database is generated data that is nonetheless checked in, so that a
consumer of basalt does not have to run a harvest before the assembler or the
checker is useful. That trade is only sound if the committed copy is held to the
same standard as code, which is what this file is for.

These run without a toolchain: they read the committed JSON and nothing else.
"""

from __future__ import annotations

import json

import pytest

from basalt.encoding import CONTROL_FIELDS, Word
from basalt.isa.database import IsaDatabase
from basalt.paths import ISA_DATABASE

DB_PATH = ISA_DATABASE


@pytest.fixture(scope="module")
def db() -> IsaDatabase:
    if not DB_PATH.is_file():
        pytest.skip(f"{DB_PATH} not present; run `basalt build-isa`")
    return IsaDatabase.read(DB_PATH)


class TestShape:
    def test_database_is_not_empty(self, db):
        assert len(db) > 100

    def test_arch_and_compiler_are_recorded(self, db):
        """An encoding without the compiler that produced it is not reproducible."""
        assert db.arch.startswith("sm_")
        assert db.cuda_version.startswith("V")

    def test_every_form_carries_an_encoding(self, db):
        for name, form in db.forms.items():
            assert len(form.encoding) == 32, f"{name} has a malformed encoding"
            int(form.encoding, 16)


class TestMnemonics:
    def test_no_mnemonic_is_a_guard_predicate(self, db):
        """A guard is printed before the opcode, so misparsing yields `@P1` forms.

        This was a real bug: predicated instructions had their guard read as the
        mnemonic, which corrupted the latency lookup and the def/use analysis for
        a large minority of real instructions.
        """
        offenders = [name for name in db.forms if name.startswith("@")]
        assert not offenders, f"guard predicates recorded as mnemonics: {offenders}"

    def test_every_mnemonic_starts_with_a_letter(self, db):
        offenders = [name for name in db.forms if not name[:1].isalpha()]
        assert not offenders, offenders

    def test_mnemonic_matches_its_opcode_and_modifiers(self, db):
        for name, form in db.forms.items():
            rebuilt = ".".join([form.opcode, *form.modifiers])
            assert rebuilt == name, f"{name} does not decompose to {rebuilt}"


class TestFieldMaps:
    def test_operand_bits_never_overlap_the_control_section(self, db):
        """An operand field inside the control word would mean the probe is wrong."""
        control = {b for f in CONTROL_FIELDS for b in range(f.lo, f.lo + f.width)}
        for name, form in db.forms.items():
            for operand in form.operands:
                clash = set(operand.bits) & control
                assert not clash, f"{name} operand {operand.slot} overlaps control bits {clash}"

    def test_operand_slots_are_distinct(self, db):
        for name, form in db.forms.items():
            slots = [o.slot for o in form.operands]
            assert len(slots) == len(set(slots)), f"{name} has duplicate operand slots"

    def test_operand_bits_are_within_the_word(self, db):
        for name, form in db.forms.items():
            for operand in form.operands:
                assert all(0 <= b < 128 for b in operand.bits), name

    def test_most_forms_have_an_operand_map(self, db):
        """A form with no attributed bits is a hole, and holes should stay rare."""
        mapped = sum(1 for f in db.forms.values() if f.operands)
        assert mapped / len(db) > 0.9


class TestPayload:
    def test_payload_is_the_encoding_with_control_zeroed(self, db):
        for name, form in db.forms.items():
            assert Word(int(form.encoding, 16)).payload == int(form.payload, 16), name


class TestCoverage:
    def test_the_low_precision_tensor_family_is_present(self, db):
        """The FP8, FP6 and FP4 tensor forms are the point of the tensor corpus."""
        qmma = [n for n in db.forms if n.startswith("QMMA")]
        assert qmma, "no QMMA forms; the tensor corpus did not reach the low-precision path"
        types = {"E4M3", "E5M2", "E3M2", "E2M3", "E2M1"}
        seen = {t for name in qmma for t in types if t in name}
        assert seen == types, f"missing low-precision types: {sorted(types - seen)}"

    def test_the_block_scaled_family_is_present(self, db):
        """Scale-factor forms carry a per-block exponent operand."""
        assert any(".SF." in n for n in db.forms), "no scale-factor tensor forms"

    def test_matrix_movement_instructions_are_present(self, db):
        for opcode in ("LDSM", "STSM"):
            assert any(n.startswith(opcode) for n in db.forms), opcode

    def test_the_uniform_datapath_is_present(self, db):
        """Uniform-register instructions are easy to miss entirely."""
        assert any(n.startswith("U") for n in db.forms)

    def test_an_operand_the_mnemonic_depends_on_still_has_a_field(self, db):
        """`IMAD.SHL.U32` is what `IMAD` is called when its multiplier is a shift.

        Any flip of that immediate makes the value something other than a power
        of two, which the disassembler prints as `IMAD.U32`, so a differential
        probe reads all thirty-two bits as suffix bits and the operand ends up
        with no field at all. Every one of them was refused by the assembler
        until the probe learned to check whether writing values through those
        bits reproduces them.
        """
        forms = [f for f in db.shapes("IMAD.SHL.U32") if f.operand_text]
        assert forms, "IMAD.SHL.U32 is not in the database"
        for form in forms:
            immediate = [o for o in form.operands if len(o.bits) >= 16]
            assert immediate, (
                f"IMAD.SHL.U32 {form.operand_text} has no wide operand field; "
                "the rendered-operand recovery is not running"
            )


class TestSerialisation:
    def test_reported_coverage_matches_the_contents(self, db):
        raw = json.loads(DB_PATH.read_text())
        assert raw["coverage"]["forms"] == len(db.forms)
        assert raw["coverage"]["opcodes"] == len(db.opcodes)

    def test_round_trips_through_disk(self, db, tmp_path):
        out = tmp_path / "db.json"
        db.write(out)
        again = IsaDatabase.read(out)
        assert set(again.forms) == set(db.forms)
        assert all(again.forms[n].encoding == db.forms[n].encoding for n in db.forms)

    def test_a_future_schema_is_refused(self, tmp_path):
        """A silently misread database is worse than one that will not load."""
        raw = json.loads(DB_PATH.read_text())
        raw["schema_version"] = 999
        path = tmp_path / "future.json"
        path.write_text(json.dumps(raw))
        with pytest.raises(ValueError, match="schema"):
            IsaDatabase.read(path)
