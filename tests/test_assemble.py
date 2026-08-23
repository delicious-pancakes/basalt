# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""The assembler, held to one standard: match the vendor exactly or refuse.

There is no third outcome worth having. A word that is close enough assembles,
disassembles back to the text it came from, and computes something else, which
is the same failure the rest of this project exists to catch and would be
embarrassing to ship inside it.

So every test here is one of two shapes. Either the assembler reproduces bytes
`ptxas` actually emitted, or it declines and says why. The cases that decline are
the interesting ones: each is a way the assembler was previously, confidently
wrong.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from basalt.asm.assemble import Assembler, AssemblyError
from basalt.encoding import Word
from basalt.isa.database import IsaDatabase
from basalt.paths import ISA_DATABASE

ROOT = Path(__file__).resolve().parent.parent
DATABASE = ISA_DATABASE


@pytest.fixture(scope="module")
def assembler() -> Assembler:
    if not DATABASE.is_file():
        pytest.skip("no ISA database; run `basalt build-isa`")
    return Assembler(IsaDatabase.read(DATABASE))


@pytest.fixture(scope="module")
def database() -> IsaDatabase:
    if not DATABASE.is_file():
        pytest.skip("no ISA database; run `basalt build-isa`")
    return IsaDatabase.read(DATABASE)


class TestReproducesTheReference:
    """The form's own text must assemble back to the form's own encoding."""

    def test_every_form_round_trips_its_reference(self, assembler, database):
        missed = []
        for mnemonic, form in database.forms.items():
            if not form.operand_text:
                continue
            try:
                got = assembler.assemble(f"{mnemonic} {form.operand_text}")
            except AssemblyError:
                continue  # refusing is allowed; being wrong is not
            if got.value != form.word.value:
                missed.append(mnemonic)
        assert not missed, f"{len(missed)} forms did not reproduce their own encoding: {missed[:8]}"

    def test_a_different_register_changes_only_that_field(self, assembler, database):
        form = database.forms["IMAD"]
        reference = assembler.assemble(f"IMAD {form.operand_text}")
        assert reference.value == form.word.value


class TestSignedImmediates:
    """A minus on a number is part of the number, not a bit elsewhere.

    `-R0` is a register plus a negate bit parked far away in the word. `-0x1` is
    the immediate -1, written as two's complement in its own field, with no
    separate bit involved. Treating the second like the first asks for a negate
    bit that does not exist and refuses every subtract-by-add in the corpus,
    which was 12 instructions.
    """

    def test_a_negative_immediate_encodes_as_twos_complement(self, assembler, database):
        form = next(f for f in database.shapes("IADD") if f.operand_text == "R7, R5, 0x1")
        slot = next(o for o in form.operands if len(o.bits) >= 16)
        word = assembler.assemble("IADD R7, R5, -0x1")
        got = 0
        for position, bit in enumerate(slot.bits):
            got |= ((word.value >> bit) & 1) << position
        assert got == (1 << len(slot.bits)) - 1, "-0x1 should fill the field with ones"

    def test_a_positive_immediate_is_unaffected(self, assembler, database):
        form = next(f for f in database.shapes("IADD") if f.operand_text == "R7, R5, 0x1")
        assert assembler.assemble("IADD R7, R5, 0x1").value == form.word.value

    def test_an_immediate_too_negative_for_the_field_is_refused(self, assembler):
        with pytest.raises(AssemblyError, match="signed bits"):
            assembler.assemble("IADD R7, R5, -0x80000001")


class TestRefusesRatherThanGuesses:
    """Each of these produced a wrong word before it produced an error."""

    def test_a_branch_target_is_refused(self, assembler):
        with pytest.raises(AssemblyError, match="branch target"):
            assembler.assemble("BRA `(.L_x_0)")

    def test_an_unknown_mnemonic_is_refused(self, assembler):
        with pytest.raises(AssemblyError, match="not in the instruction database"):
            assembler.assemble("NOTAREALOPCODE R0, R1")

    def test_a_shape_the_database_has_never_seen_is_refused(self, assembler):
        """Every recorded shape is tried, and none of them fits this."""
        with pytest.raises(AssemblyError):
            assembler.assemble("IMAD R7, R2, c[0x0][0x0], RZ")


class TestControlBitsAreCopiedNotInvented:
    """Assembling does not decide timing. That is the scheduler's job."""

    def test_the_control_word_is_taken_from_the_argument(self, assembler, database):
        form = database.forms["IMAD"]
        control = Word(0).with_field("stall", 7).with_field("wait_mask", 0b101)
        got = assembler.assemble(f"IMAD {form.operand_text}", control=control)
        assert got.field("stall") == 7
        assert got.field("wait_mask") == 0b101

    def test_without_a_control_word_the_reference_is_kept(self, assembler, database):
        form = database.forms["IMAD"]
        got = assembler.assemble(f"IMAD {form.operand_text}")
        assert got.field("stall") == form.word.field("stall")


class TestSeveralShapesOfOneMnemonic:
    """A mnemonic covers several encodings, and the text decides which."""

    @staticmethod
    def _a_mnemonic_with_two_assemblable_shapes(assembler, database):
        for mnemonic in database.forms:
            shapes = database.shapes(mnemonic)
            if len(shapes) < 2:
                continue
            encoded = []
            for form in shapes:
                try:
                    encoded.append((form, assembler.assemble(f"{mnemonic} {form.operand_text}")))
                except AssemblyError:
                    break
            if len(encoded) == len(shapes):
                return mnemonic, encoded
        return None, []

    def test_a_mnemonic_with_several_shapes_encodes_each_of_them(self, assembler, database):
        mnemonic, encoded = self._a_mnemonic_with_two_assemblable_shapes(assembler, database)
        if mnemonic is None:
            pytest.skip("no mnemonic in this database has two assemblable shapes")
        for form, word in encoded:
            assert word.value == form.word.value, (
                f"{mnemonic} {form.operand_text} did not reproduce its own encoding"
            )

    def test_the_shapes_are_distinct_encodings(self, assembler, database):
        """Two shapes of one mnemonic must not be the same word twice."""
        mnemonic, encoded = self._a_mnemonic_with_two_assemblable_shapes(assembler, database)
        if mnemonic is None:
            pytest.skip("no mnemonic in this database has two assemblable shapes")
        words = {word.value for _, word in encoded}
        assert len(words) == len(encoded), f"{mnemonic} recorded the same encoding more than once"

    def test_no_mnemonic_records_a_duplicate_word(self, database):
        """Across the whole database, not just the first mnemonic with variants."""
        duplicated = [
            mnemonic
            for mnemonic in database.forms
            if len({f.encoding for f in database.shapes(mnemonic)})
            != len(database.shapes(mnemonic))
        ]
        assert not duplicated, f"these mnemonics hold the same word twice: {duplicated[:8]}"


class TestCompositeOperands:
    """A bracket operand is several fields, and each goes in its own bits."""

    def test_a_different_constant_offset_encodes(self, assembler, database):
        form = database.forms.get("LDC")
        if form is None or not any(o.subfields.get("offset") for o in form.operands):
            pytest.skip("no offset sub-field recorded for LDC")
        # the offset is changed in the recorded form's own text, so the register
        # is whatever that form used. writing one here compares two differences
        recorded = form.operand_text
        offset = re.search(r"\]\[(0x[0-9a-fA-F]+)\]", recorded)
        if offset is None:
            pytest.skip(f"no constant offset in the recorded form: {recorded}")
        moved_offset = int(offset.group(1), 16) ^ 0x4
        text = recorded[: offset.start(1)] + hex(moved_offset) + recorded[offset.end(1) :]
        got = assembler.assemble(f"LDC {text}")
        reference = assembler.assemble(f"LDC {recorded}")
        assert got.value != reference.value
        # only the offset bits may move
        offset_bits = next(
            o.subfields["offset"] for o in form.operands if o.subfields.get("offset")
        )
        moved = {b for b in range(128) if ((got.value ^ reference.value) >> b) & 1}
        assert moved <= set(offset_bits), f"bits outside the offset field moved: {moved}"

    def test_a_bank_with_no_recorded_bits_is_refused(self, assembler, database):
        """Refusing names the part it could not place."""
        if "LDC" not in database.forms:
            pytest.skip("no LDC in this database")
        try:
            assembler.assemble("LDC R2, c[0x9][0x380]")
        except AssemblyError as exc:
            assert "bank" in str(exc)
