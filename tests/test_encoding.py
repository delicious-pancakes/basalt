# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""The 128-bit word and its control section."""

from __future__ import annotations

import pytest

from basalt.encoding import (
    CONTROL_FIELDS,
    NO_BARRIER,
    WORD_BITS,
    WORD_BYTES,
    BitField,
    Word,
    bit_diff,
    popcount,
)

# Real instructions harvested from ptxas output for sm_120a, kept as literals so
# these tests need no toolchain. Each is (lo, hi, text) as nvdisasm prints them.
S2R_TID = (0x0000000000057919, 0x000E2E0000002100)
IMAD_WIDE = (0x0000000405027825, 0x041FE200078E0002)
IADD_IMM = (0x0000002A05057835, 0x000FCA00078E0000)
STG_E = (0x0000000502007986, 0x002FE2000C101904)
EXIT = (0x000000000000794D, 0x000FEA0003800000)


class TestWord:
    def test_halves_round_trip(self):
        lo, hi = IADD_IMM
        w = Word.from_halves(lo, hi)
        assert w.lo == lo
        assert w.hi == hi

    def test_bytes_round_trip(self):
        w = Word.from_halves(*IMAD_WIDE)
        assert Word.from_bytes(w.to_bytes()) == w
        assert len(w.to_bytes()) == WORD_BYTES

    def test_str_is_high_half_first(self):
        lo, hi = IADD_IMM
        assert str(Word.from_halves(lo, hi)) == f"{hi:016x}{lo:016x}"

    def test_rejects_oversized_value(self):
        with pytest.raises(ValueError):
            Word(1 << WORD_BITS)

    def test_rejects_wrong_length_bytes(self):
        with pytest.raises(ValueError):
            Word.from_bytes(b"\x00" * 15)


class TestControlFields:
    def test_fields_do_not_overlap(self):
        seen: set[int] = set()
        for f in CONTROL_FIELDS:
            bits = set(range(f.lo, f.lo + f.width))
            assert not (bits & seen), f"{f.name} overlaps another control field"
            seen |= bits

    def test_fields_are_inside_the_word(self):
        for f in CONTROL_FIELDS:
            assert f.lo >= 0 and f.hi < WORD_BITS

    def test_producer_consumer_pair_agrees(self):
        """S2R signals scoreboard 0; the IMAD that consumes it waits on 0x01.

        This is the observation that pinned the control layout in the first
        place, so it is worth holding as a test rather than a note.
        """
        s2r = Word.from_halves(*S2R_TID)
        imad = Word.from_halves(*IMAD_WIDE)
        assert s2r.field("write_barrier") == 0
        assert imad.field("wait_mask") == 0x01

    def test_second_producer_consumer_pair_agrees(self):
        """STG waits on 0x02, the scoreboard the uniform load signalled."""
        assert Word.from_halves(*STG_E).field("wait_mask") == 0x02

    def test_reuse_flag_matches_the_printed_annotation(self):
        """nvdisasm prints `R5.reuse` on this IMAD, so a reuse bit must be set."""
        assert Word.from_halves(*IMAD_WIDE).field("reuse") == 0x1

    def test_no_barrier_sentinel(self):
        """An instruction signalling nothing uses 7 in both barrier fields."""
        exit_word = Word.from_halves(*EXIT)
        assert exit_word.field("write_barrier") == NO_BARRIER
        assert exit_word.field("read_barrier") == NO_BARRIER

    def test_stall_is_readable(self):
        assert Word.from_halves(*IADD_IMM).field("stall") == 5

    def test_with_field_changes_only_that_field(self):
        w = Word.from_halves(*IADD_IMM)
        patched = w.with_field("stall", 1)
        assert patched.field("stall") == 1
        for name, value in w.control.items():
            if name != "stall":
                assert patched.field(name) == value
        assert patched.payload == w.payload

    def test_with_field_rejects_overflow(self):
        w = Word.from_halves(*IADD_IMM)
        with pytest.raises(ValueError):
            w.with_field("stall", 16)  # four bits hold 0..15

    def test_payload_masks_every_control_field(self):
        w = Word.from_halves(*IMAD_WIDE)
        for f in CONTROL_FIELDS:
            assert (w.payload >> f.lo) & ((1 << f.width) - 1) == 0

    def test_payload_is_stable_across_scheduling(self):
        """Two schedulings of one instruction share a payload."""
        w = Word.from_halves(*IADD_IMM)
        assert w.with_field("stall", 2).payload == w.with_field("stall", 9).payload


class TestBitField:
    def test_get_and_set_round_trip(self):
        f = BitField("demo", lo=8, width=4)
        assert f.get(f.set(0, 0xA)) == 0xA

    def test_set_rejects_overflow(self):
        with pytest.raises(ValueError):
            BitField("demo", lo=8, width=4).set(0, 0x10)

    def test_hi_and_mask(self):
        f = BitField("demo", lo=8, width=4)
        assert f.hi == 11
        assert f.mask == 0xF00


class TestHelpers:
    def test_popcount(self):
        assert popcount(0b1011) == 3

    def test_bit_diff_finds_exactly_the_flipped_bits(self):
        a = 0
        b = (1 << 3) | (1 << 40) | (1 << 100)
        assert bit_diff(a, b) == [3, 40, 100]

    def test_bit_diff_is_empty_for_identical_words(self):
        assert bit_diff(12345, 12345) == []
