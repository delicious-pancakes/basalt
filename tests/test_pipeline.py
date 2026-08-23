# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""End-to-end tests against the real vendor toolchain.

These are the tests that decide whether basalt is trustworthy, and there are two
of them that matter more than the rest:

*The positive control.* The vendor compiler's own scheduling must verify clean.
If basalt flags `ptxas` output, basalt is wrong, and every finding it produces
about anyone else's code is worthless until that is fixed.

*The negative control.* Take that same clean output, shorten one stall count,
and basalt must flag exactly that instruction and nothing else. Without this, a
checker that reports "clean" unconditionally would pass the positive control.

Marked `toolchain` because they need ptxas and nvdisasm. None of them need a GPU.
"""

from __future__ import annotations

import pytest

from basalt.asm.cubin import Cubin
from basalt.disasm import decode_words, disassemble_cubin, raw_arch
from basalt.isa.build import collect_representatives
from basalt.probe.fields import BitRole, probe_word
from basalt.verify.hazards import HazardKind, split_blocks, verify_program
from basalt.verify.latency import DEFAULT_MODEL, LatencyClass
from basalt.verify.operands import operand_access

pytestmark = pytest.mark.toolchain

ARCH = "sm_120a"
RAW = raw_arch(ARCH)


class TestOracles:
    def test_cubin_oracle_produces_encodings(self, toolchain, sample_cubin):
        insns = disassemble_cubin(toolchain, sample_cubin)
        assert insns
        assert all(i.word is not None for i in insns)

    def test_probe_oracle_round_trips_compiler_output(self, toolchain, sample_cubin):
        """Every encoding ptxas emitted must decode back to the same mnemonic."""
        insns = disassemble_cubin(toolchain, sample_cubin)
        back = decode_words(toolchain, [i.word for i in insns], arch=RAW)
        assert len(back) == len(insns)
        for original, decoded in zip(insns, back, strict=True):
            assert decoded is not None
            assert decoded.mnemonic == original.mnemonic

    def test_probe_oracle_reports_undecodable_words_as_none(self, toolchain):
        """An illegal word is a measurement, so it must not abort the batch."""
        from basalt.encoding import Word

        good = Word.from_halves(0x000000000000794D, 0x000FEA0003800000)  # EXIT
        bad = Word((1 << 128) - 1)
        got = decode_words(toolchain, [good, bad, good], arch=RAW)
        assert len(got) == 3
        assert got[0] is not None and got[2] is not None
        assert got[1] is None


class TestCubinRoundTrip:
    def test_elf_words_match_the_disassembler(self, toolchain, sample_cubin):
        """Independent evidence that the ELF parsing is right."""
        from_elf = Cubin.load(sample_cubin).words()
        from_disasm = [i.word for i in disassemble_cubin(toolchain, sample_cubin)]
        assert len(from_elf) == len(from_disasm)
        assert all(a == b for a, b in zip(from_elf, from_disasm, strict=True))

    def test_writing_a_word_back_unchanged_is_byte_identical(self, sample_cubin, tmp_path):
        cb = Cubin.load(sample_cubin)
        original = cb.data
        cb.write_word(0, cb.read_word(0))
        assert cb.data == original

    def test_patching_changes_only_the_target_instruction(self, sample_cubin):
        cb = Cubin.load(sample_cubin)
        before = cb.words()
        cb.patch_control(1, "stall", 3)
        after = cb.words()
        differing = [i for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b]
        assert differing == [1]
        assert after[1].field("stall") == 3
        assert after[1].payload == before[1].payload


class TestProbe:
    def test_register_field_is_located_by_mutation(self, toolchain, sample_cubin):
        """Flipping a destination bit must change the destination register."""
        insns = disassemble_cubin(toolchain, sample_cubin)
        target = next(
            i
            for i in insns
            if i.opcode == "IMAD" and operand_access(i.mnemonic, i.operands).real_defs
        )
        fmap = probe_word(toolchain, target.word, arch=RAW)
        assert fmap is not None
        slots = fmap.operand_fields()
        assert slots, "no operand slot was attributed to any bit"
        # the destination is the first printed operand and is register-width
        assert len(slots[0]) >= 4

    def test_control_bits_are_never_attributed_to_operands(self, toolchain, sample_cubin):
        insns = disassemble_cubin(toolchain, sample_cubin)
        fmap = probe_word(toolchain, insns[0].word, arch=RAW)
        assert fmap is not None
        control_bits = set(fmap.bits_with_role(BitRole.CONTROL))
        operand_bits = {b for bits in fmap.operand_fields().values() for b in bits}
        assert not (control_bits & operand_bits)


class TestVerifierControls:
    def test_positive_control_vendor_output_is_clean(self, toolchain, sample_cubin):
        """If this fails, basalt is wrong and nothing else it says can be trusted."""
        insns = disassemble_cubin(toolchain, sample_cubin)
        report = verify_program(insns, DEFAULT_MODEL)
        assert report.checked_pairs > 0, "verified nothing, so proved nothing"
        assert not report.hazards, "\n".join(h.describe() for h in report.hazards)

    def test_negative_control_a_shortened_stall_is_caught(self, toolchain, sample_cubin, tmp_path):
        """The test that separates a checker from a no-op."""
        insns = disassemble_cubin(toolchain, sample_cubin)

        pair = _first_covered_fixed_dependency(insns)
        if pair is None:
            pytest.skip("sample has no fixed-latency dependency covered by stalls")
        def_index, needed = pair

        cb = Cubin.load(sample_cubin)
        # 1, not 0: a zero stall is the safe encoding on this architecture, so
        # injecting it would make the instruction safer rather than broken
        cb.patch_control(def_index, "stall", 1)
        patched = tmp_path / "patched.cubin"
        cb.save(patched)

        report = verify_program(disassemble_cubin(toolchain, patched), DEFAULT_MODEL)
        understalled = [
            h
            for h in report.hazards
            if h.kind is HazardKind.UNDERSTALLED and h.def_index == def_index
        ]
        assert understalled, "the injected hazard was not detected"
        assert understalled[0].required == needed
        assert understalled[0].actual < needed

    def test_negative_control_does_not_flag_unrelated_instructions(
        self, toolchain, sample_cubin, tmp_path
    ):
        """Detection is worth little if it comes with collateral false alarms."""
        insns = disassemble_cubin(toolchain, sample_cubin)
        pair = _first_covered_fixed_dependency(insns)
        if pair is None:
            pytest.skip("sample has no fixed-latency dependency covered by stalls")
        def_index, _ = pair

        cb = Cubin.load(sample_cubin)
        cb.patch_control(def_index, "stall", 1)
        patched = tmp_path / "patched.cubin"
        cb.save(patched)

        report = verify_program(disassemble_cubin(toolchain, patched), DEFAULT_MODEL)
        collateral = [h for h in report.hazards if h.def_index != def_index]
        assert not collateral, "\n".join(h.describe() for h in collateral)


class TestHarvest:
    def test_representative_selection_avoids_aliased_operands(self, toolchain, sample_cubin):
        """`IADD R5, R5, x` cannot separate the destination field from the source."""
        from basalt.harvest.runner import HarvestResult, Observation

        result = HarvestResult(cuda_version="test", arch=ARCH, generated_utc="now")
        for text, enc in (
            ("R5, R5, 0x1", "a" * 32),
            ("R2, R3, 0x1", "b" * 32),
        ):
            result.observations.append(
                Observation(
                    mnemonic="IADD",
                    opcode="IADD",
                    modifiers=(),
                    operands=text,
                    encoding=enc,
                    payload=enc,
                    control={},
                    source_kernel="k",
                    source_label="add.s32",
                    source_family="binary",
                    opt_level=3,
                )
            )
        chosen = collect_representatives(result)
        # keyed by mnemonic and shape, so these two compete and the one with
        # distinct registers wins
        from basalt.isa.build import operand_shape

        key = ("IADD", operand_shape("R2, R3, 0x1"))
        assert chosen[key].operands == "R2, R3, 0x1"

    def test_different_operand_shapes_are_kept_apart(self, toolchain):
        """`IADD R2, R3, 0x1` and `IADD R2, R3, R4` are different encodings."""
        from basalt.harvest.runner import HarvestResult, Observation

        result = HarvestResult(cuda_version="test", arch=ARCH, generated_utc="now")
        for text, enc in (("R2, R3, 0x1", "a" * 32), ("R2, R3, R4", "b" * 32)):
            result.observations.append(
                Observation(
                    mnemonic="IADD",
                    opcode="IADD",
                    modifiers=(),
                    operands=text,
                    encoding=enc,
                    payload=enc,
                    control={},
                    source_kernel="k",
                    source_label="add.s32",
                    source_family="binary",
                    opt_level=3,
                )
            )
        chosen = collect_representatives(result)
        assert len(chosen) == 2, "the two shapes were collapsed into one form"
        assert {o.operands for o in chosen.values()} == {"R2, R3, 0x1", "R2, R3, R4"}

    def test_the_guard_is_not_part_of_the_shape(self):
        """`@P0 IADD` and `IADD` are one encoding with a field set differently."""
        from basalt.isa.build import operand_shape

        assert operand_shape("@P0 R2, R3, R4") == operand_shape("R2, R3, R4")


def _first_covered_fixed_dependency(insns) -> tuple[int, int] | None:
    """Find a def/use pair whose latency the schedule currently covers.

    Returns the program-wide index of the definition and the latency it needs,
    which is what the negative control shortens. Restricted to a single block so
    the pair is unambiguous regardless of what the control-flow graph does.
    """
    for block in split_blocks(insns):
        for i in range(block.start, block.end):
            producer = insns[i]
            record = DEFAULT_MODEL.lookup(producer.opcode)
            if record.kind is not LatencyClass.FIXED or record.cycles == 0:
                continue
            produced = operand_access(producer.mnemonic, producer.operands).real_defs
            if not produced:
                continue
            for j in range(i + 1, block.end):
                consumer = insns[j]
                if produced & operand_access(consumer.mnemonic, consumer.operands).real_uses:
                    if producer.word.field("stall") >= record.cycles:
                        return i, record.cycles
                    break
    return None
