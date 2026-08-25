# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Architecture-field mining reports only visible, one-bit effects."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from basalt.disasm import Instruction
from basalt.encoding import Word


def _miner_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "mine_arch_fields.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("mine_arch_fields", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _verifier_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "verify_arch_fields.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("verify_arch_fields", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _instruction(text: str) -> Instruction:
    return Instruction(0, text, Word(0))


def test_top_level_operand_split_preserves_composites():
    miner = _miner_module()
    assert miner._split_operands("R0, desc[UR4][R2.64+0x8], {R4, R5}") == (
        "R0",
        "desc[UR4][R2.64+0x8]",
        "{R4, R5}",
    )


def test_visible_change_classes_do_not_infer_control_meaning():
    miner = _miner_module()
    base = _instruction("IADD3 R0, R1, R2, RZ")
    assert miner._classify_change(base, _instruction("IADD3 R0, R1, R2, RZ"))[0] == "text_silent"
    assert miner._classify_change(base, None)[0] == "decode_rejected"
    assert miner._classify_change(base, _instruction("@P0 IADD3 R0, R1, R2, RZ"))[0] == "predicate"
    assert miner._classify_change(base, _instruction("IMAD R0, R1, R2, RZ"))[0] == "opcode"
    assert miner._classify_change(base, _instruction("IADD3.X R0, R1, R2, RZ"))[0] == "modifier"
    kind, detail = miner._classify_change(base, _instruction("IADD3 R0, R1, R3, RZ"))
    assert kind == "operand"
    assert detail.startswith("operand[2]")


def test_contiguous_regions_are_mechanical_not_named_fields():
    miner = _miner_module()
    assert miner._contiguous([9, 7, 8, 20, 22, 21]) == [[7, 8, 9], [20, 21, 22]]


def test_verifier_requires_an_exact_one_bit_witness():
    verifier = _verifier_module()
    assert verifier._one_bit_witness(
        {"bit": 12, "original_encoding": "0" * 32, "mutated_encoding": f"{1 << 12:032x}"}
    )
    assert not verifier._one_bit_witness(
        {"bit": 12, "original_encoding": "0" * 32, "mutated_encoding": f"{3 << 12:032x}"}
    )
