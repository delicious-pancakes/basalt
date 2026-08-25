# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Architecture authority is checked before any control model is used."""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from basalt.architecture import (
    ArchitectureError,
    architecture_identity,
    artifact_architecture,
    cubin_architecture,
    require_architecture_match,
    require_control_model,
)
from basalt.cli import main


def _prototype_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "prototype_arch.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("prototype_arch", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_architecture_spellings_normalise_without_merging_families():
    assert architecture_identity("SM120a").canonical == "sm_120a"
    assert architecture_identity("sm_86").family == "sm_86"
    assert architecture_identity("SM121f").family == "sm_121"


@pytest.mark.parametrize("arch", ["sm_120", "SM120a", "sm120f"])
def test_measured_control_family_is_admitted(arch):
    assert require_control_model(arch, "verify").family == "sm_120"


@pytest.mark.parametrize("arch", ["sm_86", "sm_100", "sm_121"])
def test_unmeasured_control_family_is_refused(arch):
    with pytest.raises(ArchitectureError, match="no measured control model"):
        require_control_model(arch, "verify")


def test_artifact_architecture_is_mandatory_and_family_checked(tmp_path):
    measured = tmp_path / "latency.json"
    measured.write_text(json.dumps({"arch": "sm_86"}))
    assert artifact_architecture(measured) == "sm_86"
    with pytest.raises(ArchitectureError, match="does not match"):
        require_architecture_match("sm_120a", artifact_architecture(measured), "latency")

    missing = tmp_path / "missing.json"
    missing.write_text("{}")
    with pytest.raises(ArchitectureError, match="no architecture identity"):
        artifact_architecture(missing)


def _elf_cubin(flags: int) -> bytes:
    header = bytearray(0x34)
    header[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<I", header, 0x30, flags)
    return bytes(header)


def test_cubin_architecture_accepts_both_cuda_flag_eras(tmp_path):
    old = tmp_path / "old.cubin"
    old.write_bytes(_elf_cubin(0x00560556))
    assert cubin_architecture(old) == "sm_86"

    current = tmp_path / "current.cubin"
    current.write_bytes(_elf_cubin(0x06007802))
    assert cubin_architecture(current) == "sm_120"


def test_cli_refuses_sm86_control_work_before_opening_target(capsys):
    assert main(["--arch", "sm_86", "verify", "not-present.cubin"]) == 2
    assert "architecture authority failure" in capsys.readouterr().err


def test_raw_prototype_retargets_without_interpreting_control_fields():
    module = _prototype_module()
    source = ".version 9.0\n.target sm_120a\n.address_size 64\n"
    assert ".target sm_86" in module._retarget_ptx(source, "sm_86")
    row = module.RawBuild(
        kernel="k",
        label="add.s32",
        family="binary",
        opt_level=3,
        status="built",
        ptx_path="k.ptx",
        ptx_sha256="a",
        instructions=(module.RawInstruction(0, "IADD3 R0, R1, R2, RZ", "00" * 16),),
    )
    encoded = asdict(row)
    assert "payload" not in encoded["instructions"][0]
    assert "control" not in encoded["instructions"][0]
    predicated = module.RawBuild(
        kernel="p",
        label="add.s32",
        family="binary",
        opt_level=3,
        status="built",
        ptx_path="p.ptx",
        ptx_sha256="b",
        instructions=(module.RawInstruction(0, "@P0 IADD3 R0, R1, R2, RZ", "11" * 16),),
    )
    assert module._summary([row, predicated])["distinct_mnemonics"] == 1
