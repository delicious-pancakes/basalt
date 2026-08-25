# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""The architecture-bit runner keeps observation separate from attribution."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from basalt.gpu.driver import Device


def _probe_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "probe_arch_bits.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("probe_arch_bits", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _plan():
    return {
        "schema": "basalt.architecture-bit-execution-plan.v1",
        "arch": "sm_86",
        "repetitions": 3,
        "worker_timeout_seconds": 10,
        "cases": [
            {
                "id": "alu-bit116",
                "bit": 116,
                "expected_relation": "equivalent",
                "invocations": [
                    {
                        "id": "positive",
                        "input_hex": "0100000002000000",
                        "output_initial_hex": "cccccccc",
                        "output_bytes": 4,
                    }
                ],
            }
        ],
    }


def _worker(status: str, outputs: list[str] | None = None):
    invocations = []
    if outputs is not None:
        invocations = [
            {
                "id": "positive",
                "samples": [
                    {"repeat": index, "output_hex": output} for index, output in enumerate(outputs)
                ],
            }
        ]
    return {"status": status, "invocations": invocations}


def test_plan_validation_rejects_duplicate_cases_and_output_size_mismatch():
    probe = _probe_module()
    plan = _plan()
    probe._validate_plan(plan)

    duplicate = _plan()
    duplicate["cases"].append(dict(duplicate["cases"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        probe._validate_plan(duplicate)

    wrong_size = _plan()
    wrong_size["cases"][0]["invocations"][0]["output_bytes"] = 8
    with pytest.raises(ValueError, match="size mismatch"):
        probe._validate_plan(wrong_size)


def test_equivalence_requires_deterministic_completed_outputs():
    probe = _probe_module()
    prepared = {"expected_relation": "equivalent"}
    same = _worker("completed", ["04000000"] * 3)
    assert probe._evaluate(prepared, same, same)["passed"]

    noisy = _worker("completed", ["04000000", "05000000", "04000000"])
    assert not probe._evaluate(prepared, same, noisy)["passed"]
    assert not probe._evaluate(prepared, same, _worker("failed"))["passed"]


def test_detection_control_accepts_divergence_or_a_rejected_mutant():
    probe = _probe_module()
    prepared = {"expected_relation": "divergent_or_rejected"}
    original = _worker("completed", ["04000000"] * 3)
    different = _worker("completed", ["cccccccc"] * 3)
    assert probe._evaluate(prepared, original, different)["passed"]
    assert probe._evaluate(prepared, original, _worker("failed"))["passed"]
    assert not probe._evaluate(prepared, original, original)["passed"]


def test_device_launch_preserves_legacy_waiting_contract():
    device = object.__new__(Device)
    device.launch_async = Mock()
    device.synchronize = Mock()
    function = object()
    params = []

    device.launch(function, params, grid=(2, 1, 1), block=(32, 1, 1), shared_bytes=64)

    device.launch_async.assert_called_once_with(
        function,
        params,
        grid=(2, 1, 1),
        block=(32, 1, 1),
        shared_bytes=64,
    )
    device.synchronize.assert_called_once_with()
