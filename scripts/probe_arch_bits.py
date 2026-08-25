#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Run an identity-bound, architecture-local one-bit execution campaign.

This is deliberately an experimental runner, not an architecture profile.  It
patches one exact instruction word in an existing cubin, decodes the original
and mutant as raw words, then executes each variant in a separate process.  A
CUDA failure in one mutant therefore cannot erase the remaining first-attempt
observations or poison their contexts.

The campaign plan names every source cubin by SHA-256, every target instruction
by index/text/encoding, the exact GPU UUID/PCI identity, and an accepted runtime
manifest.  The result separates raw-decoder visibility, module load, function
lookup, launch, synchronization, output bytes, and duration.  Output equality
is evidence only for the listed inputs on the listed device; it never identifies
a scheduling field or proves a production control model.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import _repo

_repo.use_repo_source()

import basalt.asm.cubin  # noqa: E402
import basalt.disasm  # noqa: E402
import basalt.gpu.driver  # noqa: E402
import basalt.toolchain  # noqa: E402
from basalt.architecture import architecture_identity, cubin_architecture  # noqa: E402
from basalt.asm.cubin import Cubin  # noqa: E402
from basalt.disasm import decode_word, disassemble_cubin, raw_arch  # noqa: E402
from basalt.encoding import Word  # noqa: E402
from basalt.gpu.driver import CudaError, Device  # noqa: E402
from basalt.toolchain import Toolchain, find_toolchain  # noqa: E402


PLAN_SCHEMA = "basalt.architecture-bit-execution-plan.v1"
MANIFEST_SCHEMA = "basalt.runtime-manifest.v1"
RESULT_SCHEMA = "basalt.architecture-bit-execution-result.v1"
WORKER_SCHEMA = "basalt.architecture-bit-worker.v1"
RELATIONS = {"equivalent", "divergent_or_rejected", "observational"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"runtime object is not a regular file: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _cuda_library() -> Path:
    candidates = (
        Path("/usr/lib/x86_64-linux-gnu/libcuda.so.1"),
        Path("/usr/lib64/libcuda.so.1"),
    )
    for path in candidates:
        if path.exists():
            return path.resolve(strict=True)
    raise ValueError("could not resolve the load-bearing CUDA driver library")


def _runtime_paths(tc: Toolchain) -> dict[str, Path]:
    return {
        "python": Path(sys.executable),
        "nvidia_smi": Path("/usr/bin/nvidia-smi"),
        "ptxas": tc.ptxas,
        "nvdisasm": tc.nvdisasm,
        "probe_arch_bits": Path(__file__),
        "architecture_module": Path(__import__("basalt.architecture", fromlist=["x"]).__file__),
        "cubin_module": Path(basalt.asm.cubin.__file__),
        "disasm_module": Path(basalt.disasm.__file__),
        "driver_module": Path(basalt.gpu.driver.__file__),
        "encoding_module": Path(__import__("basalt.encoding", fromlist=["x"]).__file__),
        "toolchain_module": Path(basalt.toolchain.__file__),
        "cuda_driver_library": _cuda_library(),
    }


def runtime_manifest(tc: Toolchain) -> dict[str, object]:
    return {
        "schema": MANIFEST_SCHEMA,
        "toolchain_version": tc.version,
        "objects": {name: _identity(path) for name, path in _runtime_paths(tc).items()},
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def _device_rows() -> list[dict[str, object]]:
    command = [
        "/usr/bin/nvidia-smi",
        "--query-gpu=index,uuid,pci.bus_id,name,compute_cap,driver_version",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=10)
    if result.returncode != 0:
        raise ValueError(f"nvidia-smi failed: {result.stderr.strip()}")
    rows = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            raise ValueError(f"unexpected nvidia-smi row: {line!r}")
        rows.append(
            {
                "ordinal": int(parts[0]),
                "uuid": parts[1],
                "pci_bus_id": parts[2].lower(),
                "name": parts[3],
                "compute_capability": parts[4],
                "driver_version": parts[5],
            }
        )
    return rows


def _require_device(expected: dict[str, Any]) -> dict[str, object]:
    ordinal = int(expected["ordinal"])
    rows = [row for row in _device_rows() if row["ordinal"] == ordinal]
    if len(rows) != 1:
        raise ValueError(f"device ordinal {ordinal} did not resolve exactly once")
    observed = rows[0]
    for key in ("uuid", "pci_bus_id", "name", "compute_capability"):
        wanted = str(expected[key]).lower() if key == "pci_bus_id" else expected[key]
        if observed[key] != wanted:
            raise ValueError(
                f"device identity mismatch for {key}: expected {wanted!r}, observed {observed[key]!r}"
            )
    return observed


def _require_lease(plan: dict[str, Any]) -> dict[str, Any]:
    expected = plan["lease"]
    path = Path(expected["state_path"]).resolve(strict=True)
    state = _load_json(path)
    checks = {
        "status": "held",
        "pool": expected["pool"],
        "owner_project": expected["required_owner_project"],
        "owner_lane": expected["required_owner_lane"],
        "executor_lane": expected["required_executor_lane"],
        "scope": expected["scope"],
    }
    for key, wanted in checks.items():
        if state.get(key) != wanted:
            raise ValueError(
                f"lease authority mismatch for {key}: expected {wanted!r}, "
                f"observed {state.get(key)!r}"
            )
    if not state.get("lease_id"):
        raise ValueError("held lease has no lease_id")
    lease_device = state.get("device", {})
    expected_device = plan["device"]
    for key in ("uuid", "pci_bus_id", "name"):
        wanted = str(expected_device[key]).lower() if key == "pci_bus_id" else expected_device[key]
        observed = (
            str(lease_device.get(key, "")).lower()
            if key == "pci_bus_id"
            else lease_device.get(key)
        )
        if observed != wanted:
            raise ValueError(f"lease device mismatch for {key}: {observed!r} != {wanted!r}")
    return state


def _validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"plan schema is not {PLAN_SCHEMA}")
    if architecture_identity(str(plan.get("arch", ""))).canonical != "sm_86":
        raise ValueError("this campaign is bounded to sm_86")
    repetitions = plan.get("repetitions")
    if not isinstance(repetitions, int) or not 1 <= repetitions <= 20:
        raise ValueError("repetitions must be an integer from 1 through 20")
    timeout = plan.get("worker_timeout_seconds")
    if not isinstance(timeout, (int, float)) or not 1 <= timeout <= 60:
        raise ValueError("worker_timeout_seconds must be from 1 through 60")
    cases = plan.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("plan must contain cases")
    identifiers: set[str] = set()
    for case in cases:
        identifier = str(case.get("id", ""))
        if not identifier or identifier in identifiers:
            raise ValueError(f"invalid or duplicate case id {identifier!r}")
        identifiers.add(identifier)
        if case.get("expected_relation") not in RELATIONS:
            raise ValueError(f"{identifier}: invalid expected_relation")
        bit = case.get("bit")
        if not isinstance(bit, int) or not 0 <= bit < 128:
            raise ValueError(f"{identifier}: bit must be in 0..127")
        invocations = case.get("invocations")
        if not isinstance(invocations, list) or not invocations:
            raise ValueError(f"{identifier}: no invocations")
        for invocation in invocations:
            bytes.fromhex(str(invocation["input_hex"]))
            output = bytes.fromhex(str(invocation["output_initial_hex"]))
            if len(output) != int(invocation["output_bytes"]):
                raise ValueError(f"{identifier}: output_initial_hex size mismatch")


def _prepare_case(
    tc: Toolchain,
    arch: str,
    case: dict[str, Any],
    artifacts: Path,
) -> dict[str, Any]:
    identifier = str(case["id"])
    source = Path(case["cubin"]["path"]).resolve(strict=True)
    observed_identity = _identity(source)
    if observed_identity != case["cubin"]:
        raise ValueError(f"{identifier}: source cubin identity mismatch")
    detected_arch = cubin_architecture(source)
    if architecture_identity(detected_arch).canonical != arch:
        raise ValueError(f"{identifier}: cubin architecture {detected_arch} is not {arch}")

    kernel = str(case["kernel"])
    index = int(case["instruction_index"])
    instructions = disassemble_cubin(tc, source, arch=raw_arch(arch))
    if not 0 <= index < len(instructions):
        raise ValueError(f"{identifier}: instruction index {index} out of range")
    instruction = instructions[index]
    if instruction.word is None:
        raise ValueError(f"{identifier}: selected instruction has no encoding")
    if instruction.text != case["original_text"]:
        raise ValueError(
            f"{identifier}: target text mismatch: {instruction.text!r} != {case['original_text']!r}"
        )
    if str(instruction.word) != case["original_encoding"]:
        raise ValueError(f"{identifier}: target encoding mismatch")

    bit = int(case["bit"])
    mutated = Word(instruction.word.value ^ (1 << bit))
    cb = Cubin.load(source)
    if cb.read_word(index, kernel) != instruction.word:
        raise ValueError(f"{identifier}: ELF word and nvdisasm word disagree at index {index}")
    cb.write_word(index, mutated, kernel)
    mutant_path = artifacts / f"{identifier}.bit{bit}.cubin"
    cb.save(mutant_path)
    if Cubin.load(mutant_path).read_word(index, kernel) != mutated:
        raise ValueError(f"{identifier}: patched word did not read back")
    changed = sum(a != b for a, b in zip(source.read_bytes(), mutant_path.read_bytes(), strict=True))
    if changed != 1:
        raise ValueError(f"{identifier}: patch changed {changed} bytes rather than one")

    original_raw = decode_word(tc, instruction.word, arch=raw_arch(arch))
    mutant_raw = decode_word(tc, mutated, arch=raw_arch(arch))
    return {
        "id": identifier,
        "stratum": case["stratum"],
        "hypothesis": case["hypothesis"],
        "expected_relation": case["expected_relation"],
        "source_cubin": observed_identity,
        "mutant_cubin": _identity(mutant_path),
        "kernel": kernel,
        "instruction_index": index,
        "bit": bit,
        "original_encoding": str(instruction.word),
        "mutated_encoding": str(mutated),
        "original_cubin_text": instruction.text,
        "original_raw_text": original_raw.text if original_raw else None,
        "mutated_raw_text": mutant_raw.text if mutant_raw else None,
        "raw_decode_relation": (
            "rejected"
            if mutant_raw is None
            else "same"
            if original_raw is not None and mutant_raw.text == original_raw.text
            else "changed"
        ),
        "invocations": case["invocations"],
    }


def _worker_request(
    prepared: dict[str, Any],
    variant: str,
    plan: dict[str, Any],
) -> dict[str, Any]:
    cubin = prepared["source_cubin"] if variant == "original" else prepared["mutant_cubin"]
    return {
        "schema": WORKER_SCHEMA,
        "case_id": prepared["id"],
        "variant": variant,
        "device": plan["device"],
        "cubin": cubin,
        "kernel": prepared["kernel"],
        "repetitions": plan["repetitions"],
        "invocations": prepared["invocations"],
    }


def _run_worker(request: dict[str, Any]) -> dict[str, Any]:
    stages: dict[str, Any] = {}
    result: dict[str, Any] = {
        "schema": WORKER_SCHEMA,
        "case_id": request["case_id"],
        "variant": request["variant"],
        "status": "started",
        "stages": stages,
        "invocations": [],
    }
    try:
        cubin_path = Path(request["cubin"]["path"]).resolve(strict=True)
        if _identity(cubin_path) != request["cubin"]:
            raise ValueError("worker cubin identity mismatch")
        with Device(int(request["device"]["ordinal"])) as device:
            stages["context"] = "created"
            info = device.info
            stages["device_info"] = {
                "name": info.name,
                "arch": info.arch,
                "multiprocessors": info.multiprocessors,
                "clock_khz": info.clock_khz,
                "memory_bytes": info.memory_bytes,
            }
            if info.name != request["device"]["name"] or info.arch != "sm_86":
                raise ValueError("CUDA Driver API device identity mismatch")
            module = device.load_cubin(cubin_path.read_bytes())
            stages["module_load"] = "accepted"
            function = module.function(str(request["kernel"]))
            stages["function_lookup"] = "accepted"

            for invocation in request["invocations"]:
                input_bytes = bytes.fromhex(str(invocation["input_hex"]))
                output_initial = bytes.fromhex(str(invocation["output_initial_hex"]))
                source = device.alloc(len(input_bytes))
                output = device.alloc(len(output_initial))
                samples = []
                for repeat in range(int(request["repetitions"])):
                    device.upload(source, input_bytes)
                    device.upload(output, output_initial)
                    start = time.perf_counter_ns()
                    device.launch_async(
                        function,
                        [ctypes.c_size_t(source), ctypes.c_size_t(output)],
                        grid=tuple(invocation.get("grid", [1, 1, 1])),
                        block=tuple(invocation.get("block", [1, 1, 1])),
                    )
                    launched = time.perf_counter_ns()
                    device.synchronize()
                    synchronized = time.perf_counter_ns()
                    observed = device.download(output, int(invocation["output_bytes"]))
                    samples.append(
                        {
                            "repeat": repeat,
                            "launch_ns": launched - start,
                            "synchronize_ns": synchronized - launched,
                            "output_hex": observed.hex(),
                        }
                    )
                result["invocations"].append({"id": invocation["id"], "samples": samples})
            result["status"] = "completed"
            stages["module_unload"] = "requested"
            module.unload()
            stages["module_unload"] = "completed"
    except (CudaError, OSError, ValueError) as exc:
        result["status"] = "failed"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
    return result


def _invoke_worker(request: dict[str, Any], timeout: float) -> dict[str, Any]:
    command = [sys.executable, str(Path(__file__).resolve()), "--worker-json", json.dumps(request)]
    started = time.perf_counter_ns()
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        return {
            "schema": WORKER_SCHEMA,
            "case_id": request["case_id"],
            "variant": request["variant"],
            "status": "timeout",
            "timeout_seconds": timeout,
            "stdout": exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout,
            "stderr": exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr,
            "process_ns": time.perf_counter_ns() - started,
        }
    try:
        payload = json.loads(process.stdout)
    except json.JSONDecodeError:
        payload = {
            "schema": WORKER_SCHEMA,
            "case_id": request["case_id"],
            "variant": request["variant"],
            "status": "worker_protocol_error",
            "stdout": process.stdout,
        }
    payload["process_returncode"] = process.returncode
    payload["process_stderr"] = process.stderr
    payload["process_ns"] = time.perf_counter_ns() - started
    return payload


def _outputs(worker: dict[str, Any]) -> dict[str, list[str]]:
    return {
        row["id"]: [sample["output_hex"] for sample in row["samples"]]
        for row in worker.get("invocations", [])
    }


def _evaluate(prepared: dict[str, Any], original: dict[str, Any], mutant: dict[str, Any]) -> dict[str, Any]:
    original_outputs = _outputs(original)
    baseline_deterministic = original.get("status") == "completed" and all(
        values and len(set(values)) == 1 for values in original_outputs.values()
    )
    mutant_outputs = _outputs(mutant)
    mutant_deterministic = mutant.get("status") == "completed" and all(
        values and len(set(values)) == 1 for values in mutant_outputs.values()
    )
    output_equal = (
        baseline_deterministic
        and mutant_deterministic
        and original_outputs.keys() == mutant_outputs.keys()
        and all(original_outputs[key][0] == mutant_outputs[key][0] for key in original_outputs)
    )
    relation = prepared["expected_relation"]
    if relation == "equivalent":
        passed = bool(output_equal)
    elif relation == "divergent_or_rejected":
        passed = bool(
            baseline_deterministic
            and (mutant.get("status") != "completed" or (mutant_deterministic and not output_equal))
        )
    else:
        passed = bool(baseline_deterministic)
    return {
        "expected_relation": relation,
        "baseline_deterministic": baseline_deterministic,
        "mutant_deterministic": mutant_deterministic,
        "output_equal": output_equal,
        "passed": passed,
    }


def _run_campaign(plan_path: Path, manifest_path: Path, run_dir: Path, cuda_bin: str | None) -> int:
    plan = _load_json(plan_path.resolve(strict=True))
    _validate_plan(plan)
    accepted = _load_json(manifest_path.resolve(strict=True))
    tc = find_toolchain(cuda_bin)
    initial_runtime = runtime_manifest(tc)
    if initial_runtime != accepted:
        raise ValueError("accepted runtime manifest does not match current runtime objects")
    before_lease = _require_lease(plan)
    before_device = _require_device(plan["device"])

    run_dir.mkdir(parents=True, exist_ok=False)
    artifacts = run_dir / "artifacts"
    artifacts.mkdir()
    prepared_cases = [_prepare_case(tc, "sm_86", case, artifacts) for case in plan["cases"]]

    results = []
    timeout = float(plan["worker_timeout_seconds"])
    for prepared in prepared_cases:
        original = _invoke_worker(_worker_request(prepared, "original", plan), timeout)
        mutant = _invoke_worker(_worker_request(prepared, "mutant", plan), timeout)
        evaluation = _evaluate(prepared, original, mutant)
        results.append(
            {
                "case": prepared,
                "original_execution": original,
                "mutant_execution": mutant,
                "evaluation": evaluation,
            }
        )

    after_lease = _require_lease(plan)
    after_device = _require_device(plan["device"])
    final_runtime = runtime_manifest(tc)
    authority_stable = (
        initial_runtime == final_runtime == accepted
        and before_lease == after_lease
        and before_device == after_device
    )
    passing = authority_stable and all(row["evaluation"]["passed"] for row in results)
    payload = {
        "schema": RESULT_SCHEMA,
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": "PASS" if passing else "FAIL",
        "plan": _identity(plan_path),
        "accepted_runtime_manifest": _identity(manifest_path),
        "runtime_authority_stable": authority_stable,
        "runtime_manifest_before": initial_runtime,
        "runtime_manifest_after": final_runtime,
        "lease_before": before_lease,
        "lease_after": after_lease,
        "device_before": before_device,
        "device_after": after_device,
        "cases": results,
        "claim_ceiling": (
            "Loadability, launch/synchronization outcomes, deterministic output bytes, and per-run "
            "durations for the exact one-bit mutations, inputs, runtime objects, and RTX 3070 named "
            "by this campaign. Output equivalence is not field attribution, scheduling causality, "
            "general semantic equivalence, cross-device validity, or a production sm_86 control model."
        ),
    }
    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "cases": len(results)}, sort_keys=True))
    print(f"wrote {result_path}")
    return 0 if passing else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--accepted-runtime-manifest", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--cuda-bin")
    parser.add_argument("--print-runtime-manifest", action="store_true")
    parser.add_argument("--write-runtime-manifest", type=Path)
    parser.add_argument("--worker-json", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker_json is not None:
        request = json.loads(args.worker_json)
        if request.get("schema") != WORKER_SCHEMA:
            parser.error("invalid worker schema")
        print(json.dumps(_run_worker(request), sort_keys=True))
        return 0
    if args.print_runtime_manifest:
        print(json.dumps(runtime_manifest(find_toolchain(args.cuda_bin)), indent=2, sort_keys=True))
        return 0
    if args.write_runtime_manifest:
        manifest = args.write_runtime_manifest.resolve()
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("x") as stream:
            json.dump(runtime_manifest(find_toolchain(args.cuda_bin)), stream, indent=2, sort_keys=True)
            stream.write("\n")
        print(f"wrote {manifest}")
        return 0
    if not args.plan or not args.accepted_runtime_manifest or not args.run_dir:
        parser.error("campaign mode requires --plan, --accepted-runtime-manifest, and --run-dir")
    return _run_campaign(args.plan, args.accepted_runtime_manifest, args.run_dir, args.cuda_bin)


if __name__ == "__main__":
    raise SystemExit(main())
