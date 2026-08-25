#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Mechanically verify a ``basalt.architecture-bit-effects.v1`` artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import _repo

_repo.use_repo_source()

from basalt.architecture import architecture_identity  # noqa: E402
from basalt.encoding import WORD_BITS  # noqa: E402


SCHEMA = "basalt.architecture-bit-effects.v1"
RECEIPT_SCHEMA = "basalt.architecture-bit-effects-verification.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _check_identity(identity: dict[str, Any], label: str, failures: list[str]) -> None:
    try:
        path = Path(identity["path"])
        expected_bytes = int(identity["bytes"])
        expected_sha = str(identity["sha256"])
    except (KeyError, TypeError, ValueError):
        failures.append(f"{label}: malformed identity")
        return
    if not path.is_file():
        failures.append(f"{label}: missing {path}")
        return
    if path.stat().st_size != expected_bytes:
        failures.append(f"{label}: byte count changed")
    if _sha256(path) != expected_sha:
        failures.append(f"{label}: sha256 changed")


def _one_bit_witness(witness: dict[str, Any]) -> bool:
    try:
        bit = int(witness["bit"])
        original = int(witness["original_encoding"], 16)
        mutated = int(witness["mutated_encoding"], 16)
    except (KeyError, TypeError, ValueError):
        return False
    delta = original ^ mutated
    return 0 <= bit < WORD_BITS and delta == 1 << bit


def verify(payload: dict[str, Any], expected_arch: str | None = None) -> list[str]:
    failures: list[str] = []
    if payload.get("schema") != SCHEMA:
        failures.append(f"schema: expected {SCHEMA}")

    try:
        arch = architecture_identity(str(payload["arch"])).canonical
    except (KeyError, ValueError) as exc:
        failures.append(f"architecture: {exc}")
        arch = ""
    if expected_arch is not None:
        try:
            expected = architecture_identity(expected_arch).canonical
        except ValueError as exc:
            failures.append(f"expected architecture: {exc}")
        else:
            if arch != expected:
                failures.append(f"architecture: observed {arch}, expected {expected}")

    source = payload.get("source_corpus")
    if isinstance(source, dict):
        _check_identity(source, "source corpus", failures)
    else:
        failures.append("source corpus: missing identity")

    runtime = payload.get("runtime_manifest")
    objects = runtime.get("objects") if isinstance(runtime, dict) else None
    if not isinstance(objects, dict) or not objects:
        failures.append("runtime manifest: missing objects")
    else:
        for name, identity in sorted(objects.items()):
            if isinstance(identity, dict):
                _check_identity(identity, f"runtime {name}", failures)
            else:
                failures.append(f"runtime {name}: malformed identity")

    selection = payload.get("selection")
    if not isinstance(selection, dict):
        failures.append("selection: missing")
        representatives = 0
    else:
        representatives = int(selection.get("representatives", 0))
        rows = selection.get("rows")
        if not isinstance(rows, list) or len(rows) != representatives:
            failures.append("selection: representative row count mismatch")

    raw_rejected = sum(
        row.get("reason") == "raw-decode-rejected"
        for row in payload.get("baseline_differences", [])
        if isinstance(row, dict)
    )
    expected_observations = representatives - raw_rejected

    effects = payload.get("bit_effects")
    if not isinstance(effects, list) or len(effects) != WORD_BITS:
        failures.append(f"bit effects: expected {WORD_BITS} rows")
        effects_by_bit: dict[int, dict[str, Any]] = {}
    else:
        effects_by_bit = {
            int(row.get("bit", -1)): row for row in effects if isinstance(row, dict)
        }
        if set(effects_by_bit) != set(range(WORD_BITS)):
            failures.append("bit effects: bit domain is not exactly 0..127")

    witnesses = payload.get("witnesses")
    if not isinstance(witnesses, list):
        failures.append("witnesses: missing")
        witnesses = []
    witnessed_classes: set[tuple[int, str]] = set()
    for index, witness in enumerate(witnesses):
        if not isinstance(witness, dict) or not _one_bit_witness(witness):
            failures.append(f"witnesses[{index}]: not an exact one-bit pair")
            continue
        witnessed_classes.add((int(witness["bit"]), str(witness.get("classification", ""))))

    for bit, effect in sorted(effects_by_bit.items()):
        class_counts = effect.get("class_counts")
        if not isinstance(class_counts, dict):
            failures.append(f"bit {bit}: missing class counts")
            continue
        observed = int(effect.get("observed_representatives", -1))
        if observed != expected_observations:
            failures.append(f"bit {bit}: observed {observed}, expected {expected_observations}")
        if sum(int(value) for value in class_counts.values()) != observed:
            failures.append(f"bit {bit}: class counts do not sum to observed")
        for classification, count in class_counts.items():
            if int(count) > 0 and (bit, str(classification)) not in witnessed_classes:
                failures.append(f"bit {bit}: {classification} has no exact witness")

    for index, candidate in enumerate(payload.get("region_candidates", [])):
        if not isinstance(candidate, dict):
            failures.append(f"candidate[{index}]: malformed")
            continue
        bits = candidate.get("bits")
        if (
            candidate.get("kind") != "decode_silent_variable_region_candidate"
            or candidate.get("meaning") != "unknown"
            or not isinstance(bits, list)
            or not bits
        ):
            failures.append(f"candidate[{index}]: widened or malformed claim")
            continue
        numeric = [int(bit) for bit in bits]
        if numeric != list(range(numeric[0], numeric[-1] + 1)):
            failures.append(f"candidate[{index}]: bits are not contiguous")
        if (
            int(candidate.get("lo", -1)) != numeric[0]
            or int(candidate.get("hi", -1)) != numeric[-1]
            or int(candidate.get("width", -1)) != len(numeric)
        ):
            failures.append(f"candidate[{index}]: span metadata mismatch")
        for bit in numeric:
            effect = effects_by_bit.get(bit, {})
            counts = effect.get("class_counts", {})
            if counts != {"text_silent": expected_observations}:
                failures.append(f"candidate[{index}]: bit {bit} is not universally text-silent")
            if float(effect.get("natural_entropy", 0.0)) <= 0.0:
                failures.append(f"candidate[{index}]: bit {bit} has no natural variation")

    ceiling = payload.get("claim_ceiling")
    if not isinstance(ceiling, str) or "does not prove" not in ceiling:
        failures.append("claim ceiling: missing explicit non-proof boundary")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--arch", default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    artifact = args.artifact.resolve(strict=True)
    payload = json.loads(artifact.read_text())
    failures = verify(payload, args.arch)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "verified_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "artifact": {
            "path": str(artifact),
            "bytes": artifact.stat().st_size,
            "sha256": _sha256(artifact),
        },
        "arch": payload.get("arch"),
        "result": "PASS" if not failures else "FAIL",
        "failures": failures,
        "claim_ceiling": (
            "Mechanical artifact integrity and one-bit witness completeness only; no "
            "control-field, scheduling, execution, hardware-legality, causality, or semantic "
            "acceptance."
        ),
    }
    if args.out is not None:
        if args.out.exists():
            parser.error(f"refusing to overwrite {args.out}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
