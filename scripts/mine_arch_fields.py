#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Mine architecture-local bit effects without borrowing a control layout.

The input is a ``basalt.raw-architecture-corpus.v1`` artifact produced by
``prototype_arch.py``.  For a deterministic set of representative native
words, this script asks the selected-architecture ``nvdisasm`` raw oracle to
decode the original word and every one-bit mutation.  It records exact witness
pairs and classifies only visible text changes:

* predicate, opcode, modifier, or operand text changed;
* the mutated word was rejected; or
* the bit was accepted but decoder-text-silent.

Decoder-text silence is not a scheduling/control meaning.  Natural corpus
variation is reported separately, and the intersection is emitted only as a
``decode_silent_variable_region_candidate`` for later execution probes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import _repo

_repo.use_repo_source()

from basalt.architecture import (  # noqa: E402
    architecture_identity,
    require_architecture_match,
)
from basalt.disasm import Instruction, decode_words, raw_arch  # noqa: E402
from basalt.encoding import WORD_BITS, Word  # noqa: E402
from basalt.toolchain import Toolchain, find_toolchain  # noqa: E402


SCHEMA = "basalt.architecture-bit-effects.v1"
INPUT_SCHEMA = "basalt.raw-architecture-corpus.v1"
MANIFEST_SCHEMA = "basalt.runtime-manifest.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _runtime_paths(tc: Toolchain) -> dict[str, Path]:
    import basalt.architecture
    import basalt.disasm
    import basalt.encoding
    import basalt.toolchain

    return {
        "python": Path(sys.executable),
        "nvdisasm": tc.nvdisasm,
        "miner": Path(__file__),
        "architecture_module": Path(basalt.architecture.__file__),
        "disasm_module": Path(basalt.disasm.__file__),
        "encoding_module": Path(basalt.encoding.__file__),
        "toolchain_module": Path(basalt.toolchain.__file__),
    }


def _runtime_manifest(tc: Toolchain) -> dict[str, object]:
    return {
        "schema": MANIFEST_SCHEMA,
        "toolchain_version": tc.version,
        "objects": {name: _identity(path) for name, path in _runtime_paths(tc).items()},
    }


def _body(instruction: Instruction) -> str:
    text = instruction.text
    if instruction.predicate:
        text = text[len(instruction.predicate) :].lstrip()
    return text


def _operand_text(instruction: Instruction) -> str:
    parts = _body(instruction).split(None, 1)
    return parts[1].rstrip(" ;") if len(parts) > 1 else ""


def _split_operands(text: str) -> tuple[str, ...]:
    """Split top-level operands while preserving brackets and composite forms."""
    if not text:
        return ()
    parts: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(text):
        if char in "[{(":
            depth += 1
        elif char in "]})":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return tuple(parts)


def _classify_change(original: Instruction, mutated: Instruction | None) -> tuple[str, str]:
    """Return a visible-effect class and optional detail for one bit flip."""
    if mutated is None:
        return "decode_rejected", ""
    if mutated.text == original.text:
        return "text_silent", ""
    if mutated.predicate != original.predicate:
        if _body(mutated) == _body(original):
            return "predicate", f"{original.predicate or '-'}->{mutated.predicate or '-'}"
        return "mixed", "predicate-and-body"
    if mutated.opcode != original.opcode:
        return "opcode", f"{original.opcode}->{mutated.opcode}"
    if mutated.mnemonic != original.mnemonic:
        return "modifier", f"{original.mnemonic}->{mutated.mnemonic}"

    before = _split_operands(_operand_text(original))
    after = _split_operands(_operand_text(mutated))
    if len(before) == len(after):
        changed = [
            index
            for index, pair in enumerate(zip(before, after, strict=True))
            if pair[0] != pair[1]
        ]
        if len(changed) == 1:
            index = changed[0]
            return "operand", f"operand[{index}] {before[index]}->{after[index]}"
        if changed:
            return "mixed", "operands[" + ",".join(str(index) for index in changed) + "]"
    return "operand_shape", f"{len(before)}->{len(after)} operands"


@dataclass(frozen=True, slots=True)
class Representative:
    mnemonic: str
    text: str
    encoding: str
    kernel: str
    opt_level: int


@dataclass(frozen=True, slots=True)
class Witness:
    bit: int
    classification: str
    detail: str
    original_encoding: str
    mutated_encoding: str
    original_text: str
    mutated_text: str | None
    mnemonic: str
    kernel: str
    opt_level: int


def _load_corpus(path: Path, requested_arch: str) -> tuple[dict[str, Any], list[int]]:
    raw = json.loads(path.read_text())
    if raw.get("schema") != INPUT_SCHEMA:
        raise ValueError(f"{path} schema is not {INPUT_SCHEMA}")
    observed_arch = raw.get("arch")
    if not isinstance(observed_arch, str):
        raise ValueError(f"{path} has no architecture identity")
    require_architecture_match(requested_arch, observed_arch, str(path))

    words: list[int] = []
    for build in raw.get("builds", []):
        if build.get("status") != "built":
            continue
        for instruction in build.get("instructions", []):
            encoding = instruction.get("encoding")
            if isinstance(encoding, str) and len(encoding) == 32:
                words.append(int(encoding, 16))
    if not words:
        raise ValueError(f"{path} contains no built instruction words")
    return raw, words


def _representatives(raw: dict[str, Any], per_opcode: int, limit: int) -> list[Representative]:
    candidates: list[Representative] = []
    seen: set[tuple[str, str, str]] = set()
    for build in raw.get("builds", []):
        if build.get("status") != "built":
            continue
        for instruction in build.get("instructions", []):
            text = instruction.get("text")
            encoding = instruction.get("encoding")
            if not isinstance(text, str) or not isinstance(encoding, str) or len(encoding) != 32:
                continue
            parts = text.split()
            if parts and parts[0].startswith("@"):
                parts = parts[1:]
            if not parts:
                continue
            mnemonic = parts[0].rstrip(";")
            key = (mnemonic, text, encoding)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                Representative(
                    mnemonic=mnemonic,
                    text=text,
                    encoding=encoding,
                    kernel=str(build.get("kernel", "")),
                    opt_level=int(build.get("opt_level", -1)),
                )
            )

    candidates.sort(
        key=lambda row: (row.mnemonic.split(".")[0], row.mnemonic, row.text, row.encoding)
    )
    selected: list[Representative] = []
    counts: Counter[str] = Counter()
    for row in candidates:
        opcode = row.mnemonic.split(".")[0]
        if counts[opcode] >= per_opcode:
            continue
        counts[opcode] += 1
        selected.append(row)
        if limit and len(selected) >= limit:
            break
    return selected


def _entropy(ones: int, population: int) -> float:
    if ones == 0 or ones == population:
        return 0.0
    probability = ones / population
    return -probability * math.log2(probability) - (1.0 - probability) * math.log2(
        1.0 - probability
    )


def _natural_stats(words: list[int]) -> list[dict[str, object]]:
    return [
        {
            "bit": bit,
            "population": len(words),
            "ones": (ones := sum((word >> bit) & 1 for word in words)),
            "entropy": round(_entropy(ones, len(words)), 9),
        }
        for bit in range(WORD_BITS)
    ]


def _contiguous(bits: list[int]) -> list[list[int]]:
    spans: list[list[int]] = []
    for bit in sorted(set(bits)):
        if not spans or bit != spans[-1][-1] + 1:
            spans.append([bit])
        else:
            spans[-1].append(bit)
    return spans


def _mine(
    tc: Toolchain,
    arch: str,
    representatives: list[Representative],
    natural: list[dict[str, object]],
) -> tuple[
    list[dict[str, object]],
    list[Witness],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    counts: dict[int, Counter[str]] = {bit: Counter() for bit in range(WORD_BITS)}
    witnesses: dict[tuple[int, str], Witness] = {}
    baseline_mismatches: list[dict[str, object]] = []

    for done, row in enumerate(representatives, start=1):
        original_word = Word(int(row.encoding, 16))
        words = [original_word] + [
            Word(original_word.value ^ (1 << bit)) for bit in range(WORD_BITS)
        ]
        decoded = decode_words(tc, words, arch=raw_arch(arch))
        original = decoded[0]
        if original is None:
            baseline_mismatches.append(
                {
                    "mnemonic": row.mnemonic,
                    "encoding": row.encoding,
                    "reason": "raw-decode-rejected",
                }
            )
            continue
        if original.text != row.text:
            baseline_mismatches.append(
                {
                    "mnemonic": row.mnemonic,
                    "encoding": row.encoding,
                    "cubin_text": row.text,
                    "raw_text": original.text,
                    "reason": "cubin-vs-raw-text-difference",
                }
            )

        for bit, mutated in enumerate(decoded[1:]):
            classification, detail = _classify_change(original, mutated)
            counts[bit][classification] += 1
            key = (bit, classification)
            witnesses.setdefault(
                key,
                Witness(
                    bit=bit,
                    classification=classification,
                    detail=detail,
                    original_encoding=str(original_word),
                    mutated_encoding=str(words[bit + 1]),
                    original_text=original.text,
                    mutated_text=mutated.text if mutated is not None else None,
                    mnemonic=row.mnemonic,
                    kernel=row.kernel,
                    opt_level=row.opt_level,
                ),
            )
        if done % 16 == 0 or done == len(representatives):
            print(f"{done}/{len(representatives)} representatives", flush=True)

    effects: list[dict[str, object]] = []
    silent_variable: list[int] = []
    for bit in range(WORD_BITS):
        observed = sum(counts[bit].values())
        ordered = counts[bit].most_common()
        dominant, dominant_count = ordered[0] if ordered else ("unobserved", 0)
        entropy = float(natural[bit]["entropy"])
        silent_fraction = counts[bit]["text_silent"] / observed if observed else 0.0
        if observed and silent_fraction == 1.0 and entropy > 0.0:
            silent_variable.append(bit)
        effects.append(
            {
                "bit": bit,
                "observed_representatives": observed,
                "class_counts": dict(sorted(counts[bit].items())),
                "dominant_class": dominant,
                "dominant_fraction": round(dominant_count / observed, 9) if observed else 0.0,
                "natural_ones": natural[bit]["ones"],
                "natural_population": natural[bit]["population"],
                "natural_entropy": entropy,
            }
        )

    candidates = [
        {
            "kind": "decode_silent_variable_region_candidate",
            "lo": span[0],
            "hi": span[-1],
            "width": len(span),
            "bits": span,
            "meaning": "unknown",
            "evidence": (
                "Every tested one-bit mutation in this span preserved raw nvdisasm text, and each "
                "bit varies in naturally emitted corpus words. This does not identify scheduling, "
                "control, causality, legality on hardware, or field boundaries."
            ),
        }
        for span in _contiguous(silent_variable)
    ]
    ordered_witnesses = [witnesses[key] for key in sorted(witnesses)]
    return effects, ordered_witnesses, candidates, baseline_mismatches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cuda-bin", default=None)
    parser.add_argument("--per-opcode", type=int, default=2)
    parser.add_argument("--limit", type=int, default=128)
    args = parser.parse_args()

    if args.per_opcode < 1 or args.limit < 1:
        parser.error("--per-opcode and --limit must be positive")
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")

    arch = architecture_identity(args.arch).canonical
    corpus_path = args.corpus.resolve(strict=True)
    raw, words = _load_corpus(corpus_path, arch)
    representatives = _representatives(raw, args.per_opcode, args.limit)
    if not representatives:
        raise SystemExit("no representative native words selected")

    tc = find_toolchain(args.cuda_bin)
    initial_runtime = _runtime_manifest(tc)
    natural = _natural_stats(words)
    effects, witnesses, candidates, baseline_mismatches = _mine(
        tc, arch, representatives, natural
    )
    final_runtime = _runtime_manifest(tc)
    if final_runtime != initial_runtime:
        raise SystemExit("runtime authority changed during mining; refusing to seal evidence")

    payload = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "arch": arch,
        "source_corpus": _identity(corpus_path),
        "runtime_manifest": final_runtime,
        "selection": {
            "natural_words": len(words),
            "representatives": len(representatives),
            "per_opcode": args.per_opcode,
            "limit": args.limit,
            "rows": [asdict(row) for row in representatives],
        },
        "bit_effects": effects,
        "witnesses": [asdict(witness) for witness in witnesses],
        "region_candidates": candidates,
        "baseline_differences": baseline_mismatches,
        "claim_ceiling": (
            "One-bit raw-nvdisasm text effects for the selected representatives plus natural bit "
            "variation in the source corpus. Decoder-text silence does not prove a control field, "
            "scheduling meaning, hardware legality, execution, causality, semantic correctness, or "
            "a production sm_86 model."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "natural_words": len(words),
                "representatives": len(representatives),
                "witnesses": len(witnesses),
                "region_candidates": candidates,
                "baseline_differences": len(baseline_mismatches),
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
