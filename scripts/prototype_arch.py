#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Harvest an architecture-neutral SASS corpus without interpreting control bits.

This is the migration bridge for a new architecture.  It records the exact PTX,
cubin, vendor disassembly, runtime identities, and rejections, but deliberately
does not create Basalt ``payload`` or ``control`` fields.  Those fields are an
architecture model and must be measured before they are used.

Example:

    python scripts/prototype_arch.py --arch sm_86 \
        --out-dir /home/pancakes/workdirs/basalt-sm86-raw-v1 --include-tensor
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import _repo

_repo.use_repo_source()

from basalt.architecture import (  # noqa: E402
    ArchitectureError,
    architecture_identity,
    cubin_architecture,
    require_architecture_match,
)
from basalt.disasm import disassemble_cubin  # noqa: E402
from basalt.harvest.corpus import Snippet, generate as generate_scalar  # noqa: E402
from basalt.harvest.corpus_shapes import generate_shapes  # noqa: E402
from basalt.harvest.corpus_tensor import generate_tensor  # noqa: E402
from basalt.toolchain import Toolchain, find_toolchain  # noqa: E402


SCHEMA = "basalt.raw-architecture-corpus.v1"
MANIFEST_SCHEMA = "basalt.runtime-manifest.v1"
TARGET_LINE = re.compile(r"(?m)^\.target\s+\S+\s*$")


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


def _retarget_ptx(ptx: str, arch: str) -> str:
    """Replace exactly one PTX target, refusing a malformed generator result."""
    replaced, count = TARGET_LINE.subn(f".target {arch}", ptx)
    if count != 1:
        raise ValueError(f"expected one .target line, found {count}")
    return replaced


def _runtime_paths(tc: Toolchain) -> dict[str, Path]:
    import basalt.architecture
    import basalt.disasm
    import basalt.harvest.corpus
    import basalt.harvest.corpus_shapes
    import basalt.harvest.corpus_tensor
    import basalt.toolchain

    return {
        "python": Path(sys.executable),
        "ptxas": tc.ptxas,
        "nvdisasm": tc.nvdisasm,
        "prototype_arch": Path(__file__),
        "architecture_module": Path(basalt.architecture.__file__),
        "disasm_module": Path(basalt.disasm.__file__),
        "toolchain_module": Path(basalt.toolchain.__file__),
        "corpus_module": Path(basalt.harvest.corpus.__file__),
        "corpus_shapes_module": Path(basalt.harvest.corpus_shapes.__file__),
        "corpus_tensor_module": Path(basalt.harvest.corpus_tensor.__file__),
    }


def _runtime_manifest(tc: Toolchain) -> dict[str, object]:
    return {
        "schema": MANIFEST_SCHEMA,
        "toolchain_version": tc.version,
        "objects": {name: _identity(path) for name, path in _runtime_paths(tc).items()},
    }


def _repo_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_repo.ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


@dataclass(frozen=True, slots=True)
class RawInstruction:
    offset: int
    text: str
    encoding: str


@dataclass(frozen=True, slots=True)
class RawBuild:
    kernel: str
    label: str
    family: str
    opt_level: int
    status: str
    ptx_path: str
    ptx_sha256: str
    cubin_path: str = ""
    cubin_sha256: str = ""
    cubin_bytes: int = 0
    detected_arch: str = ""
    instructions: tuple[RawInstruction, ...] = ()
    rejection: str = ""


def _first_diagnostic(stdout: str, stderr: str) -> str:
    lines = [line.strip() for line in (stderr + "\n" + stdout).splitlines() if line.strip()]
    return next((line for line in lines if "error" in line.lower()), lines[0] if lines else "unknown")


def _build_one(
    tc: Toolchain,
    arch: str,
    snippet: Snippet,
    opt: int,
    ptx_path: Path,
    cubin_path: Path,
) -> RawBuild:
    result = tc.run(
        [str(tc.ptxas), f"-arch={arch}", f"-O{opt}", "-o", str(cubin_path), str(ptx_path)],
        check=False,
        timeout=120.0,
    )
    common = {
        "kernel": snippet.name,
        "label": snippet.label,
        "family": snippet.family,
        "opt_level": opt,
        "ptx_path": str(ptx_path),
        "ptx_sha256": _sha256(ptx_path),
    }
    if result.returncode != 0 or not cubin_path.is_file():
        return RawBuild(
            **common,
            status="rejected",
            rejection=_first_diagnostic(result.stdout, result.stderr),
        )

    try:
        detected = cubin_architecture(cubin_path)
        require_architecture_match(arch, detected, str(cubin_path))
    except ArchitectureError as exc:
        return RawBuild(
            **common,
            status="architecture-mismatch",
            cubin_path=str(cubin_path),
            cubin_sha256=_sha256(cubin_path),
            cubin_bytes=cubin_path.stat().st_size,
            rejection=str(exc),
        )

    instructions = tuple(
        RawInstruction(offset=ins.offset, text=ins.text, encoding=str(ins.word))
        for ins in disassemble_cubin(tc, cubin_path)
        if ins.word is not None and ins.is_valid
    )
    return RawBuild(
        **common,
        status="built",
        cubin_path=str(cubin_path),
        cubin_sha256=_sha256(cubin_path),
        cubin_bytes=cubin_path.stat().st_size,
        detected_arch=detected,
        instructions=instructions,
    )


def _prepare_sources(snippets: list[Snippet], arch: str, ptx_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for snippet in snippets:
        path = ptx_dir / f"{snippet.name}.ptx"
        path.write_text(_retarget_ptx(snippet.ptx, arch))
        paths[snippet.name] = path
    return paths


def _summary(builds: list[RawBuild]) -> dict[str, object]:
    built = [row for row in builds if row.status == "built"]
    instructions = [ins for row in built for ins in row.instructions]
    return {
        "tasks": len(builds),
        "built": len(built),
        "rejected": sum(row.status == "rejected" for row in builds),
        "architecture_mismatch": sum(row.status == "architecture-mismatch" for row in builds),
        "instructions": len(instructions),
        "distinct_mnemonics": len(
            {
                parts[1] if parts[0].startswith("@") and len(parts) > 1 else parts[0]
                for ins in instructions
                if (parts := ins.text.split())
            }
        ),
        "claim_ceiling": (
            "Exact vendor PTX-to-cubin outputs and nvdisasm text/bytes for the selected "
            "architecture. No control-field layout, operand-field attribution, latency, hazard, "
            "scheduler, semantic-correctness, or runtime-execution claim."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True, help="ptxas target, for example sm_86")
    parser.add_argument("--out-dir", type=Path, required=True, help="fresh durable evidence root")
    parser.add_argument("--cuda-bin", default=None)
    parser.add_argument("--jobs", type=int, default=min(32, (os.cpu_count() or 4) * 2))
    parser.add_argument("--limit", type=int, default=0, help="limit kernels after stable sorting")
    parser.add_argument("--include-tensor", action="store_true")
    parser.add_argument("--opt-levels", type=int, nargs="+", default=[0, 3], choices=range(4))
    args = parser.parse_args()

    arch = architecture_identity(args.arch).canonical
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    ptx_dir = out_dir / "ptx"
    cubin_dir = out_dir / "cubins"
    ptx_dir.mkdir()
    cubin_dir.mkdir()

    tc = find_toolchain(args.cuda_bin)
    initial_runtime = _runtime_manifest(tc)

    snippets = generate_scalar() + generate_shapes()
    if args.include_tensor:
        snippets += generate_tensor()
    snippets = sorted(snippets, key=lambda item: item.name)
    if args.limit:
        snippets = snippets[: args.limit]
    sources = _prepare_sources(snippets, arch, ptx_dir)

    tasks = [(snippet, opt) for snippet in snippets for opt in sorted(set(args.opt_levels))]
    builds: list[RawBuild] = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        pending = {
            pool.submit(
                _build_one,
                tc,
                arch,
                snippet,
                opt,
                sources[snippet.name],
                cubin_dir / f"{snippet.name}.O{opt}.cubin",
            ): (snippet.name, opt)
            for snippet, opt in tasks
        }
        for done, future in enumerate(as_completed(pending), start=1):
            builds.append(future.result())
            if done % 100 == 0 or done == len(tasks):
                print(f"{done}/{len(tasks)} builds", flush=True)

    builds.sort(key=lambda row: (row.kernel, row.opt_level))
    final_runtime = _runtime_manifest(tc)
    if final_runtime != initial_runtime:
        raise SystemExit("runtime authority changed during harvest; refusing to seal evidence")

    runtime_path = out_dir / "runtime-manifest.json"
    runtime_path.write_text(json.dumps(final_runtime, indent=2, sort_keys=True) + "\n")
    payload = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "arch": arch,
        "basalt_git_head": _repo_head(),
        "runtime_manifest": _identity(runtime_path),
        "summary": _summary(builds),
        "builds": [asdict(row) for row in builds],
    }
    corpus_path = out_dir / "raw-corpus.json"
    corpus_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"wrote {corpus_path}")
    return 0 if payload["summary"]["architecture_mismatch"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
