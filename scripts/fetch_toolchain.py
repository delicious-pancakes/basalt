#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Fetch a pinned CUDA redistributable containing ptxas and nvdisasm.

We deliberately do not use the full CUDA installer. The redistributable
archives are a few tens of megabytes, need no administrator rights, touch
nothing outside this repository, and pin to an exact compiler build, which
is what makes a harvest reproducible months later.

    python scripts/fetch_toolchain.py                 # default pinned version
    python scripts/fetch_toolchain.py --version 13.0.3
    python scripts/fetch_toolchain.py --list
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

REDIST_BASE = "https://developer.download.nvidia.com/compute/cuda/redist"

# Pinned default. 13.3.1 is the newest release at time of writing and carries
# instruction forms 13.0 does not; the harvester runs both and diffs them, so
# changing this default changes which build is "primary", not which are used.
DEFAULT_VERSION = "13.3.1"

# nvrtc is deliberately excluded: it is 300 MB and the pipeline never calls it.
COMPONENTS = ("cuda_nvcc", "cuda_nvdisasm", "cuda_cuobjdump", "cuda_cudart")

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "third_party" / "cuda"


def platform_key() -> str:
    system, machine = platform.system(), platform.machine().lower()
    arch = {"amd64": "x86_64", "x86_64": "x86_64", "aarch64": "sbsa", "arm64": "sbsa"}.get(machine)
    if arch is None:
        raise SystemExit(f"unsupported machine: {machine}")
    if system == "Windows":
        return "windows-x86_64"
    if system == "Linux":
        return f"linux-{arch}"
    raise SystemExit(f"unsupported platform: {system}. ptxas ships for Linux and Windows only.")


def fetch(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  cached  {dest.name}")
        return dest
    print(f"  fetch   {dest.name}")
    with urllib.request.urlopen(url) as resp, dest.open("wb") as out:  # noqa: S310
        shutil.copyfileobj(resp, out)
    return dest


def manifest(version: str) -> dict:
    url = f"{REDIST_BASE}/redistrib_{version}.json"
    with urllib.request.urlopen(url) as resp:  # noqa: S310
        return json.load(resp)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default=DEFAULT_VERSION, help=f"CUDA release (default {DEFAULT_VERSION})")
    ap.add_argument("--dest", type=Path, default=None, help="install root (default third_party/cuda)")
    ap.add_argument("--list", action="store_true", help="list components in the manifest and exit")
    args = ap.parse_args()

    key = platform_key()
    man = manifest(args.version)

    if args.list:
        for name, entry in sorted(man.items()):
            if isinstance(entry, dict) and key in entry:
                mb = entry[key]["size"] / 1e6
                print(f"{name:22s} {mb:8.1f} MB  {entry.get('version','')}")
        return 0

    root = (args.dest or DEST) / args.version
    root.mkdir(parents=True, exist_ok=True)
    cache = (args.dest or DEST) / "_archives"

    print(f"CUDA {man.get('release_label', args.version)} for {key} -> {root}")
    for comp in COMPONENTS:
        entry = man.get(comp, {}).get(key)
        if entry is None:
            print(f"  skip    {comp} (not published for {key})")
            continue
        archive = fetch(f"{REDIST_BASE}/{entry['relative_path']}", cache / Path(entry["relative_path"]).name)

        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if (expected := entry.get("sha256")) and digest != expected:
            raise SystemExit(f"checksum mismatch for {archive.name}\n  expected {expected}\n  got      {digest}")

        with zipfile.ZipFile(archive) as zf:
            zf.extractall(root / "_raw")

    # flatten every component's bin/ into one directory so BASALT_CUDA_BIN is a
    # single path regardless of how many archives were unpacked
    bindir = root / "bin"
    bindir.mkdir(exist_ok=True)
    for src in (root / "_raw").rglob("*"):
        if src.is_file() and src.parent.name == "bin":
            shutil.copy2(src, bindir / src.name)

    tools = sorted(p.name for p in bindir.iterdir() if p.suffix in {".exe", ""} and p.is_file())
    print(f"\ninstalled {len(tools)} files into {bindir}")

    missing = [t for t in ("ptxas", "nvdisasm") if not any(n.startswith(t) for n in tools)]
    if missing:
        raise SystemExit(f"required tools missing after extract: {', '.join(missing)}")

    print("\nset this for the current shell:")
    if platform.system() == "Windows":
        print(f'  $env:BASALT_CUDA_BIN = "{bindir}"')
    else:
        print(f'  export BASALT_CUDA_BIN="{bindir}"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
