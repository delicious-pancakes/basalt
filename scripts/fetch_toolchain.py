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
    python scripts/fetch_toolchain.py --libs          # the shipped libraries stage 10 audits
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

REDIST_BASE = "https://developer.download.nvidia.com/compute/cuda/redist"

# the harvester runs both and diffs them, so this picks the primary, not the set
DEFAULT_VERSION = "13.3.1"

# no nvrtc or cudart: a fresh clone without either passes doctor and the whole
# toolchain-marked suite
COMPONENTS = ("cuda_nvcc", "cuda_nvdisasm", "cuda_cuobjdump")

# ~1.2 GB of archives for the largest supply of sm_120 code nobody here wrote,
# so stage 10's audit set is opt-in rather than part of the default fetch
AUDIT_LIBS = (
    "libcublas",
    "libcufft",
    "libcurand",
    "libcusolver",
    "libcusparse",
    "libnpp",
    "libnvjpeg",
)

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
    with urllib.request.urlopen(url) as resp, dest.open("wb") as out:
        shutil.copyfileobj(resp, out)
    return dest


def _wanted(name: str, dirs: tuple[str, ...] = ("bin",)) -> str | None:
    """The flat filename to write for an archive member, or None to skip it.

    Only executables and the libraries beside them are taken, and they are
    written straight into one directory rather than reproducing the archive's
    tree. That is not tidiness: an NVIDIA redistributable nests its payload
    under a long versioned directory name, and reproducing that under a
    checkout that is itself several directories deep exceeds Windows' 260
    character path limit and the extraction fails outright. Found by cloning
    into a deep path and following the README, which is the only way this kind
    of thing gets found.

    Any component may be the marker rather than the immediate parent: the
    compiler sits at `bin/ptxas.exe` but a Windows math library sits at
    `bin/x64/cublas64_13.dll`, and matching only the parent silently extracted
    nothing at all from seven archives.
    """
    parts = name.replace("\\", "/").split("/")
    if len(parts) < 2 or not parts[-1] or not set(parts[:-1]) & set(dirs):
        return None
    # an import or static library carries no device code, only a link-time stub
    if parts[-1].endswith((".lib", ".a")):
        return None
    return parts[-1]


def extract(archive: Path, bindir: Path, dirs: tuple[str, ...] = ("bin",)) -> int:
    """Unpack the executables from a redistributable into one flat directory.

    NVIDIA ships Windows components as `.zip` and Linux ones as `.tar.xz`, so
    the extension decides. `filter="data"` is passed to tarfile because the
    default became an error in 3.14 and, more to the point, an archive fetched
    over the network should not be able to write outside the directory it is
    handed.
    """
    bindir.mkdir(parents=True, exist_ok=True)
    written = 0

    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                flat = _wanted(member, dirs)
                if flat is None:
                    continue
                with zf.open(member) as src, (bindir / flat).open("wb") as out:
                    shutil.copyfileobj(src, out)
                written += 1
        return written

    with tarfile.open(archive) as tf:
        for member in tf.getmembers():
            flat = _wanted(member.name, dirs)
            if flat is None or not member.isfile():
                continue
            src = tf.extractfile(member)
            if src is None:
                continue
            target = bindir / flat
            with src, target.open("wb") as out:
                shutil.copyfileobj(src, out)
            # tarfile does not carry the mode across a manual copy, and a
            # toolchain binary that is not executable is no toolchain at all
            target.chmod(target.stat().st_mode | 0o755)
            written += 1
    return written


def manifest(version: str) -> dict:
    url = f"{REDIST_BASE}/redistrib_{version}.json"
    with urllib.request.urlopen(url) as resp:
        return json.load(resp)


def _pull(entry: dict, cache: Path, dest: Path, dirs: tuple[str, ...]) -> int:
    """Fetch one component, check its digest, and unpack it flat into `dest`."""
    archive = fetch(
        f"{REDIST_BASE}/{entry['relative_path']}", cache / Path(entry["relative_path"]).name
    )
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if (expected := entry.get("sha256")) and digest != expected:
        raise SystemExit(
            f"checksum mismatch for {archive.name}\n  expected {expected}\n  got      {digest}"
        )
    return extract(archive, dest, dirs)


def fetch_libs(man: dict, key: str, root: Path, cache: Path, version: str) -> int:
    """Fetch the shipped libraries whose sm_120 kernels stage 10 audits.

    Written flat into one directory with a manifest recording the exact
    component version beside each file, because "we found a hazard in cuBLAS"
    is worth nothing without saying which build of it.
    """
    libdir = root / "libs"
    libdir.mkdir(parents=True, exist_ok=True)
    versions: dict[str, str] = {}

    print(f"CUDA {man.get('release_label', version)} libraries for {key} -> {libdir}")
    for comp in AUDIT_LIBS:
        entry = man.get(comp, {}).get(key)
        if entry is None:
            print(f"  skip    {comp} (not published for {key})")
            continue
        count = _pull(entry, cache, libdir, ("bin", "lib"))
        versions[comp] = man[comp].get("version", "")
        print(f"  unpack  {count} files from {comp} {versions[comp]}", flush=True)

    (libdir / "versions.json").write_text(
        json.dumps({"cuda": man.get("release_label", version), "components": versions}, indent=2),
        encoding="utf-8",
    )
    files = sorted(p.name for p in libdir.iterdir() if p.is_file())
    print(f"\ninstalled {len(files)} files into {libdir}")
    print(f"\nnext:\n  python scripts/audit_shipped.py --libs {libdir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--version", default=DEFAULT_VERSION, help=f"CUDA release (default {DEFAULT_VERSION})"
    )
    ap.add_argument(
        "--dest", type=Path, default=None, help="install root (default third_party/cuda)"
    )
    ap.add_argument("--list", action="store_true", help="list components in the manifest and exit")
    ap.add_argument(
        "--libs",
        action="store_true",
        help="fetch the shipped math libraries stage 10 audits, not the compiler",
    )
    args = ap.parse_args()

    key = platform_key()
    man = manifest(args.version)

    if args.list:
        for name, entry in sorted(man.items()):
            if isinstance(entry, dict) and key in entry:
                # the manifest quotes the size as a string, and dividing that by
                # a float is a TypeError rather than a size
                mb = int(entry[key].get("size", 0)) / 1e6
                print(f"{name:22s} {mb:8.1f} MB  {entry.get('version', '')}")
        return 0

    root = (args.dest or DEST) / args.version
    cache = (args.dest or DEST) / "_archives"

    if args.libs:
        return fetch_libs(man, key, root, cache, args.version)

    bindir = root / "bin"
    bindir.mkdir(parents=True, exist_ok=True)

    print(f"CUDA {man.get('release_label', args.version)} for {key} -> {root}")
    for comp in COMPONENTS:
        entry = man.get(comp, {}).get(key)
        if entry is None:
            print(f"  skip    {comp} (not published for {key})")
            continue
        count = _pull(entry, cache, bindir, ("bin",))
        print(f"  unpack  {count} files from {comp}")

    tools = sorted(p.name for p in bindir.iterdir() if p.is_file())
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
