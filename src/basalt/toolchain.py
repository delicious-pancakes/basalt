# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Locating and driving the stock NVIDIA tools.

basalt never links against anything NVIDIA ships. It drives `ptxas` and
`nvdisasm` as subprocesses and reads their output, which keeps the whole
pipeline reproducible from a pinned redistributable archive and keeps us
clear of any headers or libraries with redistribution terms attached.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

__all__ = ["Toolchain", "ToolchainError", "find_toolchain"]


class ToolchainError(RuntimeError):
    """Raised when a required NVIDIA tool is missing or misbehaves."""


# search order for an unconfigured toolchain. BASALT_CUDA_BIN wins so the
# harvester can be pinned to one ptxas build across a whole corpus run.
_ENV_VAR = "BASALT_CUDA_BIN"
_FALLBACK_ENVS = ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT")
# where scripts/fetch_toolchain.py installs, searched last so a configured
# toolchain always wins over one that happens to be lying in the tree
_VENDORED = Path(__file__).resolve().parents[2] / "third_party" / "cuda"


@dataclass(frozen=True)
class Toolchain:
    """A resolved set of NVIDIA command-line tools.

    Instances are frozen and carry their own version string so a harvest
    record can name the exact compiler that produced it. Encodings are
    stable across ptxas builds far more often than not, but "far more often
    than not" is not a guarantee we are willing to bake into a database.
    """

    bin_dir: Path

    def _tool(self, name: str) -> Path:
        exe = self.bin_dir / (name + (".exe" if os.name == "nt" else ""))
        if not exe.is_file():
            raise ToolchainError(f"{name} not found in {self.bin_dir}")
        return exe

    @cached_property
    def ptxas(self) -> Path:
        return self._tool("ptxas")

    @cached_property
    def nvdisasm(self) -> Path:
        return self._tool("nvdisasm")

    @cached_property
    def cuobjdump(self) -> Path:
        return self._tool("cuobjdump")

    @cached_property
    def version(self) -> str:
        """The `release X.Y, VX.Y.Z` string ptxas reports, normalised to VX.Y.Z."""
        out = self.run([str(self.ptxas), "--version"]).stdout
        for token in out.split():
            if token.startswith("V") and token[1:2].isdigit():
                return token.rstrip(",")
        raise ToolchainError(f"could not parse a version out of: {out!r}")

    @staticmethod
    def run(
        argv: list[str],
        *,
        check: bool = True,
        timeout: float = 120.0,
        stdin: bytes | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run a tool and capture text output.

        `check` defaults to True, but the harvester deliberately turns it off:
        a corpus entry that fails to assemble is data, not an error.
        """
        try:
            proc = subprocess.run(
                argv,
                input=stdin,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolchainError(f"{argv[0]} timed out after {timeout}s") from exc

        result = subprocess.CompletedProcess(
            argv,
            proc.returncode,
            proc.stdout.decode("utf-8", "replace"),
            proc.stderr.decode("utf-8", "replace"),
        )
        if check and result.returncode != 0:
            raise ToolchainError(
                f"{Path(argv[0]).name} exited {result.returncode}\n"
                f"argv: {' '.join(argv)}\n"
                f"stderr: {result.stderr.strip()}"
            )
        return result

    def describe(self) -> dict[str, str]:
        """Provenance block embedded in every artefact we emit."""
        return {"cuda_version": self.version, "bin_dir": str(self.bin_dir)}


def find_toolchain(explicit: str | os.PathLike[str] | None = None) -> Toolchain:
    """Resolve a Toolchain from an explicit path, the environment, or PATH."""
    candidates: list[Path] = []

    if explicit is not None:
        candidates.append(Path(explicit))
    if env := os.environ.get(_ENV_VAR):
        candidates.append(Path(env))
    for var in _FALLBACK_ENVS:
        if root := os.environ.get(var):
            candidates.append(Path(root) / "bin")
    if which := shutil.which("ptxas"):
        candidates.append(Path(which).parent)
    if _VENDORED.is_dir():
        # newest first, so a tree holding two fetched versions picks one and
        # keeps picking it rather than depending on directory order
        candidates.extend(
            sorted((d / "bin" for d in _VENDORED.iterdir() if d.is_dir()), reverse=True)
        )

    for cand in candidates:
        if (cand / "ptxas").is_file() or (cand / "ptxas.exe").is_file():
            return Toolchain(cand.resolve())

    raise ToolchainError(
        "no CUDA toolchain found. set "
        f"{_ENV_VAR}=/path/to/cuda/bin, or run scripts/fetch_toolchain.py "
        "to download a pinned redistributable."
    )
