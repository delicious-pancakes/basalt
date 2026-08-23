# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Shared fixtures.

The suite is split so that the majority of it runs anywhere. Tests that need
`ptxas` and `nvdisasm` are marked `toolchain`; tests that need a physical sm_120
device are marked `gpu`. Neither marker is required for the encoding, operand,
or hazard logic, which is deliberate: the parts most likely to be wrong are the
parts that need nothing installed to check.
"""

from __future__ import annotations

import os
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from basalt.disasm import Program, disassemble_program
from basalt.toolchain import Toolchain, ToolchainError, find_toolchain

ARCH = "sm_120a"


@pytest.fixture(scope="session")
def toolchain() -> Toolchain:
    try:
        return find_toolchain()
    except ToolchainError as exc:
        pytest.skip(f"no CUDA toolchain: {exc}")


@pytest.fixture(scope="session")
def sample_ptx() -> str:
    """A kernel with a real dependency chain through several pipelines.

    Deliberately mixes a variable-latency load, fixed-latency arithmetic, a
    predicate, and a store, so a verifier bug in any one of those shows up.
    """
    return textwrap.dedent(
        """\
        .version 9.0
        .target sm_120a
        .address_size 64
        .visible .entry k(.param .u64 pin, .param .u64 pout)
        {
          .reg .b32 %r<12>; .reg .b64 %d<8>; .reg .f32 %f<12>; .reg .pred %p<4>;
          ld.param.u64 %d1,[pin]; cvta.to.global.u64 %d1,%d1;
          ld.param.u64 %d2,[pout]; cvta.to.global.u64 %d2,%d2;
          mov.u32 %r1, %tid.x;
          mul.wide.u32 %d3, %r1, 4;
          add.s64 %d4, %d1, %d3;
          ld.global.f32 %f1, [%d4];
          ld.global.f32 %f2, [%d4+4];
          fma.rn.f32 %f3, %f1, %f2, %f1;
          mul.f32 %f4, %f3, %f3;
          add.f32 %f5, %f4, %f3;
          setp.gt.f32 %p1, %f5, 0f3F800000;
          selp.f32 %f6, %f5, %f4, %p1;
          add.s64 %d5, %d2, %d3;
          st.global.f32 [%d5], %f6;
          ret;
        }
        """
    )


@pytest.fixture(scope="session")
def sample_cubin(toolchain: Toolchain, sample_ptx: str, tmp_path_factory) -> Path:
    tmp = tmp_path_factory.mktemp("basalt-sample")
    src, cubin = tmp / "k.ptx", tmp / "k.cubin"
    src.write_text(sample_ptx)
    res = toolchain.run(
        [str(toolchain.ptxas), f"-arch={ARCH}", "-O3", "-o", str(cubin), str(src)],
        check=False,
    )
    if res.returncode != 0:
        pytest.skip(f"ptxas could not build the sample: {res.stderr.strip()}")
    return cubin


class CorpusBuilds:
    """The corpus compiled and disassembled once per optimisation level.

    Six fixtures wanted the same kernels at the same levels and each paid for
    its own `ptxas` and `nvdisasm` run, which was most of the suite's runtime.
    Sharing the result is safe because `Program` is frozen and nothing in the
    suite mutates one; what the fixtures actually differ in is the analysis they
    run afterwards, which is cheap.

    Levels are built on first use, so asking for one class does not pay for the
    other two.
    """

    def __init__(self, toolchain: Toolchain, root: Path) -> None:
        self._toolchain = toolchain
        self._root = root
        self._cache: dict[int, dict[str, tuple[Path, Program]]] = {}
        # kept so a kernel that never compiled cannot go on looking like a form
        # the architecture does not have
        self.rejected: dict[str, str] = {}

    def at(self, opt: int) -> dict[str, tuple[Path, Program]]:
        """Every kernel that built at `-O{opt}`, by name, with its cubin path."""
        if opt not in self._cache:
            self._cache[opt] = self._build(opt)
        return self._cache[opt]

    def _build(self, opt: int) -> dict[str, tuple[Path, Program]]:
        from basalt.harvest.corpus import generate
        from basalt.harvest.corpus_shapes import generate_shapes
        from basalt.harvest.corpus_tensor import generate_tensor

        out_dir = self._root / f"O{opt}"
        out_dir.mkdir(parents=True, exist_ok=True)

        def one(snippet):
            src = out_dir / f"{snippet.name}.ptx"
            cubin = out_dir / f"{snippet.name}.cubin"
            src.write_text(snippet.ptx)
            built = self._toolchain.run(
                [
                    str(self._toolchain.ptxas),
                    f"-arch={ARCH}",
                    f"-O{opt}",
                    "-o",
                    str(cubin),
                    str(src),
                ],
                check=False,
                timeout=120.0,
            )
            # a kernel ptxas rejects is a recorded negative, not a failure
            if built.returncode != 0:
                return snippet.name, None, f"{built.stdout}\n{built.stderr}"
            program = disassemble_program(self._toolchain, cubin)
            return snippet.name, (cubin, program), None

        snippets = generate() + generate_tensor() + generate_shapes()
        with ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 4) * 2)) as pool:
            rows = list(pool.map(one, snippets))
        for name, built, why in rows:
            if built is None:
                self.rejected[f"{name}@O{opt}"] = why or ""
        return {name: built for name, built, _ in rows if built is not None}


@pytest.fixture(scope="session")
def corpus_builds(toolchain: Toolchain, tmp_path_factory) -> CorpusBuilds:
    return CorpusBuilds(toolchain, tmp_path_factory.mktemp("basalt-corpus"))
