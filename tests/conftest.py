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

import textwrap
from pathlib import Path

import pytest

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
