# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""The archive member filter, which decides what lands in the toolchain.

Worth testing on its own because the rule it encodes is not obvious and was
learned the hard way: NVIDIA nests a redistributable's payload under a long
versioned directory, and reproducing that tree under a checkout that is itself
a few directories deep exceeds Windows' 260 character path limit and the
extraction fails outright. Taking only the executables, and writing them flat,
avoids the problem entirely.

Needs no network and no toolchain.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "fetch_toolchain.py"


def _load():
    spec = importlib.util.spec_from_file_location("fetch_toolchain", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fetch():
    return _load()


class TestMemberFilter:
    @pytest.mark.parametrize(
        ("member", "expected"),
        [
            ("cuda_nvcc-windows-x86_64-13.3.73-archive/bin/ptxas.exe", "ptxas.exe"),
            ("cuda_nvdisasm-linux-x86_64-13.3.73-archive/bin/nvdisasm", "nvdisasm"),
            # a backslash separator, which some zip writers emit
            (r"cuda_nvcc-archive\bin\nvcc.exe", "nvcc.exe"),
        ],
    )
    def test_executables_are_taken_and_flattened(self, fetch, member, expected):
        assert fetch._wanted(member) == expected

    @pytest.mark.parametrize(
        "member",
        [
            "cuda_nvcc-archive/include/cuda.h",
            "cuda_nvcc-archive/nvvm/bin/cicc",  # nested bin, but still bin
            "cuda_nvcc-archive/LICENSE",
            "cuda_nvcc-archive/bin/",  # the directory entry itself
            "ptxas.exe",  # no directory at all
            "",
        ],
    )
    def test_everything_else_is_skipped_or_handled(self, fetch, member):
        result = fetch._wanted(member)
        if member.endswith("/cicc"):
            # nvvm/bin is still a bin directory, and cicc belongs with the rest
            assert result == "cicc"
        else:
            assert result is None

    def test_the_result_never_contains_a_separator(self, fetch):
        """A flat name is the whole point; a separator would rebuild the tree."""
        got = fetch._wanted("a/very/deeply/nested/archive/bin/ptxas.exe")
        assert got is not None
        assert "/" not in got and "\\" not in got


class TestComponents:
    def test_only_what_is_needed_is_downloaded(self, fetch):
        """nvrtc is 300 MB and cudart is unused; neither should creep back in."""
        assert "cuda_nvrtc" not in fetch.COMPONENTS
        assert "cuda_cudart" not in fetch.COMPONENTS
        assert "cuda_nvcc" in fetch.COMPONENTS
        assert "cuda_nvdisasm" in fetch.COMPONENTS


class TestPlatformKey:
    def test_a_known_platform_resolves(self, fetch):
        key = fetch.platform_key()
        assert key in {"windows-x86_64", "linux-x86_64", "linux-sbsa"}
