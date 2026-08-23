# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""The documentation's code is code, so it is run rather than reviewed.

`docs/API.md` is what someone reads before importing anything, and its examples
were written by running a script and transcribing the result. Transcription is
the step that rots: a rename lands in the source, the tests still pass, and the
page keeps telling people to call something that no longer exists.

So the blocks are lifted back out of the markdown and executed in order, in one
namespace, the way a reader would work through them.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
API = ROOT / "docs" / "API.md"


def _blocks() -> list[str]:
    return re.findall(r"```python\n(.*?)```", API.read_text(encoding="utf-8"), re.S)


def test_the_api_page_has_examples() -> None:
    # a page whose blocks all disappeared would otherwise pass the next test
    assert len(_blocks()) >= 6


@pytest.mark.toolchain
def test_every_documented_example_runs(toolchain, sample_cubin, tmp_path, capsys) -> None:
    blocks = _blocks()
    # the first block opens "demo.cubin" relative to the working directory,
    # because that is what reads well on the page
    (tmp_path / "demo.cubin").write_bytes(sample_cubin.read_bytes())

    namespace: dict = {}
    here = Path.cwd()
    os.chdir(tmp_path)
    try:
        for number, block in enumerate(blocks, 1):
            try:
                exec(compile(block, f"API.md block {number}", "exec"), namespace)
            except Exception as exc:
                pytest.fail(f"docs/API.md block {number} does not run: {exc!r}")
    finally:
        os.chdir(here)
    capsys.readouterr()


class TestTheClaimsAroundThem:
    """Two things the page states that a rename would quietly falsify."""

    def test_the_module_map_names_modules_that_import(self) -> None:
        import importlib

        text = API.read_text(encoding="utf-8")
        table = text.split("## Module map", 1)[1]
        modules = re.findall(r"^\| `(basalt[\w.]*)` \|", table, re.M)
        assert len(modules) >= 10
        for name in modules:
            importlib.import_module(name)

    def test_every_name_the_module_map_advertises_exists(self) -> None:
        import importlib

        text = API.read_text(encoding="utf-8")
        table = text.split("## Module map", 1)[1]
        for module, cell in re.findall(r"^\| `(basalt[\w.]*)` \| (.+) \|$", table, re.M):
            imported = importlib.import_module(module)
            for symbol in re.findall(r"`([A-Za-z_][\w]*)(?:\(\))?`", cell):
                assert hasattr(imported, symbol), f"{module}.{symbol} is advertised and absent"
