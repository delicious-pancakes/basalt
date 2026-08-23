# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""What the package says about itself.

`--version` reads `basalt.__version__` and the wheel's metadata reads
`pyproject.toml`. They are two files, so they can disagree, and a release that
reports one version while carrying another is the kind of thing nobody notices
until someone quotes it back.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import basalt
from basalt.paths import ISA_DATABASE, LATENCIES, OBSERVED_STALLS

ROOT = Path(__file__).resolve().parent.parent


def _pyproject() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def test_the_module_and_the_package_agree_on_the_version() -> None:
    assert basalt.__version__ == _pyproject()["project"]["version"]


def test_the_citation_file_carries_the_same_version() -> None:
    text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    line = next(ln for ln in text.splitlines() if ln.startswith("version:"))
    assert line.split(":", 1)[1].strip().strip('"') == basalt.__version__


class TestTheMeasuredDataShips:
    """The tables are the product, not documentation.

    Read from `data/` relative to the working directory, an installed wheel run
    anywhere else silently fell back to assumed latencies and then reported
    hazards it could not ground, on code that is clean.
    """

    def test_every_default_table_exists_beside_the_package(self) -> None:
        for path in (ISA_DATABASE, LATENCIES, OBSERVED_STALLS):
            assert path.is_file(), path
            assert path.is_relative_to(Path(basalt.__file__).resolve().parent)

    def test_the_package_data_is_declared_for_the_wheel(self) -> None:
        declared = _pyproject()["tool"]["setuptools"]["package-data"]["basalt"]
        assert "data/isa/*.json" in declared
        assert "data/latency/*.json" in declared

    def test_the_tables_load(self) -> None:
        from basalt.isa.database import IsaDatabase
        from basalt.verify.latency import Confidence, LatencyModel
        from basalt.verify.observed import ObservedStalls

        assert IsaDatabase.read(ISA_DATABASE).forms
        assert LatencyModel.assumed().overlay(LATENCIES).lookup("IMAD").confidence is (
            Confidence.MEASURED
        )
        assert ObservedStalls.read(OBSERVED_STALLS).by_pair


class TestTheReadmeIsAlsoThePackagePage:
    """`readme = "README.md"` makes this file PyPI's project description.

    PyPI serves it from its own domain and resolves nothing against the
    repository, so a relative path that works on GitHub renders as a broken
    image there. Six of them did, including the header.
    """

    @staticmethod
    def _references(text: str) -> list[str]:
        live = re.sub(r"<!--.*?-->", "", text, flags=re.S)
        return (
            re.findall(r'<img[^>]+src="([^"]+)"', live)
            + re.findall(r'<a[^>]+href="([^"]+)"', live)
            + [url for _, url in re.findall(r"\[([^\]]*)\]\(([^)\s]+)\)", live)]
        )

    def test_nothing_in_it_is_relative(self) -> None:
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        relative = [
            url
            for url in self._references(text)
            if not url.startswith(("http://", "https://", "#", "mailto:"))
        ]
        assert not relative, relative

    def test_every_asset_it_points_at_is_in_the_tree(self) -> None:
        # absolute URLs cannot be checked by existence, so they are checked by
        # shape: each one has to name a file this repository actually carries
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        prefix = "https://raw.githubusercontent.com/sunnypatell/basalt/main/"
        assets = [u for u in self._references(text) if u.startswith(prefix)]
        assert assets
        for url in assets:
            assert (ROOT / url[len(prefix) :]).is_file(), url
