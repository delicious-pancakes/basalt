#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Rebuild the ISA database into a scratch file and diff it against the committed one.

The database is generated data that is committed, so a consumer needs no harvest.
That trade only works if drift is loud, and loud means one command rather than
two that have to be run in the right order with a temporary path threaded between
them.

    python scripts/rebuild_and_compare.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import _repo

_repo.use_repo_source()

ROOT = _repo.ROOT


def main() -> int:
    scratch = Path(os.environ.get("TMP", "/tmp")) / "basalt-rebuilt-isa.json"
    build = subprocess.run(
        [sys.executable, "-m", "basalt.cli", "build-isa", "-o", str(scratch)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        sys.stderr.write(build.stdout[-2000:] + build.stderr[-2000:])
        return build.returncode
    print(next((ln for ln in build.stdout.splitlines() if ln.startswith("database:")), ""))
    return subprocess.run(
        [sys.executable, "scripts/compare_isa.py", "data/isa/sm_120a.json", str(scratch)],
        cwd=str(ROOT),
    ).returncode


if __name__ == "__main__":
    sys.exit(main())
