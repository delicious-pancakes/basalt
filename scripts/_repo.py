# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Make a script import the working tree, and say which tree that was.

These scripts produce the numbers the README quotes. That makes the question
"which copy of basalt produced this?" part of the result rather than a detail
of how the machine happens to be set up.

It is not a hypothetical. An editable install pointing at a stale clone
elsewhere on the disk shadowed the working tree for fifty-four commits, and
because the sweep printed only its verdicts there was nothing in the output to
show it. Every run since prints its own provenance.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"


def use_repo_source() -> None:
    """Put this checkout's `src` ahead of any installed copy.

    Call before importing basalt.
    """
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))


def provenance() -> str:
    """One line naming the source tree and commit under test."""
    import basalt

    origin = Path(basalt.__file__).resolve().parent.parent
    try:
        described = subprocess.run(
            ["git", "-C", str(ROOT), "describe", "--always", "--dirty"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        described = "unknown"
    # loud, because a mismatch here invalidates every number below it
    marker = "" if origin == SRC else f"  [!] not {SRC}"
    return f"basalt {described} from {origin}{marker}"
