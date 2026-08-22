#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Compare two ISA databases and fail if the committed one has drifted.

The database is generated data that is nonetheless committed, because a
consumer of basalt should not have to run a harvest to get an assembler. That
trade only works if drift is loud. This is what makes it loud.

Encodings are not promised to be stable across ptxas releases, so a difference
is not automatically a bug. It is automatically something a human should look at.

    python scripts/compare_isa.py data/isa/sm_120a.json /tmp/rebuilt.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        raise SystemExit(f"error: {path} does not exist") from None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("committed", type=Path)
    ap.add_argument("rebuilt", type=Path)
    ap.add_argument(
        "--allow-new",
        action="store_true",
        help="treat forms present only in the rebuild as informational rather than a failure",
    )
    args = ap.parse_args()

    a, b = load(args.committed), load(args.rebuilt)
    fa: dict = a.get("forms", {})
    fb: dict = b.get("forms", {})

    if a.get("cuda_version") != b.get("cuda_version"):
        print(
            f"note: built with different compilers, {a.get('cuda_version')} vs "
            f"{b.get('cuda_version')}; differences below may be legitimate"
        )

    missing = sorted(set(fa) - set(fb))
    added = sorted(set(fb) - set(fa))
    changed = sorted(
        name
        for name in set(fa) & set(fb)
        if fa[name].get("encoding") != fb[name].get("encoding")
        or fa[name].get("operands") != fb[name].get("operands")
    )

    print(f"committed {len(fa)} forms, rebuilt {len(fb)} forms")
    for label, names in (
        ("missing from rebuild", missing),
        ("new in rebuild", added),
        ("changed", changed),
    ):
        if names:
            print(f"\n{label} ({len(names)}):")
            for n in names[:40]:
                print(f"  {n}")
            if len(names) > 40:
                print(f"  ... and {len(names) - 40} more")

    # a form disappearing or an encoding moving is a real signal; new forms are
    # usually just a wider corpus and are opt-in as a failure
    bad = bool(missing or changed) or (bool(added) and not args.allow_new)
    if bad:
        print(
            "\nthe committed database no longer matches a fresh build.\n"
            "if this is expected, rebuild and commit it:\n"
            "  python -m basalt.cli build-isa",
            file=sys.stderr,
        )
        return 1

    print("\nno drift")
    return 0


if __name__ == "__main__":
    sys.exit(main())
