#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Draw the header each long document wears, from one template.

`docs/assets/social-preview.svg` establishes the look: basalt cools into
interlocking hexagonal columns of near-uniform width, which is the same shape as
a fixed-width instruction stream. The long documents borrow it so the repository
reads as one thing rather than a pile of files.

Nothing here may go stale, which is the whole reason it is generated rather than
drawn. A header carries a document's *contract*, never a measurement: "every
number has a command that reproduces it" stays true on any day, where "8,554 of
8,584" is wrong within the hour. Anything with a digit in it belongs in the
document, under a command that regenerates it.

Adding a document is one row in HEADERS.

    python scripts/render_headers.py
"""

from __future__ import annotations

import sys

import _repo

ROOT = _repo.ROOT
ASSETS = ROOT / "docs" / "assets"

# name -> (title, the document's contract, one word of context)
HEADERS: dict[str, tuple[str, str, str]] = {
    "findings": (
        "Findings",
        "What the silicon does, and the command that reproduces each answer.",
        "MEASURED",
    ),
    "method": (
        "Method",
        "How every claim here is made, and what would falsify it.",
        "REPRODUCIBLE",
    ),
    "roadmap": (
        "Roadmap",
        "What is done, what is not, and what is deliberately out of scope.",
        "HONEST",
    ),
}

TEMPLATE = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 200"
     width="1280" height="200" role="img" aria-label="basalt {title}. {summary}">
  <title>basalt {title}</title>

  <defs>
    <pattern id="cols" width="112" height="130" patternUnits="userSpaceOnUse"
             patternTransform="translate(-18 -46)">
      <g fill="#151b23" stroke="#2b3542" stroke-width="1.4">
        <path d="M56 2 L106 30.5 L106 87.5 L56 116 L6 87.5 L6 30.5 Z"/>
        <path d="M0 67 L50 95.5 L50 152.5 L0 181 L-50 152.5 L-50 95.5 Z"/>
        <path d="M112 67 L162 95.5 L162 152.5 L112 181 L62 152.5 L62 95.5 Z"/>
      </g>
    </pattern>

    <linearGradient id="fade" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0.30" stop-color="#000000"/>
      <stop offset="0.66" stop-color="#5c5c5c"/>
      <stop offset="0.96" stop-color="#cfcfcf"/>
    </linearGradient>
    <mask id="fadeMask"><rect width="1280" height="200" fill="url(#fade)"/></mask>

    <radialGradient id="glow" cx="0.16" cy="0.5" r="0.9">
      <stop offset="0" stop-color="#1b2531" stop-opacity="0.9"/>
      <stop offset="1" stop-color="#0d1117" stop-opacity="0"/>
    </radialGradient>

    <linearGradient id="accent" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#a6e831"/>
      <stop offset="1" stop-color="#76b900"/>
    </linearGradient>

    <style>
      .sans {{ font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, Inter,
               Helvetica, Arial, sans-serif; }}
      .mono {{ font-family: ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas,
               "Liberation Mono", monospace; }}
      .title {{ font-weight: 700; font-size: 52px; letter-spacing: -1.6px; fill: #e6edf3; }}
      .lede  {{ font-weight: 400; font-size: 19px; fill: #8b949e; }}
      .mark  {{ font-weight: 700; font-size: 19px; letter-spacing: -0.6px; fill: #5c7086; }}
      .tag   {{ font-size: 12px; fill: #6e7781; letter-spacing: 1.5px; }}
    </style>
  </defs>

  <rect width="1280" height="200" fill="#0d1117"/>
  <rect width="1280" height="200" fill="url(#glow)"/>
  <rect width="1280" height="200" fill="url(#cols)" mask="url(#fadeMask)"/>

  <rect x="0" y="0" width="6" height="200" fill="url(#accent)"/>

  <text x="52" y="86" class="sans title">{title}</text>
  <text x="52" y="124" class="sans lede">{summary}</text>

  <text x="52" y="164" class="mono tag">{tag}  &#183;  sm_120  &#183;  RTX 50 SERIES</text>
  <text x="1228" y="164" class="sans mark" text-anchor="end">basalt</text>
</svg>
"""


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for name, (title, summary, tag) in HEADERS.items():
        if any(character.isdigit() for character in summary):
            print(f"error: {name} header carries a number and will go stale", file=sys.stderr)
            return 1
        path = ASSETS / f"header-{name}.svg"
        path.write_text(TEMPLATE.format(title=title, summary=summary, tag=tag), encoding="utf-8")
        print(f"  {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
