#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Rasterise the SVG assets to PNG.

The artwork is authored as SVG and that is what the README embeds, because it
stays crisp at any size and can be edited as text. GitHub's repository social
preview slot only accepts a raster image, so this renders the same source to
PNG at the 1280x640 the setting expects.

Rendering goes through headless Chrome rather than a Python SVG library: the
artwork uses masks, `mix-blend-mode` and a system font stack, and the browser is
the only renderer guaranteed to agree with what a viewer will actually see.

    python scripts/render_assets.py
    python scripts/render_assets.py --chrome "/path/to/chrome"
"""

from __future__ import annotations

import argparse
import http.server
import shutil
import socket
import socketserver
import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "assets"

# render at 4x and downsample: 16 samples a pixel beats the browser's own
# antialiasing on the small type, and the smoother result compresses smaller too
SCALE = 4
CAP_BYTES = 1_000_000

# well under the 1 MB cap, so the artwork can gain an element without failing
TARGET_BYTES = 600_000

# (source svg, output png, css width, css height)
TARGETS: tuple[tuple[str, Path, int, int], ...] = (
    ("social-preview.svg", ROOT / ".github" / "social-preview.png", 1280, 640),
)

CHROME_CANDIDATES = (
    "C:/Program Files/Google/Chrome/Application/chrome.exe",
    "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
    "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
)


def find_chrome(explicit: str | None) -> str:
    for cand in filter(None, (explicit, *CHROME_CANDIDATES)):
        if Path(cand).is_file():
            return cand
    if found := shutil.which("chrome") or shutil.which("chromium") or shutil.which("google-chrome"):
        return found
    raise SystemExit(
        "no Chrome or Chromium found. pass --chrome /path/to/binary, or install one; "
        "the SVG sources in docs/assets are the real artefacts and render fine in any browser."
    )


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@contextmanager
def serve(directory: Path):
    """Serve `directory` on localhost.

    Chrome refuses `file:` URLs for anything that loads sub-resources, and the
    SVG references its own gradients and patterns, so a real origin is simpler
    than fighting the flag matrix.
    """

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(directory), **kw)

        def log_message(self, *a):  # keep the script's output about the render
            pass

    port = free_port()
    with socketserver.TCPServer(("127.0.0.1", port), Handler) as httpd:
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            yield port
        finally:
            httpd.shutdown()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--chrome", default=None, help="path to a Chrome or Chromium binary")
    args = ap.parse_args()

    chrome = find_chrome(args.chrome)
    print(f"renderer: {chrome}")

    with serve(ASSETS) as port:
        for svg, out, w, h in TARGETS:
            if not (ASSETS / svg).is_file():
                raise SystemExit(f"missing source: {(ASSETS / svg).relative_to(ROOT)}")
            out.parent.mkdir(parents=True, exist_ok=True)
            proc = subprocess.run(
                [
                    chrome,
                    "--headless",
                    "--disable-gpu",
                    "--hide-scrollbars",
                    f"--force-device-scale-factor={SCALE}",
                    f"--window-size={w},{h}",
                    f"--screenshot={out}",
                    f"http://127.0.0.1:{port}/{svg}",
                ],
                capture_output=True,
                timeout=120,
            )
            if not out.is_file():
                sys.stderr.write(proc.stderr.decode("utf-8", "replace"))
                raise SystemExit(f"chrome produced no output for {svg}")

            px, mode = optimise(out, w, h)
            print(
                f"  {svg} -> {out.relative_to(ROOT)}  {px}  "
                f"{out.stat().st_size / 1024:.0f} KB  ({mode})"
            )

    print("\nupload .github/social-preview.png under Settings > General > Social preview")
    return 0


def optimise(path: Path, width: int, height: int) -> tuple[str, str]:
    """Write the exact form GitHub's social preview pipeline hands back.

    GitHub re-encodes whatever it accepts, to 1280x640 8-bit truecolour with no
    palette and no alpha. Handing it that form already means nothing is resized
    or converted on the way in, so the upload has one less step to fail at and
    the bytes that get served are the bytes that were reviewed here.

    Chrome renders at SCALE and the file is saved at 1x: sharper than letting
    the browser antialias at 1x, and a fraction of the size of shipping the
    supersample.
    """
    try:
        from PIL import Image
    except ImportError:
        return "unscaled", "no Pillow, left as rendered"

    with Image.open(path) as raw:
        # RGB, not RGBA and not P: an alpha channel composites against whatever
        # the unfurling client uses for a background, which is not a choice to
        # leave to Slack
        im = raw.convert("RGB")
        if (im.width, im.height) != (width, height):
            im = im.resize((width, height), Image.Resampling.LANCZOS)
        im.save(path, "PNG", optimize=True, compress_level=9)

    size = path.stat().st_size
    px = f"{width}x{height}"
    if size > CAP_BYTES:
        return px, "over GitHub's 1 MB cap; simplify the artwork"
    if size > TARGET_BYTES:
        return px, "under GitHub's cap, over the comfort target"
    return px, "truecolour, the form GitHub serves"


if __name__ == "__main__":
    raise SystemExit(main())
