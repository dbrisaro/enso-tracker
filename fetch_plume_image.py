#!/usr/bin/env python3
"""Fetch the CCSR/IRI ENSO plume figure and render it to PNG.

Background: the machine-readable plume feed (plumes_json) started returning
403 in August 2026. The block is path-specific -- IRI's own page still draws
the plume from ensoforecast2.iri.columbia.edu/figure4_plot/{year}/{month},
which serves the figure as a Matplotlib SVG and answers normally as long as
the request carries browser headers. That endpoint is the fallback used here.

The SVG is vector, so the PNG is rendered rather than scraped: headless Chrome
is the default engine because it needs no install on macOS, with rsvg-convert
and cairosvg tried first if either happens to be present.

Usage:
    python fetch_plume_image.py                 # latest issued plume
    python fetch_plume_image.py --month 2026-07 # the plume issued July 2026
    python fetch_plume_image.py --width 2200    # wider render
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from enso.plume_svg import fetch_plume_svg, parse_plume_svg

CHROME = ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
          "/Applications/Chromium.app/Contents/MacOS/Chromium",
          "google-chrome", "chromium")



def _svg_size(svg: str) -> tuple:
    """Intrinsic width/height in points, defaulting to the figure's usual shape."""
    try:
        root = ET.fromstring(svg)
        w = float(root.get("width", "0").rstrip("pt"))
        h = float(root.get("height", "0").rstrip("pt"))
        if w and h:
            return w, h
    except (ET.ParseError, ValueError):
        pass
    return 933.8, 542.6


def render_png(svg: str, dest: Path, width: int = 1900) -> str:
    """Rasterise the SVG to dest at the given pixel width. Returns the engine."""
    w_pt, h_pt = _svg_size(svg)
    height = max(1, round(width * h_pt / w_pt))

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "plume.svg"
        src.write_text(svg, encoding="utf-8")

        if shutil.which("rsvg-convert"):
            subprocess.run(["rsvg-convert", "-w", str(width), "-h", str(height),
                            "-b", "white", "-o", str(dest), str(src)], check=True)
            return "rsvg-convert"

        try:
            import cairosvg
            cairosvg.svg2png(bytestring=svg.encode(), write_to=str(dest),
                             output_width=width, output_height=height,
                             background_color="white")
            return "cairosvg"
        except ImportError:
            pass

        chrome = next((c for c in CHROME if Path(c).exists() or shutil.which(c)),
                      None)
        if chrome is None:
            raise RuntimeError("No SVG renderer found (rsvg-convert, cairosvg, "
                               "or Chrome).")

        # Screenshot an HTML wrapper rather than the SVG directly: the wrapper
        # pins the image to an exact pixel box, so the shot has no stray margin
        # and the window size does not have to guess at the intrinsic pt->px
        # conversion.
        page = Path(tmp) / "plume.html"
        page.write_text(
            "<style>html,body{margin:0;padding:0;background:#fff}"
            f"img{{display:block;width:{width // 2}px}}</style>"
            f'<img src="{src.as_uri()}">', encoding="utf-8")
        subprocess.run(
            [chrome, "--headless", "--disable-gpu", "--hide-scrollbars",
             "--force-device-scale-factor=2",
             "--default-background-color=FFFFFFFF",
             f"--window-size={width // 2},{height // 2}",
             f"--screenshot={dest}", page.as_uri()],
            check=False, capture_output=True)
        if not dest.exists() or dest.stat().st_size == 0:
            raise RuntimeError("Chrome produced no screenshot.")
        return "chrome"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--month", metavar="YYYY-MM",
                   help="issue month of the plume (default: latest issued)")
    p.add_argument("--width", type=int, default=1900,
                   help="PNG width in pixels (default: 1900)")
    p.add_argument("--out", default="output/figures/8_plume.png",
                   help="PNG path (the SVG is written alongside it)")
    args = p.parse_args()

    year = month = None
    if args.month:
        stamp = pd.Timestamp(args.month + "-01")
        year, month = stamp.year, stamp.month

    try:
        fig = fetch_plume_svg(year, month)
    except Exception as e:
        print(f"Plume figure fetch failed: {e}", file=sys.stderr)
        sys.exit(1)

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    svg_dest = dest.with_suffix(".svg")
    svg_dest.write_text(fig["svg"], encoding="utf-8")

    try:
        engine = render_png(fig["svg"], dest, args.width)
    except Exception as e:
        print(f"SVG saved -> {svg_dest}")
        print(f"PNG render failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Title    : {fig['title']}")
    print(f"Issued   : {fig['issue_label']}")
    print(f"Source   : {fig['url']}")
    print(f"SVG      -> {svg_dest}  ({len(fig['svg']) // 1024} KB)")
    print(f"PNG      -> {dest}  ({dest.stat().st_size // 1024} KB, via {engine})")

    # The same figure is the tracker's data fallback, so report what it parses
    # to -- a silent shape change upstream shows up here first.
    try:
        plume = parse_plume_svg(fig["svg"], fig["issue_label"])
    except Exception as e:
        print(f"Ensemble parse failed: {e}", file=sys.stderr)
        return
    peak = max((m, s) for m, s in zip(plume["oni_mean"], plume["seasons"])
               if m is not None)
    print(f"Ensemble : {plume['n_oni_models']} models, peak {peak[0]:+.2f} C "
          f"in {peak[1]['code']} {peak[1]['centre']:%Y}")


if __name__ == "__main__":
    main()
