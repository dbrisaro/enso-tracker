#!/usr/bin/env python3
"""Export the tracker charts as LinkedIn-ready PNGs.

Everything is rendered at deviceScaleFactor 3 on an opaque white background.
Fonts are enlarged on the figure before capture rather than by upscaling the
PNG afterwards, so axis labels stay sharp at the width each image is actually
viewed at (the feed shows inline images at roughly 550 px).
"""
import argparse
from datetime import datetime
from pathlib import Path

import kaleido
import pandas as pd
from PIL import Image

OUT_DIR = Path("/Users/daniela/Documents/Newsletter/ENSO blog")
SCALE   = 3

# LinkedIn renders a transparent PNG on a grey card, so every export is
# flattened onto white.
BG = (255, 255, 255)


def _restyle(fig, *, base, title, annot, margin):
    """Enlarge every font on the figure and widen the margins to match."""
    fig.update_layout(
        font=dict(size=base),
        title=dict(font=dict(size=title)),
        legend=dict(font=dict(size=base)),
        margin=margin,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    for axis in (fig.layout.xaxis, fig.layout.yaxis):
        axis.title.font.size = base
        axis.tickfont.size   = base
    # Threshold labels and the annotated peaks are layout annotations, and both
    # are set small for the web page.
    for a in fig.layout.annotations:
        a.font.size = annot
    return fig


def _dates_to_str(fig):
    """Serialise datetime coordinates as ISO strings.

    Kaleido 1.3.0 paired with Plotly 5.24.1 fails to serialise pandas
    Timestamps and writes no file at all, without raising, so any figure on a
    date axis has to be converted before export. Plotly still reads the axis as
    a date axis from the string form.
    """
    for tr in fig.data:
        if getattr(tr, "x", None) is not None and len(tr.x):
            if isinstance(tr.x[0], (pd.Timestamp, datetime)):
                tr.x = [pd.Timestamp(v).strftime("%Y-%m-%d") for v in tr.x]
    for a in fig.layout.annotations:
        if isinstance(a.x, (pd.Timestamp, datetime)):
            a.x = pd.Timestamp(a.x).strftime("%Y-%m-%d")
    fig.update_xaxes(type="date")
    return fig


def _write(fig, name, width, height, *, label):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    kaleido.write_fig_sync(
        fig, path=path,
        opts={"format": "png", "width": width, "height": height, "scale": SCALE},
    )
    if not path.exists():
        raise RuntimeError(
            f"kaleido wrote nothing for {name}. This is the silent Plotly 5.x / "
            f"Kaleido 1.x serialisation failure; run the figure through "
            f"_dates_to_str() if it uses a date axis.")
    # Flatten any alpha onto white and drop the alpha channel entirely.
    with Image.open(path) as im:
        flat = Image.new("RGB", im.size, BG)
        flat.paste(im, mask=im.split()[3] if im.mode == "RGBA" else None)
        flat.save(path, "PNG", optimize=True)
    with Image.open(path) as im:
        print(f"  {name:34s} {im.size[0]}x{im.size[1]}  {im.mode}  "
              f"{path.stat().st_size / 1024:6.0f} KB   {label}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-maps", action="store_true",
                    help="skip image 4, which downloads two ERSSTv5 granules")
    args = ap.parse_args()

    from enso.data import (fetch_sst, fetch_oni, fetch_roni, fetch_weekly_sst,
                           fetch_sst_anomaly_map, fetch_iri_plume)
    from enso.analogues import rank_analogues
    from enso import render

    print("Fetching...")
    sst    = fetch_sst()
    oni    = fetch_oni()
    weekly = fetch_weekly_sst()
    plume  = fetch_iri_plume()
    analogues = rank_analogues(sst["nino34_anom"])
    try:
        analogues_roni = rank_analogues(fetch_roni())
    except Exception as e:
        print(f"  RONI unavailable: {e}")
        analogues_roni = None

    show_oni = [r["event"] for r in analogues[:4]]
    pk = max((m, s) for m, s in zip(plume["oni_mean"], plume["seasons"])
             if m is not None)
    print(f"  monthly through {sst.index[-1]:%B %Y}, "
          f"weekly through {weekly.index[-1]:%d %b %Y}")
    print(f"  top ONI analogues: {', '.join(show_oni)}")
    print(f"  IRI plume issued {plume['issue_label']}, "
          f"{plume['n_oni_models']} models, peak {pk[0]:+.2f} in "
          f"{pk[1]['code']} {pk[1]['centre']:%Y}")
    print("\nWriting to", OUT_DIR)

    # --- Image 1, article cover -------------------------------------------
    # Strict 16:9. LinkedIn crops the article cover to that ratio, and a
    # different aspect would cut the peak annotations off.
    fig = render.chart_oni_full(oni, return_fig=True)
    _restyle(fig, base=26, title=38, annot=30,
             margin=dict(l=140, r=100, t=175, b=120))
    fig.update_layout(
        title=dict(text="ENSO since 1950: four events stand apart", x=0.5,
                   xanchor="center", y=0.955, yanchor="top"),
        legend=dict(orientation="h", yanchor="bottom", y=1.005,
                    xanchor="center", x=0.5),
    )
    _dates_to_str(fig)
    _write(fig, "01-cover-oni-1950-2026.png", 1920, 1080, label="16:9 article cover")

    # --- Image 2, Niño-1+2 ------------------------------------------------
    fig = render.chart_nino12(sst["nino12_anom"], show_events=show_oni, return_fig=True)
    _restyle(fig, base=23, title=31, annot=22,
             margin=dict(l=115, r=195, t=175, b=100))
    fig.update_layout(
        title=dict(y=0.975, yanchor="top", x=0.5, xanchor="center"),
        legend=dict(orientation="h", yanchor="bottom", y=1.015,
                    xanchor="center", x=0.5, font=dict(size=20)),
    )
    for tr in fig.data:
        if tr.name and "2026" in tr.name:
            tr.line.width = 7.5
            tr.line.color = "#0d3d66"
        elif tr.line is not None and tr.line.width:
            tr.line.width = 3
            tr.marker.size = 0
    _write(fig, "02-nino12-acceleration.png", 1200, 700, label="inline, free ratio")

    # --- Image 3, Niño-3.4 24-month trajectories --------------------------
    # Read at ~550 px in the feed, so the 2026 lines need to separate from the
    # analogue bundle on weight and hue, not just on dash pattern.
    # The analogues are demoted to a grey ramp for the feed exports. On the page
    # each event keeps its own colour for cross-chart consistency, but here the
    # 1997-98 red collides with the forecast red, and at 550 px the reader needs
    # 2026 and the plume to separate from the bundle instantly.
    # Every event colour on the page collides with something needed here, so the
    # analogues give up the colour channel entirely and separate on lightness
    # plus dash pattern, which survives the downscale to feed width.
    GREYS  = ["#333333", "#6b6b6b", "#8f8f8f", "#b0b0b0"]
    DASHES = ["solid", "dash", "dashdot", "longdash"]

    def _nino34(width, height, name, label):
        fig = render.chart_nino34(sst["nino34_anom"], show_events=show_oni,
                                  weekly_series=weekly["nino34_anom"],
                                  plume=plume, return_fig=True)
        _restyle(fig, base=23, title=31, annot=22,
                 margin=dict(l=115, r=205, t=215, b=100))
        fig.update_layout(
            title=dict(y=0.975, yanchor="top", x=0.5, xanchor="center"),
            legend=dict(orientation="h", yanchor="bottom", y=1.015,
                        xanchor="center", x=0.5, font=dict(size=20),
                        tracegroupgap=0),
        )
        style = iter(zip(GREYS, DASHES))
        for tr in fig.data:
            if tr.name and "weekly" in tr.name:
                tr.line.width = 4.5
                tr.line.color = "#e8590c"
                tr.line.dash  = "dot"
                tr.opacity    = 1.0
            elif tr.name and "monthly" in tr.name:
                tr.line.width = 7.5
                tr.line.color = "#0d3d66"
            elif tr.name and "IRI plume" in tr.name:
                tr.line.width  = 5.5
                tr.line.color  = "#b5179e"
                tr.line.dash   = "dash"
                tr.marker.size = 12
                tr.marker.color = "#b5179e"
            elif tr.name and "spread" in tr.name:
                tr.fillcolor = "rgba(181,23,158,0.11)"
            elif tr.line is not None and tr.line.width:
                col, dsh = next(style, ("#b0b0b0", "solid"))
                tr.line.width = 3.2
                tr.line.color = col
                tr.line.dash  = dsh
                tr.opacity    = 0.95
                tr.marker.size = 0
        return _write(fig, name, width, height, label=label)

    _nino34(1200, 700, "03-nino34-24month.png", label="inline, free ratio")
    _nino34(1200, 627, "05-nino34-native-post.png",
            label="1.91:1 for a native post")

    # --- Image 4, SST map composite --------------------------------------
    if args.skip_maps:
        print("  (image 4 skipped)")
    elif not analogues_roni:
        print("  image 4 skipped: no RONI ranking, so there is no RONI analogue")
    else:
        _sst_composite(sst, analogues, analogues_roni, fetch_sst_anomaly_map)


def _sst_composite(sst, analogues, analogues_roni, fetch_map):
    """Top ONI analogue against top RONI analogue, shared colour scale.

    Sized by figsize x dpi rather than cropped with bbox_inches="tight", so the
    output lands on exactly 1600x800 at scale 3 with nothing clipped.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    current = sst.index[-1]
    start   = pd.Timestamp(2025, 7, 1)
    offset  = (current.year - start.year) * 12 + current.month - start.month

    panels = []
    for tag, source in (("ONI", analogues), ("RONI", analogues_roni)):
        top   = source[0]
        adate = pd.Timestamp(top["onset_yr"] - 1, 7, 1) + pd.DateOffset(months=offset)
        panels.append({
            "tag": tag, "event": top["event"], "score": top["score"],
            "label": f"Top {tag} analogue: {top['event']}",
            "sub":   f"{adate:%B %Y}   similarity {top['score']:.2f}",
            "data":  fetch_map(int(adate.year), int(adate.month)),
        })
        print(f"  image 4 {tag}: {top['event']} at {adate:%B %Y}")

    proj, data_crs = ccrs.PlateCarree(central_longitude=180), ccrs.PlateCarree()
    norm = mcolors.TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)

    # Two panels of 180 deg by 70 deg sit at an aspect of about 2.57 each. Side
    # by side across 1600 px they can only be about 320 px tall, so a 1600x800
    # frame would letterbox them inside two thick white bands. The height is
    # therefore derived from the panel geometry plus the fixed furniture.
    EXTENT = [110, 290, -35, 35]
    W      = 1600
    left, right, gap = 0.035, 0.985, 0.035
    pw     = (right - left - gap) / 2
    aspect = (EXTENT[1] - EXTENT[0]) / (EXTENT[3] - EXTENT[2])
    px_w   = pw * W
    px_h   = px_w / aspect
    # furniture: panel title + subtitle above, colourbar + source note below
    TOP_PX, BOT_PX = 96, 196
    H = int(round(px_h + TOP_PX + BOT_PX))

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100 * SCALE, facecolor="white")
    ax_bottom = BOT_PX / H
    ax_height = px_h / H

    im = None
    for i, p in enumerate(panels):
        ax = fig.add_axes([left + i * (pw + gap), ax_bottom, pw, ax_height],
                          projection=proj)
        ax.set_extent(EXTENT, crs=data_crs)
        im = ax.pcolormesh(p["data"]["lon"], p["data"]["lat"], p["data"]["ssta"],
                           transform=data_crs, cmap=plt.cm.RdBu_r, norm=norm,
                           rasterized=True, zorder=0)
        ax.add_feature(cfeature.LAND, facecolor="#D4C5A0", zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="#666666", zorder=2)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray",
                          alpha=0.5, zorder=3)
        gl.top_labels = gl.right_labels = False
        gl.left_labels = (i == 0)
        gl.xlabel_style = gl.ylabel_style = {"size": 9}
        ax.set_title(p["label"], fontsize=16, fontweight="bold", pad=26)
        ax.text(0.5, 1.015, p["sub"], transform=ax.transAxes, ha="center",
                va="bottom", fontsize=11, color="#444444")

    cax = fig.add_axes([0.30, 92 / H, 0.40, 24 / H])
    cb  = fig.colorbar(im, cax=cax, orientation="horizontal", extend="both")
    # Label above the bar, ticks below it, so neither can land on the source note.
    cb.ax.xaxis.set_label_position("top")
    cb.set_label("SST anomaly (°C), shared scale", fontsize=11.5, labelpad=9)
    cb.ax.tick_params(labelsize=10.5)
    fig.text(0.5, 22 / H, "ERSSTv5 anomaly relative to 1991-2020. Source: NOAA NCEI.",
             ha="center", fontsize=10, color="#666666")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "04-sst-maps-oni-vs-roni.png"
    fig.savefig(path, format="png", facecolor="white", dpi=100 * SCALE)
    print(f"  image 4 frame fitted to {W}x{H} at 1x "
          f"(a 1600x800 frame would letterbox the panels)")
    plt.close(fig)
    with Image.open(path) as raw:
        flat = Image.new("RGB", raw.size, BG)
        flat.paste(raw, mask=raw.split()[3] if raw.mode == "RGBA" else None)
        flat.save(path, "PNG", optimize=True)
    with Image.open(path) as chk:
        print(f"  {path.name:34s} {chk.size[0]}x{chk.size[1]}  {chk.mode}  "
              f"{path.stat().st_size / 1024:6.0f} KB   side-by-side composite")


if __name__ == "__main__":
    main()
