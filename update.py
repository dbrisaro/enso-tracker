#!/usr/bin/env python3
import argparse
import pandas as pd
import shutil
import sys
from pathlib import Path


_DEFAULT_FRONT_MATTER = """---
layout: null
title: "ENSO 2026 Tracker"
excerpt: "Live tracker comparing 2026 El Niño conditions against every major ENSO event since 1950: Niño-3.4 SST anomaly, SOI, Kelvin waves, and analogue rankings updated with NOAA CPC data."
date: 2026-07-13
permalink: /portfolio/enso-tracker/
category: "Data Science"
tags:
  - Climate
  - ENSO
  - El Niño
  - SST
  - NOAA
  - Data Visualization
redirect_from:
  - /enso_tracker.html
---"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy", metavar="JEKYLL_REPO")
    args = parser.parse_args()

    try:
        from enso.data import fetch_sst, fetch_soi, fetch_oni
        sst    = fetch_sst()
        soi_df = fetch_soi()
        oni    = fetch_oni()
    except Exception as e:
        print(f"Data fetch failed: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        from enso.data import fetch_weekly_sst
        weekly = fetch_weekly_sst()
    except Exception as e:
        print(f"Weekly SST fetch skipped: {e}")
        weekly = None

    try:
        from enso.data import fetch_iri_plume
        plume = fetch_iri_plume()
    except Exception as e:
        print(f"IRI plume fetch skipped: {e}")
        plume = None

    try:
        from enso.data import fetch_mei
        mei = fetch_mei()
    except Exception as e:
        print(f"MEI fetch skipped: {e}")
        mei = None

    latest = sst.iloc[-1]
    print(f"Nino-3.4 : {latest['nino34_anom']:+.2f} C")
    print(f"Nino-1+2 : {latest['nino12_anom']:+.2f} C")
    print(f"Nino-4   : {latest['nino4_anom']:+.2f} C")
    print(f"SOI      : {soi_df['soi'].iloc[-1]:+.1f}")
    if plume is not None:
        pk = max((m, s) for m, s in zip(plume["oni_mean"], plume["seasons"])
                 if m is not None)
        print(f"IRI plume: issued {plume['issue_label']}, "
              f"{plume['n_oni_models']} ONI models, peak {pk[0]:+.2f} C "
              f"in {pk[1]['code']} {pk[1]['centre']:%Y}")
    if mei is not None:
        md = mei.index[-1]
        seas = f"{(md - pd.DateOffset(months=1)).strftime('%b')}-{md.strftime('%b %Y')}"
        print(f"MEI.v2   : {mei.iloc[-1]:+.2f}  ({seas})")
    print(f"As of    : {sst.index[-1].strftime('%B %Y')} (monthly)")
    if weekly is not None:
        wk = weekly.iloc[-1]
        print(f"Weekly   : {weekly.index[-1].strftime('%d %b %Y')}  "
              f"N3.4 {wk['nino34_anom']:+.2f}  N1+2 {wk['nino12_anom']:+.2f}  "
              f"N3 {wk['nino3_anom']:+.2f}  N4 {wk['nino4_anom']:+.2f}")

    from enso.analogues import rank_analogues
    from enso.data import fetch_roni
    analogues      = rank_analogues(sst["nino34_anom"])
    try:
        roni           = fetch_roni()
        analogues_roni = rank_analogues(roni)
    except Exception as e:
        print(f"RONI fetch skipped: {e}")
        analogues_roni = None
    for r in analogues[:3]:
        print(f"  ONI #{r['rank']}  {r['event']}  score={r['score']:.3f}")
    if analogues_roni:
        for r in analogues_roni[:3]:
            print(f"  RONI #{r['rank']}  {r['event']}  score={r['score']:.3f}")

    from enso.data import fetch_sst_anomaly_map
    current_date  = sst.index[-1]
    current_start = pd.Timestamp(2025, 7, 1)
    offset = (current_date.year - current_start.year) * 12 + current_date.month - current_start.month

    def _build_sst_maps(source_analogues):
        maps = [{"data": None, "label": current_date.strftime("%B %Y") + " (2026)"}]
        for r in source_analogues[:2]:
            adate = pd.Timestamp(r["onset_yr"] - 1, 7, 1) + pd.DateOffset(months=offset)
            maps.append({
                "data": None,
                "label": adate.strftime("%B %Y") + f" ({r['event']})",
                "year": int(adate.year),
                "month": int(adate.month),
            })
        return maps

    sst_maps_oni  = _build_sst_maps(analogues)
    sst_maps_roni = _build_sst_maps(analogues_roni) if analogues_roni else None

    try:
        current_data = fetch_sst_anomaly_map(current_date.year, current_date.month)
        sst_maps_oni[0]["data"] = current_data
        for m in sst_maps_oni[1:]:
            m["data"] = fetch_sst_anomaly_map(m["year"], m["month"])
        print("SST maps (ONI): " + " | ".join(m["label"] for m in sst_maps_oni))
        if sst_maps_roni:
            sst_maps_roni[0]["data"] = current_data
            for m in sst_maps_roni[1:]:
                m["data"] = fetch_sst_anomaly_map(m["year"], m["month"])
            print("SST maps (RONI): " + " | ".join(m["label"] for m in sst_maps_roni))
    except Exception as e:
        print(f"SST map fetch skipped: {e}")
        sst_maps_oni = sst_maps_roni = None

    from enso.render import build_html
    cdn = bool(args.deploy)
    html = build_html(sst, soi_df, oni, analogues, cdn=cdn,
                      sst_maps_oni=sst_maps_oni, sst_maps_roni=sst_maps_roni,
                      analogues_roni=analogues_roni, weekly=weekly, mei=mei, plume=plume)

    out = Path("output/enso_tracker.html")
    out.parent.mkdir(exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Written -> {out}  ({len(html) // 1024} KB)")

    if args.deploy:
        repo = Path(args.deploy).expanduser()
        if not repo.is_dir():
            print(f"Deploy path not found: {repo}", file=sys.stderr)
            sys.exit(1)
        # The site serves this page from a Jekyll collection file that carries
        # YAML front matter (layout, permalink, tags, ...). Preserve that front
        # matter and only swap in the freshly generated HTML body.
        dest = repo / "_portfolio" / "enso-tracker.html"
        if dest.exists():
            old = dest.read_text(encoding="utf-8")
            if old.startswith("---"):
                front = old[:old.index("\n---", 3) + 4]
            else:
                front = _DEFAULT_FRONT_MATTER
        else:
            front = _DEFAULT_FRONT_MATTER
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(front + "\n" + html, encoding="utf-8")
        print(f"Deployed -> {dest}")


if __name__ == "__main__":
    main()
