#!/usr/bin/env python3
import argparse
import pandas as pd
import shutil
import sys
from pathlib import Path


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

    latest = sst.iloc[-1]
    print(f"Nino-3.4 : {latest['nino34_anom']:+.2f} C")
    print(f"Nino-1+2 : {latest['nino12_anom']:+.2f} C")
    print(f"Nino-4   : {latest['nino4_anom']:+.2f} C")
    print(f"SOI      : {soi_df['soi'].iloc[-1]:+.1f}")
    print(f"As of    : {sst.index[-1].strftime('%B %Y')}")

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
                      analogues_roni=analogues_roni)

    out = Path("output/enso_tracker.html")
    out.parent.mkdir(exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Written -> {out}  ({len(html) // 1024} KB)")

    if args.deploy:
        repo = Path(args.deploy).expanduser()
        if not repo.is_dir():
            print(f"Deploy path not found: {repo}", file=sys.stderr)
            sys.exit(1)
        dest = repo / "portfolio" / "enso-tracker" / "index.html"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(out, dest)
        print(f"Deployed -> {dest}")


if __name__ == "__main__":
    main()
