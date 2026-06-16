#!/usr/bin/env python3
"""
Fetch live ENSO data from NOAA, rank historical analogues, and generate
a self-contained HTML page ready to deploy to danielarisaro.ar.

Usage
-----
  python update.py                                    # writes output/enso_tracker.html
  python update.py --deploy ~/repos/dbrisaro.github.io  # also copies to Jekyll repo root
"""
import argparse
import shutil
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Generate ENSO 2026 tracker page")
    parser.add_argument(
        "--deploy", metavar="JEKYLL_REPO",
        help="Path to the dbrisaro.github.io repo — copies enso_tracker.html there",
    )
    args = parser.parse_args()

    print("📡  Fetching NOAA data…")
    try:
        from enso.data import fetch_sst, fetch_soi, fetch_oni
        sst    = fetch_sst()
        soi_df = fetch_soi()
        oni    = fetch_oni()
    except Exception as e:
        print(f"❌  Data fetch failed: {e}", file=sys.stderr)
        sys.exit(1)

    latest = sst.iloc[-1]
    print(f"    Niño-3.4 : {latest['nino34_anom']:+.2f} °C")
    print(f"    Niño-1+2 : {latest['nino12_anom']:+.2f} °C")
    print(f"    Niño-4   : {latest['nino4_anom']:+.2f} °C")
    print(f"    SOI      : {soi_df['soi'].iloc[-1]:+.1f}")
    print(f"    As of    : {sst.index[-1].strftime('%B %Y')}")

    print("\n🔍  Computing historical analogues…")
    from enso.analogues import rank_analogues
    analogues = rank_analogues(sst["nino34_anom"])
    print("    Top 3 analogues:")
    for r in analogues[:3]:
        print(f"      #{r['rank']}  {r['event']}  (score {r['score']:.3f}, peak ONI {r['peak_oni']} °C, {r['type']}-type)")

    print("\n🏗️   Generating HTML…")
    from enso.render import build_html
    cdn = bool(args.deploy)   # inline for local preview, CDN for deploy
    html = build_html(sst, soi_df, oni, analogues, cdn=cdn)

    out = Path("output/enso_tracker.html")
    out.parent.mkdir(exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"✅  Written → {out}  ({len(html) // 1024} KB)")

    if args.deploy:
        repo = Path(args.deploy).expanduser()
        if not repo.is_dir():
            print(f"❌  Deploy path not found: {repo}", file=sys.stderr)
            sys.exit(1)
        dest = repo / "enso_tracker.html"
        shutil.copy(out, dest)
        print(f"✅  Deployed → {dest}")
        print()
        print("    Next steps:")
        print(f"      cd {repo}")
        print("      git add enso_tracker.html _portfolio/enso-tracker.md")
        print('      git commit -m "update ENSO tracker"')
        print("      git push")
    else:
        print()
        print("    To deploy directly:")
        print("      python update.py --deploy ~/path/to/dbrisaro.github.io")


if __name__ == "__main__":
    main()
