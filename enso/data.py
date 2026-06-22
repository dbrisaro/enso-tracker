"""
Fetches ENSO indicator data from NOAA CPC public endpoints.
All functions return pandas DataFrames indexed by date.
"""
import io
import re
import requests
import numpy as np
import pandas as pd

_TIMEOUT = 20

URLS = {
    "sst": "https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices",
    "oni": "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
    "soi": "https://www.cpc.ncep.noaa.gov/data/indices/soi",
}

# All confirmed El Niño events since 1950 with metadata for analogue ranking
ENSO_EVENTS = {
    "1957–58": {"onset_yr": 1957, "peak_oni": 1.7, "type": "EP", "color": "#7f8c8d"},
    "1963–64": {"onset_yr": 1963, "peak_oni": 1.2, "type": "EP", "color": "#95a5a6"},
    "1965–66": {"onset_yr": 1965, "peak_oni": 1.6, "type": "EP", "color": "#7f8c8d"},
    "1968–69": {"onset_yr": 1968, "peak_oni": 1.0, "type": "CP", "color": "#bdc3c7"},
    "1972–73": {"onset_yr": 1972, "peak_oni": 2.1, "type": "EP", "color": "#8e44ad"},
    "1976–77": {"onset_yr": 1976, "peak_oni": 0.8, "type": "EP", "color": "#bdc3c7"},
    "1982–83": {"onset_yr": 1982, "peak_oni": 2.2, "type": "EP", "color": "#6c3483"},
    "1986–87": {"onset_yr": 1986, "peak_oni": 1.6, "type": "EP", "color": "#27ae60"},
    "1991–92": {"onset_yr": 1991, "peak_oni": 1.7, "type": "EP", "color": "#16a085"},
    "1994–95": {"onset_yr": 1994, "peak_oni": 1.0, "type": "CP", "color": "#7f8c8d"},
    "1997–98": {"onset_yr": 1997, "peak_oni": 2.4, "type": "EP", "color": "#c0392b"},
    "2002–03": {"onset_yr": 2002, "peak_oni": 1.3, "type": "CP", "color": "#2980b9"},
    "2004–05": {"onset_yr": 2004, "peak_oni": 0.9, "type": "CP", "color": "#95a5a6"},
    "2006–07": {"onset_yr": 2006, "peak_oni": 1.1, "type": "EP", "color": "#f39c12"},
    "2009–10": {"onset_yr": 2009, "peak_oni": 1.3, "type": "CP", "color": "#3498db"},
    "2015–16": {"onset_yr": 2015, "peak_oni": 2.6, "type": "EP", "color": "#e07b39"},
    "2018–19": {"onset_yr": 2018, "peak_oni": 0.9, "type": "CP", "color": "#95a5a6"},
    "2023–24": {"onset_yr": 2023, "peak_oni": 2.0, "type": "EP", "color": "#d35400"},
    "2026–27": {"onset_yr": 2026, "peak_oni": None, "type": "EP", "color": "#1e6fa5"},
}


def _get(url: str) -> str:
    r = requests.get(url, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.text


def fetch_sst() -> pd.DataFrame:
    """
    Monthly Nino-region SST and anomalies from NOAA CPC (1950-present).
    Columns: nino12, nino12_anom, nino3, nino3_anom, nino4, nino4_anom, nino34, nino34_anom
    """
    raw = _get(URLS["sst"])
    rows = []
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) < 10 or not parts[0].isdigit():
            continue
        try:
            rows.append({
                "year": int(parts[0]), "month": int(parts[1]),
                "nino12":      float(parts[2]),  "nino12_anom":  float(parts[3]),
                "nino3":       float(parts[4]),  "nino3_anom":   float(parts[5]),
                "nino4":       float(parts[6]),  "nino4_anom":   float(parts[7]),
                "nino34":      float(parts[8]),  "nino34_anom":  float(parts[9]),
            })
        except (ValueError, IndexError):
            continue
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    return df.set_index("date").sort_index()


def fetch_oni() -> pd.DataFrame:
    """
    3-month running ONI from NOAA CPC (1950-present).
    File format: SEAS  YR  SST  ANOM  (4 columns; ANOM is the ONI value)
    Maps each season to its centre month: DJF->Jan, JFM->Feb, ... NDJ->Dec.
    Columns: season, year, oni
    """
    CENTRE = {
        "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4,
        "AMJ": 5, "MJJ": 6, "JJA": 7, "JAS": 8,
        "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
    }
    rows = []
    for line in _get(URLS["oni"]).splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0] in CENTRE:
            try:
                rows.append({
                    "season": parts[0], "year": int(parts[1]),
                    "oni": float(parts[3]), "month": CENTRE[parts[0]],
                })
            except ValueError:
                continue
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    return df.set_index("date")[["season", "year", "oni"]].sort_index()


def fetch_soi() -> pd.DataFrame:
    """
    Monthly standardised SOI from NOAA CPC (1951-present).
    Column: soi
    """
    rows = []
    for line in _get(URLS["soi"]).splitlines():
        line = line.strip()
        # Year is the first 4 characters; skip non-data lines
        if len(line) < 4 or not line[:4].isdigit():
            continue
        yr   = int(line[:4])
        rest = line[4:]
        # NOAA format omits spaces before negative values: "1.5-0.3" -> "1.5 -0.3"
        rest = re.sub(r"(\d)(-)", r"\1 \2", rest)
        for m, v in enumerate(rest.split()[:12], start=1):
            try:
                val = float(v)
                if abs(val) < 50:   # exclude fill values (-999.9 etc.)
                    rows.append({"year": yr, "month": m, "soi": val})
            except ValueError:
                continue
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    return df.set_index("date")[["soi"]].sort_index()


def aligned_series(series: pd.Series, onset_yr: int, n: int = 24) -> list:
    """
    Extract n months starting from July of (onset_yr - 1).
    Returns a list of floats (NaN where data is missing).
    """
    start = pd.Timestamp(onset_yr - 1, 7, 1)
    idx   = pd.date_range(start, periods=n, freq="MS")
    vals  = series.reindex(idx)
    return [None if np.isnan(v) else round(float(v), 2) for v in vals]
