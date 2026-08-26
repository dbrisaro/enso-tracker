import io
import json
import os
import re
import tempfile
import requests
import numpy as np
import pandas as pd

_TIMEOUT = 20

URLS = {
    "sst": "https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices",
    "oni": "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
    "soi": "https://www.cpc.ncep.noaa.gov/data/indices/soi",
    # 1991-2020 base period, matching sstoi.indices. The wksst8110.for file
    # (1981-2010 base) is frozen at January 2021 and must not be used.
    "wksst": "https://www.cpc.ncep.noaa.gov/data/indices/wksst9120.for",
    "mei": "https://psl.noaa.gov/enso/mei/data/meiv2.data",
    "plume": "https://ensoforecast.iri.columbia.edu/plumes_json/{year}/{month}",
}

# Three-month season codes indexed by their STARTING month, so index 0 is the
# season beginning in January.
SEASON_BY_START = ["JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA",
                   "JAS", "ASO", "SON", "OND", "NDJ", "DJF"]

# MEI.v2 begins with the Dec 1978 - Jan 1979 season, so events whose aligned
# window opens before July 1979 have no MEI to plot.
MEI_START_YEAR = 1979

ENSO_EVENTS = {
    "1957–58": {"onset_yr": 1957, "peak_oni": 1.7, "type": "EP", "color": "#7f8c8d"},
    "1963–64": {"onset_yr": 1963, "peak_oni": 1.2, "type": "EP", "color": "#95a5a6"},
    "1965–66": {"onset_yr": 1965, "peak_oni": 1.6, "type": "EP", "color": "#7f8c8d"},
    "1968–69": {"onset_yr": 1968, "peak_oni": 1.0, "type": "CP", "color": "#bdc3c7"},
    "1972–73": {"onset_yr": 1972, "peak_oni": 2.1, "type": "EP", "color": "#8e44ad"},
    "1976–77": {"onset_yr": 1976, "peak_oni": 0.8, "type": "EP", "color": "#bdc3c7"},
    "1982–83": {"onset_yr": 1982, "peak_oni": 2.2, "type": "EP", "color": "#8e44ad"},
    "1986–87": {"onset_yr": 1986, "peak_oni": 1.6, "type": "EP", "color": "#27ae60"},
    "1991–92": {"onset_yr": 1991, "peak_oni": 1.7, "type": "EP", "color": "#16a085"},
    "1994–95": {"onset_yr": 1994, "peak_oni": 1.0, "type": "CP", "color": "#7f8c8d"},
    "1997–98": {"onset_yr": 1997, "peak_oni": 2.4, "type": "EP", "color": "#c0392b"},
    "2002–03": {"onset_yr": 2002, "peak_oni": 1.3, "type": "CP", "color": "#2980b9"},
    "2004–05": {"onset_yr": 2004, "peak_oni": 0.9, "type": "CP", "color": "#95a5a6"},
    "2006–07": {"onset_yr": 2006, "peak_oni": 1.1, "type": "EP", "color": "#f39c12"},
    "2009–10": {"onset_yr": 2009, "peak_oni": 1.3, "type": "CP", "color": "#3498db"},
    "2015–16": {"onset_yr": 2015, "peak_oni": 2.6, "type": "EP", "color": "#16a085"},
    "2018–19": {"onset_yr": 2018, "peak_oni": 0.9, "type": "CP", "color": "#95a5a6"},
    "2023–24": {"onset_yr": 2023, "peak_oni": 2.0, "type": "EP", "color": "#e67e22"},
    "2026–27": {"onset_yr": 2026, "peak_oni": None, "type": "EP", "color": "#1e6fa5"},
}


def _get(url: str) -> str:
    r = requests.get(url, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.text


def fetch_sst() -> pd.DataFrame:
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
    # The CPC file stacks two tables: the Tahiti-Darwin pressure ANOMALY (in mb)
    # first, then STANDARDIZED DATA. "SOI" conventionally means the standardized
    # index, so drop everything before that header.
    raw   = _get(URLS["soi"])
    lines = raw.splitlines()
    for i, line in enumerate(lines):
        if "STANDARDIZED" in line.upper():
            lines = lines[i + 1:]
            break
    else:
        raise ValueError("SOI file: STANDARDIZED DATA section not found")

    rows = []
    for line in lines:
        line = line.strip()
        if len(line) < 4 or not line[:4].isdigit():
            continue
        yr   = int(line[:4])
        rest = line[4:]
        rest = re.sub(r"(\d)(-)", r"\1 \2", rest)
        for m, v in enumerate(rest.split()[:12], start=1):
            try:
                val = float(v)
                if abs(val) < 50:
                    rows.append({"year": yr, "month": m, "soi": val})
            except ValueError:
                continue
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    s = df.set_index("date")["soi"].sort_index()
    return s[~s.index.duplicated(keep="first")].to_frame()


def fetch_mei() -> pd.Series:
    """Multivariate ENSO Index v2 from NOAA PSL, in standard deviations.

    Rows are twelve overlapping two-month seasons (DJ JF FM ... ND). Each is
    labelled by its pair, and DJ of row Y spans Dec of Y-1 and Jan of Y, so
    assigning every season to its second month makes column i land on month i
    and keeps the series monotonic within a row. Verified against known peaks:
    1982 ND +2.36, 1998 JF +2.45, 2015 AS +2.24.
    """
    rows = []
    for line in _get(URLS["mei"]).splitlines():
        parts = line.split()
        # Skip the leading year-range header, the trailing -999.00 flag line and
        # the free-text provenance footer; real data rows are a year plus 12
        # values.
        if len(parts) != 13 or not parts[0].isdigit():
            continue
        yr = int(parts[0])
        for m, v in enumerate(parts[1:], start=1):
            try:
                val = float(v)
            except ValueError:
                continue
            if val <= -999.0:
                continue
            rows.append({"year": yr, "month": m, "mei": val})
    if not rows:
        raise ValueError("MEI file: no data rows parsed")
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    return df.set_index("date")["mei"].sort_index()


def fetch_iri_plume(year: int = None, month: int = None, back: int = 6) -> dict:
    """IRI ENSO forecast plume, from the JSON feed if it answers, else the figure.

    The JSON feed started returning 403 in August 2026. The block is specific to
    that path -- the plume figure on the same host still answers -- so on failure
    this falls back to reading the numbers back out of the figure's SVG. Both
    paths return the same shape; "source" says which one answered.
    """
    try:
        return _fetch_iri_plume_json(year, month, back)
    except Exception as json_err:
        from .plume_svg import fetch_iri_plume_svg
        try:
            return fetch_iri_plume_svg(year, month, back)
        except Exception as svg_err:
            raise ValueError(f"IRI plume: JSON feed failed ({json_err}); "
                             f"figure fallback failed ({svg_err})") from svg_err


def _fetch_iri_plume_json(year: int = None, month: int = None,
                          back: int = 6) -> dict:
    """IRI ENSO forecast plume: per-model Niño-3.4 forecasts by 3-month season.

    Two things about this feed are not self-evident and were established by
    checking it against the observed ONI record:

    1. The nine values start at the season BEGINNING in the issue month, so a
       June issue starts at JJA (centred July) and skips the partially complete
       MJJ. Tested on 33 plumes from 2015-2025 by comparing the shortest-lead
       value against the season that actually verified: mean absolute error is
       0.132 C for this alignment against 0.265 for JJA-as-second and 0.169 for
       JJA-as-first-plus-one.

    2. Models are typed Dynamical, Statistical, Relative or Other. "Relative"
       models forecast the relative ONI, which sits on a different baseline, and
       the feed's own averages.total blends all three types together. The ONI
       ensemble returned here therefore uses Dynamical and Statistical only.
       "Other" entries are the feed's precomputed averages, not models.

    A request for a month that has not been issued yet returns the previous
    plume rather than an error, so the payload's own month field decides the
    issue date.
    """
    now = pd.Timestamp.today()
    year  = year  or now.year
    month = month or now.month

    payload = err = None
    for i in range(back + 1):
        target = pd.Timestamp(year, month, 1) - pd.DateOffset(months=i)
        url = URLS["plume"].format(year=target.year, month=target.month)
        try:
            payload = json.loads(_get(url))
        except Exception as e:            # not issued yet -> 500
            err = e
            continue
        got = int(payload["month"])
        # The feed silently serves the latest available plume, so recover the
        # true issue year from whether its month ran past the one requested.
        if got == target.month:
            iy, im = target.year, got
        elif got < target.month:
            iy, im = target.year, got
        else:
            iy, im = target.year - 1, got
        break
    else:
        raise ValueError(f"IRI plume: nothing issued in the last {back} months ({err})")

    models = {}
    for m in payload.get("models", []):
        vals = [None if v is None or v <= -998 else float(v) for v in m["data"]]
        models[m["model"]] = {"type": m["type"], "data": vals}

    n = len(next(iter(models.values()))["data"]) if models else 0
    seasons = []
    for k in range(n):
        start  = pd.Timestamp(iy, im, 1) + pd.DateOffset(months=k)
        centre = start + pd.DateOffset(months=1)
        seasons.append({"code": SEASON_BY_START[start.month - 1],
                        "start": start, "centre": centre})

    oni_types = ("Dynamical", "Statistical")
    mean, lo, hi, q25, q75, counts = [], [], [], [], [], []
    for k in range(n):
        vals = [v["data"][k] for v in models.values()
                if v["type"] in oni_types and v["data"][k] is not None]
        counts.append(len(vals))
        if vals:
            a = np.array(vals)
            mean.append(round(float(a.mean()), 3))
            lo.append(round(float(a.min()), 3))
            hi.append(round(float(a.max()), 3))
            # The full envelope reaches well past the chart's y range and is
            # driven by one or two extreme members, so the interquartile band is
            # what gets shaded; min and max stay available for the caption.
            q25.append(round(float(np.percentile(a, 25)), 3))
            q75.append(round(float(np.percentile(a, 75)), 3))
        else:
            for lst in (mean, lo, hi, q25, q75):
                lst.append(None)

    observed = [{"label": o["month"], "value": float(o["data"])}
                for o in payload.get("observed", [])]

    return {
        "issue_year": iy, "issue_month": im,
        "issue_label": pd.Timestamp(iy, im, 1).strftime("%B %Y"),
        "seasons": seasons, "models": models,
        "oni_mean": mean, "oni_min": lo, "oni_max": hi,
        "oni_q25": q25, "oni_q75": q75,
        "n_models": counts,
        "n_oni_models": sum(1 for v in models.values() if v["type"] in oni_types),
        "observed": observed,
        "source": "plumes_json",
    }


def fetch_weekly_sst() -> pd.DataFrame:
    """Weekly Nino-region SSTs and anomalies from CPC.

    Column order in this file is Nino1+2, Nino3, Nino34, Nino4 -- note that
    Nino34 and Nino4 are swapped relative to the monthly sstoi.indices file.
    SST and SSTA run together when the anomaly is negative ("20.6-0.1"), so the
    numbers have to be pulled out by regex rather than split on whitespace.
    """
    MONTHS = {m: i + 1 for i, m in enumerate(
        ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
         "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}
    head = re.compile(r"^\s*(\d{2})([A-Z]{3})(\d{4})\s+(.*)$")

    rows = []
    for line in _get(URLS["wksst"]).splitlines():
        m = head.match(line)
        if not m:
            continue
        day, mon, yr, rest = m.groups()
        if mon not in MONTHS:
            continue
        nums = [float(v) for v in re.findall(r"-?\d+\.\d", rest)]
        if len(nums) < 8:
            continue
        rows.append({
            "date": pd.Timestamp(int(yr), MONTHS[mon], int(day)),
            "nino12": nums[0], "nino12_anom": nums[1],
            "nino3":  nums[2], "nino3_anom":  nums[3],
            "nino34": nums[4], "nino34_anom": nums[5],
            "nino4":  nums[6], "nino4_anom":  nums[7],
        })
    if not rows:
        raise ValueError("weekly SST file: no data rows parsed")
    return pd.DataFrame(rows).set_index("date").sort_index()


def fetch_sst_anomaly_map(year: int, month: int) -> dict:
    import xarray as xr
    url = (f"https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/netcdf/"
           f"ersst.v5.{year:04d}{month:02d}.nc")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as fh:
        fh.write(r.content)
        fname = fh.name
    try:
        with xr.open_dataset(fname) as ds:
            ssta = ds["ssta"].squeeze().values
            lats = ds["lat"].values
            lons = ds["lon"].values
    finally:
        os.unlink(fname)
    return {"lat": lats, "lon": lons, "ssta": ssta}


def fetch_roni() -> pd.Series:
    from html.parser import HTMLParser

    class _TP(HTMLParser):
        def __init__(self):
            super().__init__()
            self.in_table = self.in_cell = False
            self.rows, self.row, self.cell = [], [], ""
        def handle_starttag(self, tag, attrs):
            if tag == "table": self.in_table = True
            if self.in_table and tag in ("td","th"): self.in_cell = True; self.cell = ""
            if self.in_table and tag == "tr": self.row = []
        def handle_endtag(self, tag):
            if tag == "table": self.in_table = False
            if self.in_table and tag in ("td","th"): self.in_cell = False; self.row.append(self.cell.strip())
            if self.in_table and tag == "tr":
                if self.row: self.rows.append(self.row)
        def handle_data(self, data):
            if self.in_cell: self.cell += data

    SEASONS = ["DJF","JFM","FMA","MAM","AMJ","MJJ","JJA","JAS","ASO","SON","OND","NDJ"]
    CENTRE  = {s: i+1 for i, s in enumerate(SEASONS)}

    p = _TP()
    p.feed(_get("https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/roni/"))

    rows = []
    for row in p.rows:
        if not row or not row[0].strip().isdigit():
            continue
        yr = int(row[0].strip())
        for j, seas in enumerate(SEASONS):
            if j + 1 < len(row):
                try:
                    rows.append({"year": yr, "month": CENTRE[seas], "roni": float(row[j+1].strip())})
                except ValueError:
                    pass

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    return df.set_index("date")["roni"].sort_index()


def aligned_series(series: pd.Series, onset_yr: int, n: int = 24) -> list:
    start = pd.Timestamp(onset_yr - 1, 7, 1)
    idx   = pd.date_range(start, periods=n, freq="MS")
    vals  = series.reindex(idx)
    return [None if np.isnan(v) else round(float(v), 2) for v in vals]
