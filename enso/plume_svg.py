"""Recover the IRI ENSO plume from IRI's own figure when the JSON feed is shut.

The machine-readable plume feed (plumes_json) began returning 403 in August
2026. The block is path-specific: IRI's page still draws the plume from
figure4_plot/{year}/{month} on the same host, and that path answers normally to
a request carrying browser headers. It serves a Matplotlib SVG, which is vector
-- the model lines are real coordinates and the axis ticks carry their own
labels, so the numbers behind the picture can be read back out rather than
approximated.

Three things about the SVG are not self-evident:

1. Matplotlib writes text as <use> references to glyph paths named by DejaVu
   Sans GLYPH INDEX, not by codepoint. Across printable ASCII the two differ by
   a constant 29, so labels decode with a shift. The Unicode minus in the y
   labels falls outside that range and is handled by sign detection instead.

2. The month in the figure URL is not the issue month: /2026/7 serves the plume
   issued in August 2026. IRI's own page shows the same offset. Requesting past
   the latest issue clamps to it rather than erroring, so the decoded figure
   title is what settles the issue date.

3. Marker definition ids are shared between the legend swatch and the plotted
   line, but two models drawn in an identical colour and marker share one id, so
   the id alone does not identify a series. Series are therefore paired with
   legend entries by document order, which was verified against style matching
   on all 26 members of the August 2026 plume.

Unlike the JSON feed, this figure carries only dynamical and statistical models
-- there are no relative-ONI members to filter out.
"""
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests

FIGURE_URL = "https://ensoforecast2.iri.columbia.edu/figure4_plot/{year}/{month}"

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/128.0.0.0 Safari/537.36"),
    "Accept": "image/svg+xml,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://iri.columbia.edu/our-expertise/climate/forecasts/enso/current/",
    "Origin": "https://iri.columbia.edu",
}

URL_MONTH_OFFSET = 1
_TIMEOUT = 30

SVG = "{http://www.w3.org/2000/svg}"
HREF = "{http://www.w3.org/1999/xlink}href"
GLYPH_TO_ASCII_OFFSET = 29

# Line styles that identify the non-model curves. The three ensemble averages
# are recomputed from the members rather than read off these, but they are
# matched so they stay out of the member pool.
AVERAGE_COLORS = {"#000090": "COMBINED AVG",
                  "#900000": "DYN Average",
                  "#008000": "STAT Average"}
OBS_COLOR = "#000000"

SEASON_BY_START = ["JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA",
                   "JAS", "ASO", "SON", "OND", "NDJ", "DJF"]


# --------------------------------------------------------------------------
# fetching


def fetch_plume_svg(year: int = None, month: int = None, back: int = 6) -> dict:
    """Return the plume figure issued in the given month as SVG text.

    year/month name the ISSUE month; the URL offset is applied here.
    """
    now = pd.Timestamp.today()
    issue = pd.Timestamp(year or now.year, month or now.month, 1)

    err = None
    for i in range(back + 1):
        wanted = issue - pd.DateOffset(months=i)
        target = wanted - pd.DateOffset(months=URL_MONTH_OFFSET)
        url = FIGURE_URL.format(year=target.year, month=target.month)
        try:
            r = requests.get(url, headers=HEADERS, timeout=_TIMEOUT)
            r.raise_for_status()
        except Exception as e:
            err = e
            continue
        if "svg" not in r.headers.get("Content-Type", ""):
            err = ValueError(f"{url}: not SVG ({r.headers.get('Content-Type')})")
            continue
        title = _figure_title(r.text)
        return {"svg": r.text, "title": title,
                "issue_label": _issue_from_title(title) or wanted.strftime("%b %Y"),
                "url": url}
    raise ValueError(f"IRI plume figure: nothing served in the last {back} "
                     f"months ({err})")


# --------------------------------------------------------------------------
# SVG text decoding


def _decode(elem) -> str:
    """Decode a Matplotlib glyph run into text."""
    out = []
    for use in elem.iter(SVG + "use"):
        href = use.get(HREF, "")
        if href.startswith("#DejaVuSans-"):
            out.append(chr(int(href.rsplit("-", 1)[1], 16) + GLYPH_TO_ASCII_OFFSET))
    return "".join(out)


def _figure_title(svg: str) -> str:
    try:
        root = ET.fromstring(svg)
    except ET.ParseError:
        return ""
    best = ""
    for g in root.iter(SVG + "g"):
        if not g.get("id", "").startswith("text_"):
            continue
        text = _decode(g)
        if "Predictions of ENSO" in text:
            return text
        if len(text) > len(best):
            best = text
    return best


def _issue_from_title(title: str):
    m = re.search(r"from\s+([A-Z][a-z]{2})\s+(\d{4})", title or "")
    return f"{m.group(1)} {m.group(2)}" if m else None


def _tick_value(label: str):
    """Parse a y tick label. The minus sign is Unicode, outside the glyph shift,
    so the sign is taken from any leading non-numeric character."""
    m = re.search(r"[0-9]+(?:\.[0-9]+)?", label)
    if not m:
        return None
    val = float(m.group(0))
    return -val if m.start() > 0 else val


# --------------------------------------------------------------------------
# geometry


def _style(elem) -> str:
    return elem.get("style") or ""


def _line_path(g):
    """The stroked path of a line2d group, or None."""
    for p in g:
        if p.tag == SVG + "path" and "stroke" in _style(p) and p.get("id") is None:
            return p
    return None


def _stroke(style: str):
    m = re.search(r"stroke:\s*(#[0-9a-fA-F]{3,8})", style)
    return m.group(1).lower() if m else None


def _stroke_width(style: str) -> float:
    m = re.search(r"stroke-width:\s*([0-9.]+)", style)
    return float(m.group(1)) if m else 1.0


def _vertices(path_elem):
    """Coordinates of an SVG path made of M/L commands."""
    d = path_elem.get("d") or ""
    return [(float(x), float(y))
            for x, y in re.findall(r"[ML]\s*(-?[0-9.]+)\s+(-?[0-9.]+)", d)]


def _markers(g):
    """Data-point coordinates from a line2d group's marker <use> elements."""
    pts = []
    for use in g.iter(SVG + "use"):
        if (use.get(HREF) or "").startswith("#DejaVuSans"):
            continue
        try:
            pts.append((float(use.get("x")), float(use.get("y"))))
        except (TypeError, ValueError):
            continue
    return pts


def _axes_transform(axes):
    """Return (to_value, x_columns): a y-pixel->value function and the list of
    (pixel_x, label) axis columns."""
    yticks, xcols = [], []
    for axis in axes:
        if not axis.get("id", "").startswith("matplotlib.axis"):
            continue
        for tick in axis:
            tid = tick.get("id", "")
            if "tick" not in tid:
                continue
            mark = next((u for c in tick for u in c.iter(SVG + "use")
                         if not (u.get(HREF) or "").startswith("#DejaVuSans")), None)
            label = "".join(_decode(c) for c in tick
                            if c.get("id", "").startswith("text_"))
            if mark is None or not label:
                continue
            if tid.startswith("ytick"):
                val = _tick_value(label)
                if val is not None:
                    yticks.append((float(mark.get("y")), val))
            elif tid.startswith("xtick"):
                xcols.append((float(mark.get("x")), label))

    if len(yticks) < 2:
        raise ValueError("plume SVG: could not read the y axis ticks")

    py = np.array([p for p, _ in yticks])
    vv = np.array([v for _, v in yticks])
    slope, intercept = np.polyfit(py, vv, 1)
    resid = float(np.abs(slope * py + intercept - vv).max())
    if resid > 0.01:
        raise ValueError(f"plume SVG: y axis is not linear (residual {resid:.3f})")

    xcols.sort()
    return (lambda y: float(slope * y + intercept)), xcols


def _column_index(x, xcols, tol):
    for i, (px, _) in enumerate(xcols):
        if abs(px - x) <= tol:
            return i
    return None


# --------------------------------------------------------------------------
# parsing


def parse_plume_svg(svg: str, issue_label: str = None) -> dict:
    """Read the model ensemble out of an IRI plume figure.

    Returns the same shape as the old plumes_json reader so callers do not have
    to care which source answered.
    """
    root = ET.fromstring(svg)
    axes = next((g for g in root.iter(SVG + "g")
                 if g.get("id", "").startswith("axes_")), None)
    if axes is None:
        raise ValueError("plume SVG: no axes group")

    to_value, xcols = _axes_transform(axes)
    if len(xcols) < 3:
        raise ValueError("plume SVG: too few x axis columns")
    tol = abs(xcols[1][0] - xcols[0][0]) / 3

    # Split the plotted curves. Dashed grey segments are the lead-in connectors
    # between the last observation and each member's first forecast, not data.
    members, obs_pts = [], []
    for g in axes:
        if not g.get("id", "").startswith("line2d_"):
            continue
        path = _line_path(g)
        if path is None:
            continue
        style = _style(path)
        if "stroke-dasharray" in style:
            continue
        stroke = _stroke(style)
        if stroke in AVERAGE_COLORS and _stroke_width(style) >= 4:
            continue
        pts = _markers(g)
        if not pts:
            # The observed curve carries no markers, so it is read off its own
            # vertices; anything else unmarked is a reference line.
            if stroke == OBS_COLOR and _stroke_width(style) >= 2:
                obs_pts = _vertices(path)
            continue
        members.append(pts)

    names = _legend_entries(axes)
    if len(names) != len(members):
        raise ValueError(f"plume SVG: {len(members)} plotted series but "
                         f"{len(names)} legend entries")

    # Forecast columns are those any member actually plots on, in axis order.
    used = sorted({i for pts in members
                   for i in [_column_index(x, xcols, tol) for x, _ in pts]
                   if i is not None})
    if not used:
        raise ValueError("plume SVG: no member points landed on an axis column")

    codes = [xcols[i][1] for i in used]
    n = len(codes)

    models = {}
    for (name, mtype), pts in zip(names, members):
        data = [None] * n
        for x, y in pts:
            i = _column_index(x, xcols, tol)
            if i in used:
                data[used.index(i)] = round(to_value(y), 3)
        models[name] = {"type": mtype, "data": data}

    label = issue_label or _issue_from_title(_figure_title(svg))
    issue = pd.Timestamp(label + " 1") if label else None
    if issue is None:
        raise ValueError("plume SVG: could not read the issue month")
    seasons = _seasons(codes, issue)

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
            q25.append(round(float(np.percentile(a, 25)), 3))
            q75.append(round(float(np.percentile(a, 75)), 3))
        else:
            for lst in (mean, lo, hi, q25, q75):
                lst.append(None)

    observed = []
    for x, y in obs_pts:
        i = _column_index(x, xcols, tol)
        if i is not None and i not in used:
            observed.append({"label": xcols[i][1], "value": round(to_value(y), 3)})

    return {
        "issue_year": int(issue.year), "issue_month": int(issue.month),
        "issue_label": issue.strftime("%B %Y"),
        "seasons": seasons, "models": models,
        "oni_mean": mean, "oni_min": lo, "oni_max": hi,
        "oni_q25": q25, "oni_q75": q75,
        "n_models": counts,
        "n_oni_models": sum(1 for v in models.values() if v["type"] in oni_types),
        "observed": observed,
        "source": "figure4_plot SVG",
    }


def _legend_entries(axes):
    """(name, type) for each plotted member, in the order the legend lists them.

    The legend alternates swatch and label, with the two section headers
    switching the model type that follows.
    """
    legend = next((g for g in axes.iter(SVG + "g")
                   if g.get("id", "").startswith("legend_")), None)
    if legend is None:
        raise ValueError("plume SVG: no legend group")

    entries, mtype, swatch = [], None, None
    for child in legend:
        cid = child.get("id", "")
        if cid.startswith("line2d_"):
            swatch = child
        elif cid.startswith("text_"):
            text = _decode(child).strip()
            if text == "DYN MODELS:":
                mtype = "Dynamical"
            elif text == "STAT MODELS:":
                mtype = "Statistical"
            elif (text and mtype and swatch is not None
                  and _markers(swatch)
                  and text not in AVERAGE_COLORS.values()):
                entries.append((text, mtype))
            swatch = None
    return entries


def _seasons(codes, issue):
    """Turn season codes into dated seasons, walking forward from the issue
    month. The code sets the starting month, so no fixed offset is assumed."""
    seasons, year = [], issue.year
    prev = None
    for code in codes:
        if code not in SEASON_BY_START:
            raise ValueError(f"plume SVG: unrecognised season code {code!r}")
        month = SEASON_BY_START.index(code) + 1
        start = pd.Timestamp(year, month, 1)
        if prev is not None and start <= prev:
            year += 1
            start = pd.Timestamp(year, month, 1)
        prev = start
        seasons.append({"code": code, "start": start,
                        "centre": start + pd.DateOffset(months=1)})
    return seasons


def fetch_iri_plume_svg(year: int = None, month: int = None,
                        back: int = 6) -> dict:
    """Fetch and parse the plume figure. Same return shape as the JSON reader."""
    fig = fetch_plume_svg(year, month, back)
    plume = parse_plume_svg(fig["svg"], fig["issue_label"])
    plume["url"] = fig["url"]
    return plume
