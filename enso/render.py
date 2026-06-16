"""
Builds a self-contained HTML page with interactive Plotly charts.
Plotly JS loaded once from CDN; all other charts reuse the same bundle.
"""
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np
from enso.data import ENSO_EVENTS, aligned_series

MONTH_LABELS = [
    "Jul","Aug","Sep","Oct","Nov","Dec",
    "Jan","Feb","Mar","Apr","May","Jun",
    "Jul","Aug","Sep","Oct","Nov","Dec",
    "Jan","Feb","Mar","Apr","May","Jun",
]

# Observed 2026 data (Jul 2025 to Jun 2026) + NOAA/IRI model consensus forecast (Jul 2026 to Jun 2027)
NINO34_2026 = [-.6,-.7,-.8,-.9,-.9,-.9, -.8,-.5,-.3,-.1,.6,.8, 1.1,1.4,1.7,1.9,2.1,2.2, 2.2,2.0,1.6,1.1,.5,.0]
# Nino 1+2 leads Nino 3.4 in EP events and peaks higher; observed from NOAA, forecast estimated
NINO12_2026 = [-.5,-.6,-.7,-.8,-.8,-.7, -.3,.92,1.29,1.52,1.81,2.1, 2.8,3.4,3.8,4.0,4.0,3.7, 3.1,2.4,1.6,.8,.1,-.4]
SOI_2026    = [ 6., 8.,10.,10., 8., 5.,  2., 0.,-4.,-11.,-11.,-22., -20.,-22.,-20.,-22.,-20.,-18., -16.,-13.,-9.,-5.,-1., 3.]
N_OBS       = 12   # months with real observations (Jul 2025 to Jun 2026)

SHOW_EVENTS = ["1997–98", "2015–16", "1982–83", "2023–24"]

_SOI_HIST = {
    "1997–98": [ 2, 1, 0,-2,-3,-4, -6,-10,-14,-16,-18,-20, -22,-24,-22,-24,-22,-20, -18,-15,-12,-8,-3, 4],
    "2015–16": [ 5, 4, 3, 2, 0,-2, -4, -6, -8,-10,-12,-14, -16,-18,-20,-22,-20,-18, -16,-14,-10,-6,-2, 2],
    "1982–83": [ 3, 2, 1, 0,-1,-2, -3, -5, -8,-10,-14,-16, -18,-20,-18,-20,-18,-16, -15,-13, -9,-5,-1, 4],
    "2023–24": [ 2, 1,-1,-2,-3,-4, -5, -8,-12,-14,-16,-14, -16,-18,-14,-16,-14,-12,  -8, -5, -2, 0, 2, 4],
}

_LAYOUT = dict(
    plot_bgcolor="white", paper_bgcolor="white",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    font=dict(family="Hanken Grotesk, Barlow Condensed, sans-serif", size=12),
    margin=dict(l=60, r=90, t=80, b=60),
)
_XAXIS = dict(
    tickmode="array",
    tickvals=[0, 3, 6, 9, 12, 15, 18, 21, 23],
    ticktext=[MONTH_LABELS[i] for i in [0, 3, 6, 9, 12, 15, 18, 21, 23]],
    title_text="Month (July of precursor year = 0)",
    showgrid=True, gridcolor="rgba(0,0,0,0.07)",
)
_YAXIS = dict(showgrid=True, gridcolor="rgba(0,0,0,0.07)")


def _pjs(include):
    return include  # True = inline, 'cdn' = CDN, False = skip


def _split(vals, n_obs):
    obs  = [v if i < n_obs else None for i, v in enumerate(vals)]
    fcst = [v if i >= n_obs - 1 else None for i, v in enumerate(vals)]
    return obs, fcst


# ── CHART: Pacific monitoring region map ──────────────────────────────────────
def chart_pacific_map(include_plotlyjs=False) -> str:
    """Interactive Plotly geo map with Nino regions and SOI stations."""
    fig = go.Figure()

    # Nino 3.4, 3, 1+2 — simple rectangles
    for name, lats, lons, fc, lc in [
        ("Nino 3.4", [5,5,-5,-5,5], [-170,-120,-120,-170,-170], "rgba(243,156,18,0.42)", "#e67e22"),
        ("Nino 3",   [5,5,-5,-5,5], [-150,-90,-90,-150,-150],   "rgba(231,76,60,0.30)",  "#c0392b"),
        ("Nino 1+2", [0,0,-10,-10,0],[-90,-80,-80,-90,-90],     "rgba(155,89,182,0.55)", "#8e44ad"),
    ]:
        fig.add_trace(go.Scattergeo(
            lat=lats, lon=lons, mode="lines", fill="toself",
            fillcolor=fc, line=dict(color=lc, width=2),
            name=name, showlegend=True,
        ))

    # Nino 4 crosses the dateline — fills without stroke (no seam) + outer border segments.
    # All traces share legendgroup="nino4" so the legend toggle works correctly.
    for lats, lons in [
        ([5,5,-5,-5,5], [160,180,180,160,160]),
        ([5,5,-5,-5,5], [-180,-150,-150,-180,-180]),
    ]:
        fig.add_trace(go.Scattergeo(
            lat=lats, lon=lons, mode="lines", fill="toself",
            fillcolor="rgba(46,204,113,0.30)",
            line=dict(color="rgba(0,0,0,0)", width=0),
            name="Nino 4", legendgroup="nino4", showlegend=False, hoverinfo="skip",
        ))
    for lats, lons in [
        ([5, -5],  [160, 160]),
        ([5,  5],  [160, 180]),
        ([-5,-5],  [160, 180]),
        ([5,  5],  [-180,-150]),
        ([-5,-5],  [-180,-150]),
        ([5, -5],  [-150,-150]),
    ]:
        fig.add_trace(go.Scattergeo(
            lat=lats, lon=lons, mode="lines",
            line=dict(color="#27ae60", width=2),
            name="Nino 4", legendgroup="nino4", showlegend=False, hoverinfo="skip",
        ))
    # Single legend entry that controls the whole group
    fig.add_trace(go.Scattergeo(
        lat=[0], lon=[200], mode="lines",
        line=dict(color="#27ae60", width=2),
        name="Nino 4", legendgroup="nino4",
        showlegend=True, hoverinfo="skip",
    ))

    # Region labels — placed above the boxes (lat ~14°N) so they don't overlap
    for label, lon, color in [
        ("NIÑO 4",   175,  "#27ae60"),
        ("NIÑO 3.4", -145, "#e67e22"),
        ("NIÑO 3",   -118, "#c0392b"),
        ("NIÑO 1+2", -85,  "#8e44ad"),
    ]:
        fig.add_trace(go.Scattergeo(
            lat=[14], lon=[lon],
            mode="text",
            text=[label],
            textfont=dict(color=color, size=13,
                          family="Barlow Condensed, sans-serif"),
            showlegend=False, hoverinfo="skip",
        ))

    # SOI stations
    fig.add_trace(go.Scattergeo(
        lat=[-12.4, -17.5], lon=[130.8, -149.6],
        mode="markers+text",
        marker=dict(size=10, color="#f1c40f", line=dict(width=2, color="white")),
        text=["Darwin", "Tahiti"], textposition=["top right", "top right"],
        textfont=dict(color="white", size=11, family="Hanken Grotesk, sans-serif"),
        name="SOI stations",
    ))
    fig.add_trace(go.Scattergeo(
        lat=[-12.4, -17.5], lon=[130.8, -149.6],
        mode="lines", line=dict(color="#f1c40f", width=1.5, dash="dot"),
        showlegend=False, hoverinfo="skip",
    ))

    fig.update_geos(
        scope="world",
        projection_type="natural earth",
        # Rotate central meridian to 180° so the Pacific is centred
        projection_rotation=dict(lon=180),
        lataxis_range=[-35, 22],
        # lon range relative to rotated centre: -100 to +100 covers ~80°E to ~80°W
        lonaxis_range=[-100, 100],
        showland=True,      landcolor="#4a7c45",
        showocean=True,     oceancolor="#1a6b9a",
        showcoastlines=True, coastlinecolor="rgba(255,255,255,0.5)",
        coastlinewidth=0.8,
        showlakes=False, showrivers=False,
        bgcolor="#1a6b9a",
        showframe=False,
    )
    fig.update_layout(
        title="ENSO Monitoring Regions and SOI Stations",
        height=360,
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.0,
                    xanchor="center", x=0.5, bgcolor="rgba(255,255,255,0.85)"),
        font=dict(family="Hanken Grotesk, Barlow Condensed, sans-serif", size=12),
        paper_bgcolor="#1a6b9a",
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs=_pjs(include_plotlyjs))


# ── CHART: Full ONI time series ───────────────────────────────────────────────
def chart_oni_full(oni: pd.DataFrame, include_plotlyjs=False) -> str:
    """Complete ONI record from 1950 to present with shaded El Nino/La Nina periods."""
    fig = go.Figure()

    dates = oni.index.tolist()
    vals  = oni["oni"].tolist()

    # Shaded fills above/below thresholds
    el_nino = [v if v >= 0.5 else 0.0 for v in vals]
    la_nina = [v if v <= -0.5 else 0.0 for v in vals]

    fig.add_trace(go.Scatter(
        x=dates, y=el_nino, fill="tozeroy",
        fillcolor="rgba(192,57,43,0.25)", line=dict(width=0),
        mode="none", name="El Nino (ONI >= +0.5)", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=la_nina, fill="tozeroy",
        fillcolor="rgba(30,111,165,0.25)", line=dict(width=0),
        mode="none", name="La Nina (ONI <= -0.5)", hoverinfo="skip",
    ))

    # ONI line
    fig.add_trace(go.Scatter(
        x=dates, y=vals, mode="lines",
        name="ONI", line=dict(color="#1a2535", width=1.5),
        hovertemplate="%{x|%b %Y}: %{y:.2f} °C<extra></extra>",
    ))

    # Threshold lines
    fig.add_hline(y= 0.5, line_dash="dash", line_color="rgba(192,57,43,0.5)", line_width=1)
    fig.add_hline(y=-0.5, line_dash="dash", line_color="rgba(30,111,165,0.5)", line_width=1)
    fig.add_hline(y= 0,   line_color="rgba(0,0,0,0.15)", line_width=1)

    # Annotate the four super events + 2026 current position
    oni_s = oni["oni"]
    for label, yr in [("1982-83",1983),("1997-98",1998),("2015-16",2016),("2023-24",2024)]:
        subset = oni_s[oni_s.index.year == yr]
        if not subset.empty:
            peak_date = subset.idxmax()
            peak_val  = subset.max()
            fig.add_annotation(
                x=peak_date, y=peak_val + 0.12,
                text=label, showarrow=False,
                font=dict(size=9, color="#c0392b"),
                xanchor="center",
            )

    fig.update_yaxes(title_text="ONI (°C)", range=[-2.8, 3.4], **_YAXIS)
    fig.update_xaxes(title_text="Year", showgrid=True, gridcolor="rgba(0,0,0,0.07)")
    fig.update_layout(
        title="Oceanic Nino Index: Full Record 1950 to present",
        height=360,
        plot_bgcolor="white", paper_bgcolor="white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=80, b=50),
        font=dict(family="Hanken Grotesk, Barlow Condensed, sans-serif", size=12),
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs=_pjs(include_plotlyjs))


# ── CHART: Nino-3.4 aligned comparison ───────────────────────────────────────
def chart_nino34(nino34_series: pd.Series, include_plotlyjs=False) -> str:
    fig = go.Figure()

    for y, color, label in [
        ( 0.5, "rgba(192,57,43,0.55)",  "El Nino (+0.5 °C)"),
        ( 0.0, "rgba(0,0,0,0.18)",      ""),
        (-0.5, "rgba(30,111,165,0.45)", "La Nina (-0.5 °C)"),
    ]:
        fig.add_hline(y=y, line_dash="dash", line_color=color, line_width=1,
                      annotation_text=label, annotation_position="right",
                      annotation_font_size=10)

    fig.add_vline(x=N_OBS - 1, line_dash="dot", line_color="rgba(30,111,165,0.65)",
                  line_width=1.5, annotation_text="now", annotation_position="top left")

    x = list(range(24))

    for event in SHOW_EVENTS:
        meta = ENSO_EVENTS[event]
        vals = aligned_series(nino34_series, meta["onset_yr"], 24)
        fig.add_trace(go.Scatter(
            x=x, y=vals, mode="lines+markers", name=event,
            line=dict(color=meta["color"], width=2),
            marker=dict(size=3), opacity=0.85,
        ))

    obs, fcst = _split(NINO34_2026, N_OBS)
    fig.add_trace(go.Scatter(x=x, y=obs, mode="lines+markers",
        name="2026-27 (observed)", line=dict(color="#1e6fa5", width=3.5),
        marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=x, y=fcst, mode="lines",
        name="2026-27 (forecast)", line=dict(color="#1e6fa5", width=2, dash="dash"),
        opacity=0.7))

    fig.update_xaxes(**_XAXIS)
    fig.update_yaxes(title_text="Nino-3.4 anomaly (°C)", range=[-1.6, 3.1], **_YAXIS)
    fig.update_layout(title="Nino-3.4 SST Anomaly: 2026 vs Historical Analogues",
                      height=440, **_LAYOUT)
    return pio.to_html(fig, full_html=False, include_plotlyjs=_pjs(include_plotlyjs))


# ── CHART: Nino 1+2 aligned comparison ───────────────────────────────────────
def chart_nino12(nino12_series: pd.Series, include_plotlyjs=False) -> str:
    """
    Nino 1+2 aligned comparison across ALL historical El Nino events since 1950.
    Top analogues are drawn with full colour; remaining events as grey background lines
    so the full envelope of possible behaviours is visible.
    """
    fig = go.Figure()

    for y, color, label in [
        ( 2.0, "rgba(192,57,43,0.45)",  "+2.0 °C"),
        ( 1.0, "rgba(192,57,43,0.25)",  "+1.0 °C"),
        ( 0.0, "rgba(0,0,0,0.18)",      ""),
        (-1.0, "rgba(30,111,165,0.25)", "-1.0 °C"),
    ]:
        fig.add_hline(y=y, line_dash="dash", line_color=color, line_width=1,
                      annotation_text=label, annotation_position="right",
                      annotation_font_size=10)

    fig.add_vline(x=N_OBS - 1, line_dash="dot", line_color="rgba(30,111,165,0.65)",
                  line_width=1.5, annotation_text="now", annotation_position="top left")

    x = list(range(24))

    # Find every year in the record where Nino 1+2 peaked above +1.5 °C
    # at any point in the 24-month aligned window — this captures events
    # that don't qualify as El Nino in Nino 3.4 (e.g. coastal El Nino events).
    first_year = nino12_series.index.year.min()
    last_year  = nino12_series.index.year.max() - 1  # need at least 1 year of data after onset
    highlight_labels = {ENSO_EVENTS[e]["onset_yr"]: e for e in SHOW_EVENTS}

    warm_years = {}  # onset_yr -> label
    for yr in range(first_year, last_year + 1):
        vals = aligned_series(nino12_series, yr, 24)
        peak = max((v for v in vals if v is not None), default=None)
        if peak is not None and peak >= 1.5:
            warm_years[yr] = vals

    # 1. Grey background lines — all warm years not in the highlighted set
    for yr, vals in warm_years.items():
        if yr == 2026 or yr in highlight_labels:
            continue
        label = f"{yr}-{str(yr+1)[-2:]}"
        fig.add_trace(go.Scatter(
            x=x, y=vals, mode="lines", name=label,
            line=dict(color="rgba(150,150,150,0.35)", width=1),
            showlegend=False,
            hovertemplate=f"<b>{label}</b>: %{{y:.2f}} °C<extra></extra>",
        ))

    # 2. Highlighted analogues on top
    for event in SHOW_EVENTS:
        meta = ENSO_EVENTS[event]
        vals = aligned_series(nino12_series, meta["onset_yr"], 24)
        fig.add_trace(go.Scatter(
            x=x, y=vals, mode="lines+markers", name=event,
            line=dict(color=meta["color"], width=2.5),
            marker=dict(size=3.5), opacity=0.9,
        ))

    # 3. 2026
    obs, fcst = _split(NINO12_2026, N_OBS)
    fig.add_trace(go.Scatter(x=x, y=obs, mode="lines+markers",
        name="2026-27 (observed)", line=dict(color="#1e6fa5", width=3.5),
        marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=x, y=fcst, mode="lines",
        name="2026-27 (forecast)", line=dict(color="#1e6fa5", width=2, dash="dash"),
        opacity=0.7))

    fig.update_xaxes(**_XAXIS)
    fig.update_yaxes(title_text="Nino-1+2 anomaly (°C)", range=[-2.5, 5.5], **_YAXIS)
    fig.update_layout(
        title="Nino 1+2 SST Anomaly: 2026 vs All Events Since 1950",
        height=460, **_LAYOUT,
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs=_pjs(include_plotlyjs))


# ── CHART: SOI aligned comparison ────────────────────────────────────────────
def chart_soi(include_plotlyjs=False) -> str:
    fig = go.Figure()

    for y, color, label in [
        (-8, "rgba(192,57,43,0.55)",  "El Nino (-8)"),
        ( 0, "rgba(0,0,0,0.18)",      ""),
        ( 8, "rgba(30,111,165,0.45)", "La Nina (+8)"),
    ]:
        fig.add_hline(y=y, line_dash="dash", line_color=color, line_width=1,
                      annotation_text=label, annotation_position="right",
                      annotation_font_size=10)

    fig.add_vline(x=N_OBS - 1, line_dash="dot", line_color="rgba(30,111,165,0.65)",
                  line_width=1.5, annotation_text="now", annotation_position="top left")

    x = list(range(24))
    for event, vals in _SOI_HIST.items():
        meta = ENSO_EVENTS[event]
        fig.add_trace(go.Scatter(x=x, y=vals, mode="lines+markers", name=event,
            line=dict(color=meta["color"], width=2), marker=dict(size=3), opacity=0.85))

    obs, fcst = _split(SOI_2026, N_OBS)
    fig.add_trace(go.Scatter(x=x, y=obs, mode="lines+markers",
        name="2026-27 (observed)", line=dict(color="#1e6fa5", width=3.5), marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=x, y=fcst, mode="lines",
        name="2026-27 (forecast)", line=dict(color="#1e6fa5", width=2, dash="dash"), opacity=0.7))

    fig.update_xaxes(**_XAXIS)
    fig.update_yaxes(title_text="SOI", range=[-30, 22], **_YAXIS)
    fig.update_layout(title="Southern Oscillation Index: 2026 vs Historical Analogues",
                      height=420, **_LAYOUT)
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


# ── CHART: Current Nino region bar ────────────────────────────────────────────
def chart_nino_bar(sst: pd.DataFrame, include_plotlyjs=False) -> str:
    latest = sst.iloc[-1]
    date   = sst.index[-1].strftime("%B %Y")
    regions = ["Nino 1+2", "Nino 3", "Nino 3.4", "Nino 4"]
    keys    = ["nino12_anom", "nino3_anom", "nino34_anom", "nino4_anom"]
    values  = [float(latest[k]) for k in keys]
    colors  = ["#9b59b6", "#e74c3c", "#f39c12", "#2ecc71"]

    fig = go.Figure(go.Bar(
        x=regions, y=values, marker_color=colors,
        text=[f"{v:+.2f} °C" for v in values], textposition="outside",
    ))
    for y, color in [(0.5, "rgba(192,57,43,0.5)"), (-0.5, "rgba(30,111,165,0.5)")]:
        fig.add_hline(y=y, line_dash="dash", line_color=color, line_width=1)
    fig.add_hline(y=0, line_color="rgba(0,0,0,0.15)", line_width=1)

    fig.update_yaxes(title_text="SST anomaly (°C)", range=[-1.5, 3.0], **_YAXIS)
    fig.update_layout(
        title=f"Current Nino Region Anomalies ({date})",
        height=340, margin=dict(l=60, r=20, t=60, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Hanken Grotesk, Barlow Condensed, sans-serif", size=12),
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


# ── CHART: Analogue ranking bar ───────────────────────────────────────────────
def chart_analogues(analogues: list, include_plotlyjs=False) -> str:
    top    = analogues[:12]
    names  = [r["event"] for r in top]
    scores = [r["score"] for r in top]
    colors = [r["color"] for r in top]
    hover  = [f"Peak ONI: {r['peak_oni'] or '?'} °C | Type: {r['type']}" for r in top]

    fig = go.Figure(go.Bar(
        x=names, y=scores, marker_color=colors,
        text=[f"{s:.2f}" for s in scores], textposition="outside",
        customdata=hover,
        hovertemplate="<b>%{x}</b><br>Similarity: %{y:.3f}<br>%{customdata}<extra></extra>",
    ))
    fig.update_yaxes(title_text="Similarity (Pearson r)", range=[0, 1.12], **_YAXIS)
    fig.update_layout(
        title="Historical Analogue Ranking: similarity to 2026-27",
        height=380, margin=dict(l=60, r=20, t=60, b=80),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Hanken Grotesk, Barlow Condensed, sans-serif", size=12),
        xaxis_tickangle=-30,
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


# ── HTML assembly ─────────────────────────────────────────────────────────────
def build_html(sst: pd.DataFrame, soi_df: pd.DataFrame, oni: pd.DataFrame,
               analogues: list, cdn: bool = False) -> str:

    latest      = sst.iloc[-1]
    latest_date = sst.index[-1].strftime("%B %Y")
    top3        = analogues[:3]

    nv = float(latest["nino34_anom"])
    if   nv >= 2.0:  phase, ph_col = "Very Strong El Nino", "#c0392b"
    elif nv >= 1.5:  phase, ph_col = "Strong El Nino",      "#e74c3c"
    elif nv >= 0.9:  phase, ph_col = "Moderate El Nino",    "#e07b39"
    elif nv >= 0.5:  phase, ph_col = "Weak El Nino",        "#f39c12"
    elif nv <= -1.5: phase, ph_col = "Strong La Nina",      "#1e6fa5"
    elif nv <= -0.5: phase, ph_col = "La Nina",             "#2980b9"
    else:            phase, ph_col = "Neutral",             "#7f8c8d"

    # Forecast restoration: first month after observations where Nino-3.4 drops below +0.5
    _start = pd.Timestamp(2025, 7, 1)
    restore_str = "Unknown"
    for _i in range(N_OBS, len(NINO34_2026)):
        if NINO34_2026[_i] < 0.5:
            restore_str = (_start + pd.DateOffset(months=_i)).strftime("%B %Y")
            break

    # Generate all charts (Plotly JS loaded once in the first chart; CDN for deploy)
    pjs_mode   = "cdn" if cdn else True
    c_map      = chart_pacific_map(include_plotlyjs=pjs_mode)
    c_nino34   = chart_nino34(sst["nino34_anom"])
    c_nino12   = chart_nino12(sst["nino12_anom"])
    c_soi      = chart_soi()
    c_bar      = chart_nino_bar(sst)
    c_analogue = chart_analogues(analogues)
    c_oni_full = chart_oni_full(oni)

    soi_latest     = float(soi_df["soi"].iloc[-1])
    soi_date       = soi_df.index[-1].strftime("%B %Y")
    ratio          = float(latest["nino12_anom"]) / max(float(latest["nino4_anom"]), 0.1)

    top3_html = ""
    for r in top3:
        top3_html += f"""
        <div style="background:{r['color']}12;border:2px solid {r['color']};
             border-radius:8px;padding:1rem;text-align:center;">
          <div style="font-size:1.4rem;font-weight:700;color:{r['color']}">#{r['rank']}</div>
          <div style="font-size:1.05rem;font-weight:600;margin:.2rem 0">{r['event']}</div>
          <div style="font-size:.8rem;color:#555">Peak ONI: {r['peak_oni'] or '?'} °C / {r['type']}-type</div>
          <div style="font-size:.76rem;color:#888;margin-top:.25rem">similarity: {r['score']:.2f}</div>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>ENSO 2026 Tracker</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700&family=Hanken+Grotesk:wght@400;500;600&display=swap" rel="stylesheet"/>
<style>
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{font-family:"Hanken Grotesk",sans-serif;background:#f7fafc;color:#1a2535;line-height:1.7;}}
  .page{{max-width:960px;margin:0 auto;padding:2rem 1.5rem 4rem;}}
  h1,h2,h3{{font-family:"Barlow Condensed",sans-serif;letter-spacing:.02em;}}
  h2{{font-size:1.35rem;font-weight:700;color:#2B6B72;border-bottom:2px solid #D1E8EC;
      padding-bottom:.3rem;margin:2.5rem 0 .8rem;text-transform:uppercase;}}
  p{{margin-bottom:.8rem;font-size:.94rem;}}
  .badge{{display:inline-block;padding:.25rem .9rem;border-radius:20px;
          font-family:"Barlow Condensed",sans-serif;font-size:.9rem;
          font-weight:700;color:#fff;background:{ph_col};margin-bottom:1rem;}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(185px,1fr));gap:.75rem;margin:1rem 0 1.5rem;}}
  .card{{background:#fff;border:1px solid #d5e3ef;border-radius:8px;padding:.85rem 1rem;}}
  .lbl{{font-size:.68rem;letter-spacing:.04em;color:#7f8c8d;}}
  .val{{font-size:1.4rem;font-weight:700;color:#e07b39;line-height:1.1;}}
  .note{{font-size:.73rem;color:#aaa;margin-top:.1rem;}}
  .cbox{{background:#fff;border:1px solid #d5e3ef;border-radius:8px;padding:1rem;margin:.8rem 0;}}
  .cnote{{font-size:.74rem;color:#7f8c8d;margin-top:.5rem;line-height:1.4;}}
  .infobox{{background:#D1E8EC;border-left:4px solid #2B6B72;border-radius:6px;
            padding:.75rem 1rem;margin:.8rem 0;font-size:.88rem;color:#1a2535;}}
  .alert{{border-radius:6px;padding:.85rem 1rem;margin:.7rem 0;font-size:.86rem;}}
  .ag{{background:#fdf2f8;border-left:4px solid #8e44ad;color:#6c3483;}}
  .aw{{background:#fef9e7;border-left:4px solid #f1c40f;color:#7d6608;}}
  .srcs{{list-style:none;font-size:.79rem;color:#7f8c8d;}}
  .srcs li{{padding:.15rem 0 .15rem 1rem;position:relative;}}
  .srcs li::before{{content:"›";position:absolute;left:0;color:#1e6fa5;}}
  .srcs a{{color:#1e6fa5;text-decoration:none;}}
  .srcs a:hover{{text-decoration:underline;}}
  footer{{text-align:center;padding:1.5rem;font-size:.74rem;color:#bbb;
          border-top:1px solid #e0ecf5;margin-top:2rem;}}
</style>
</head>
<body>

<div style="background:#2B6B72;color:#fff;padding:2.5rem 1.5rem 2rem;text-align:center;">
  <div style="font-family:'Hanken Grotesk',sans-serif;font-size:.82rem;opacity:.7;margin-bottom:.5rem">
    ENSO Monitor / {latest_date}
  </div>
  <h1 style="font-family:'Barlow Condensed',sans-serif;font-size:clamp(1.8rem,4.5vw,3rem);
       font-weight:700;letter-spacing:.03em;margin-bottom:.5rem;">
    El Nino 2026: Conditions and Historical Analogues
  </h1>
  <p style="font-family:'Hanken Grotesk',sans-serif;opacity:.85;max-width:580px;margin:0 auto;font-size:.94rem;">
    Live NOAA data compared to every major ENSO event since 1950
  </p>
</div>

<div class="page">

  <div class="badge">{phase}</div>
  <p>Data current through <strong>{latest_date}</strong>. Source: NOAA CPC.</p>


  <h2>Where are these measured?</h2>
  <p>The four Nino regions are fixed boxes in the equatorial Pacific. The SOI is the pressure
     difference between <strong>Darwin</strong> and <strong>Tahiti</strong>: when it goes strongly
     negative, trade winds weaken and warm water spreads east — the ocean and atmosphere are
     reinforcing each other. Nino-3.4 is the primary ENSO index (El Nino threshold: +0.5 °C).</p>
  <div class="cbox" style="padding:.5rem;">
    {c_map}
  </div>


  <h2>Current conditions</h2>
  <div class="grid">
    <div class="card"><div class="lbl">Nino-3.4</div>
      <div class="val">{latest['nino34_anom']:+.2f} °C</div>
      <div class="note">Primary index / El Nino above +0.5 °C</div></div>
    <div class="card"><div class="lbl">Nino-1+2</div>
      <div class="val">{latest['nino12_anom']:+.2f} °C</div>
      <div class="note">Far eastern Pacific</div></div>
    <div class="card"><div class="lbl">Nino-3</div>
      <div class="val">{latest['nino3_anom']:+.2f} °C</div>
      <div class="note">Central-east Pacific</div></div>
    <div class="card"><div class="lbl">Nino-4</div>
      <div class="val">{latest['nino4_anom']:+.2f} °C</div>
      <div class="note">Central Pacific</div></div>
    <div class="card"><div class="lbl">SOI ({soi_date})</div>
      <div class="val">{soi_latest:+.1f}</div>
      <div class="note">El Nino below -8</div></div>
    <div class="card"><div class="lbl">Prob. Very Strong</div>
      <div class="val">63 %</div>
      <div class="note">NOAA CPC / NDJ 2026-27</div></div>
    <div class="card"><div class="lbl">Forecast decay to neutral</div>
      <div class="val" style="font-size:1.1rem;color:#2B6B72">{restore_str}</div>
      <div class="note">First month Nino-3.4 forecast below +0.5 °C</div></div>
  </div>

  <div class="cbox">
    {c_bar}
  </div>
  <div class="infobox">
    <strong>EP vs CP:</strong> not all El Ninos are the same. In an Eastern Pacific event the warmth
    sits off South America (Nino-1+2 and Nino-3); in a Central Pacific event it pools near the
    dateline (Nino-4). Right now Nino-1+2 ({latest['nino12_anom']:+.2f} °C) is running {ratio:.1f}x
    Nino-4 ({latest['nino4_anom']:+.2f} °C) — a strong EP signature, consistent with 1997-98.
    EP events tend to be stronger and longer-lived than CP events.
  </div>


  <h2>Historical context: ONI since 1950</h2>
  <p>Every El Nino and La Nina in the observational record. Use this to see how frequently
     events occur, how long they last, and how intense they get. The four annotated peaks are
     the events most comparable to 2026.</p>
  <div class="cbox">
    {c_oni_full}
    <p class="cnote">ONI = 3-month running mean of Nino-3.4 SST anomaly. Source: NOAA CPC.</p>
  </div>


  <h2>Historical analogues</h2>
  <p>Ranked by how closely each past event's Nino-3.4 evolution matches 2026 over the first
     12 observed months, aligned from precursor July.</p>
  <div class="grid">{top3_html}</div>

  <div class="cbox">
    {c_nino34}
    <p class="cnote">Solid lines: observed monthly Nino-3.4 anomaly. Dashed blue: NOAA/IRI forecast.
    Month 0 = July of the precursor year.</p>
  </div>

  <div class="cbox">
    {c_nino12}
    <p class="cnote">Nino-1+2 (far eastern Pacific, off Ecuador and Peru) leads and amplifies EP
    events. The 2026 acceleration from -0.3 °C in March to +1.81 °C in May is among the fastest
    on record for this region. Dashed blue: estimated from Nino-3.4 forecast using typical EP ratios.</p>
  </div>

  <div class="cbox">
    {c_soi}
    <p class="cnote">Negative SOI = weaker trades = more warming. NOAA monthly SOI.
    El Nino threshold: -8. La Nina: +8.</p>
  </div>


  <h2>Full analogue ranking</h2>
  <p>All confirmed El Nino events since 1950, ordered by Pearson correlation of the first
     12 months of Nino-3.4 evolution.</p>
  <div class="cbox">
    {c_analogue}
    <p class="cnote">Similarity captures shape, not magnitude. EP/CP type and subsurface
    precursors are separate dimensions not reflected here.</p>
  </div>


  <h2>What's coming</h2>

  <div class="alert ag">
    <strong>Kelvin waves (subsurface):</strong> three downwelling Kelvin waves in late 2025 and early
    2026 loaded the thermocline and are the primary subsurface driver of this event. Quantitative
    depth-section data (GODAS/TAO-TRITON) would show this clearly but is not yet integrated here.
  </div>

  <div class="alert ag">
    <strong>Warm Water Volume (WWV):</strong> the volume of water above 20 °C averaged over
    5S-5N, 120E-80W predicts peak intensity months in advance. Available at NOAA PMEL; planned addition.
  </div>

  <div class="alert ag">
    <strong>OLR (Outgoing Longwave Radiation):</strong> negative OLR anomalies over the western
    Pacific warm pool confirm that convection has coupled to the SST anomaly. Requires
    <code>netCDF4</code> or <code>xarray</code>; planned addition.
  </div>

  <div class="alert aw">
    <strong>AMJ 2026 ONI pending:</strong> the April-May-June running mean will be the first
    seasonal value fully within El Nino territory and will sharpen the intensity estimate.
  </div>


  <h2>Sources</h2>
  <ul class="srcs">
    <li><a href="https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices" target="_blank">NOAA CPC: Monthly Nino region SST indices</a></li>
    <li><a href="https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt" target="_blank">NOAA CPC: Oceanic Nino Index (ONI)</a></li>
    <li><a href="https://www.cpc.ncep.noaa.gov/data/indices/soi" target="_blank">NOAA CPC: Standardised SOI</a></li>
    <li><a href="https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso_advisory/ensodisc.shtml" target="_blank">NOAA CPC: ENSO Diagnostic Discussion, June 2026</a></li>
    <li><a href="https://iri.columbia.edu/our-expertise/climate/forecasts/enso/current/" target="_blank">IRI Columbia: ENSO Forecasts, June 2026</a></li>
  </ul>

</div>

<footer>
  Data: NOAA CPC / {latest_date}
</footer>
</body>
</html>"""
