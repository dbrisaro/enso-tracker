import base64
import io as _io
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np
from enso.data import ENSO_EVENTS, aligned_series, MEI_START_YEAR

MONTH_LABELS = [
    "Jul","Aug","Sep","Oct","Nov","Dec",
    "Jan","Feb","Mar","Apr","May","Jun",
    "Jul","Aug","Sep","Oct","Nov","Dec",
    "Jan","Feb","Mar","Apr","May","Jun",
]

N_OBS = 12

SHOW_EVENTS = ["1997–98", "2015–16", "1982–83", "2023–24"]

_LAYOUT = dict(
    plot_bgcolor="white", paper_bgcolor="white",
    hovermode="x unified",
    # The legend sits below the x axis rather than above the plot. Above, it
    # shares a band with the title and grows upward into it as it wraps, which
    # it now does: the IRI plume added two entries, and a narrow viewport wraps
    # further. Below, it has margin.autoexpand to grow into instead.
    legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0),
    font=dict(family="Hanken Grotesk, Barlow Condensed, sans-serif", size=12),
    margin=dict(l=60, r=90, t=60, b=70),
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
    return include


def _aligned_weekly(series: pd.Series, onset_yr: int, n: int = 24):
    """Map weekly observations onto the same x axis the monthly charts use,
    where x=0 is July of the precursor year and each unit is one month.
    Mid-month dates land on fractional positions."""
    start = pd.Timestamp(onset_yr - 1, 7, 1)
    xs, ys = [], []
    for date, val in series.dropna().items():
        x = ((date.year - start.year) * 12 + date.month - start.month
             + (date.day - 1) / date.days_in_month)
        if 0 <= x <= n - 1:
            xs.append(round(x, 3))
            ys.append(round(float(val), 2))
    return xs, ys



def _now_x(series: pd.Series, onset_yr: int = 2026, n: int = 24) -> float:
    """x position of the most recent observed month, for the "now" marker.

    Derived from the data rather than from a fixed N_OBS, which silently went
    stale each time CPC published a new month.
    """
    vals = aligned_series(series, onset_yr, n)
    last = max((i for i, v in enumerate(vals) if v is not None), default=N_OBS - 1)
    return last


def _rect(lat1, lat2, lon1, lon2, n=40):
    top    = ([lat1]*n,             list(np.linspace(lon1, lon2, n)))
    right  = (list(np.linspace(lat1, lat2, n)), [lon2]*n)
    bottom = ([lat2]*n,             list(np.linspace(lon2, lon1, n)))
    left   = (list(np.linspace(lat2, lat1, n)), [lon1]*n)
    lats = top[0] + right[0] + bottom[0] + left[0] + [lat1]
    lons = top[1] + right[1] + bottom[1] + left[1] + [lon1]
    return lats, lons


def chart_pacific_map(include_plotlyjs=False, div_id=None) -> str:
    fig = go.Figure()

    n = 40
    regions = [
        ("Niño 4",    5,   -5,     160,  -150,    "rgba(44,160,44,0.35)",  "#2ca02c",  "nino4",     True),
        ("Niño 3.4",  5,   -5,    -170,  -120,    "rgba(255,127,14,0.35)", "#ff7f0e",  None,        False),
        ("Niño 3",    5,   -5,    -150,   -90,    "rgba(214,39,40,0.30)",  "#d62728",  None,        False),
        ("Niño 1+2",  0,  -10,     -90,   -80,    "rgba(148,103,189,0.50)","#9467bd",  None,        False),
    ]
    for name, lat1, lat2, lon1, lon2, fc, lc, lg, dateline in regions:
        if dateline:
            halves = [(lat1, lat2, 160, 180), (lat1, lat2, -180, lon2)]
            for i, (a1, a2, l1, l2) in enumerate(halves):
                lats, lons = _rect(a1, a2, l1, l2)
                fig.add_trace(go.Scattergeo(
                    lat=lats, lon=lons, mode="lines", fill="toself",
                    fillcolor=fc, line=dict(color="rgba(0,0,0,0)", width=0),
                    name=name, legendgroup=lg, showlegend=(i == 0), hoverinfo="skip",
                ))
            for seg_lats, seg_lons in [
                (list(np.linspace(lat1, lat2, n)), [160]*n),
                ([lat1]*n, list(np.linspace(160, 180, n))),
                ([lat1]*n, list(np.linspace(-180, lon2, n))),
                (list(np.linspace(lat1, lat2, n)), [lon2]*n),
                ([lat2]*n, list(np.linspace(lon2, -180, n))),
                ([lat2]*n, list(np.linspace(180, 160, n))),
            ]:
                fig.add_trace(go.Scattergeo(
                    lat=seg_lats, lon=seg_lons, mode="lines",
                    line=dict(color=lc, width=1.5),
                    name=name, legendgroup=lg, showlegend=False, hoverinfo="skip",
                ))
        else:
            lats, lons = _rect(lat1, lat2, lon1, lon2)
            fig.add_trace(go.Scattergeo(
                lat=lats, lon=lons, mode="lines", fill="toself",
                fillcolor=fc, line=dict(color="rgba(0,0,0,0)", width=0),
                name=name, showlegend=True, hoverinfo="skip",
            ))
            fig.add_trace(go.Scattergeo(
                lat=lats, lon=lons, mode="lines",
                line=dict(color=lc, width=1.5),
                name=name, showlegend=False, hoverinfo="skip",
            ))

    for label, lon, color in [
        ("NIÑO 4",   175,  "#2ca02c"),
        ("NIÑO 3.4", -145, "#ff7f0e"),
        ("NIÑO 3",   -118, "#d62728"),
        ("NIÑO 1+2", -85,  "#9467bd"),
    ]:
        fig.add_trace(go.Scattergeo(
            lat=[14], lon=[lon], mode="text", text=[label],
            textfont=dict(color=color, size=12, family="Barlow Condensed, sans-serif"),
            showlegend=False, hoverinfo="skip",
        ))

    fig.add_trace(go.Scattergeo(
        lat=[-12.4, -17.5], lon=[130.8, -149.6],
        mode="markers+text",
        marker=dict(size=9, color="#333333", symbol="circle",
                    line=dict(width=1.5, color="white")),
        text=["Darwin", "Tahiti"], textposition=["top right", "top right"],
        textfont=dict(color="#333333", size=11, family="Hanken Grotesk, sans-serif"),
        name="SOI stations",
    ))
    fig.add_trace(go.Scattergeo(
        lat=[-12.4, -17.5], lon=[130.8, -149.6],
        mode="lines", line=dict(color="#555555", width=1.2, dash="dot"),
        showlegend=False, hoverinfo="skip",
    ))

    fig.update_geos(
        scope="world",
        projection_type="mercator",
        projection_rotation=dict(lon=180),
        lataxis_range=[-30, 20],
        lonaxis_range=[-100, 100],
        showland=True, landcolor="#D4C5A0",
        showocean=True, oceancolor="#9DC4E8",
        showcoastlines=True, coastlinecolor="rgba(80,80,80,0.55)",
        coastlinewidth=0.7,
        showlakes=False, showrivers=False,
        bgcolor="#9DC4E8",
        showframe=False,
    )
    fig.update_layout(
        title="ENSO Monitoring Regions and SOI Stations",
        height=380,
        margin=dict(l=0, r=0, t=50, b=55),
        legend=dict(orientation="h", yanchor="top", y=-0.05,
                    xanchor="center", x=0.5, bgcolor="rgba(255,255,255,0)"),
        font=dict(family="Hanken Grotesk, Barlow Condensed, sans-serif", size=12),
        paper_bgcolor="#9DC4E8",
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs=_pjs(include_plotlyjs),
                       div_id=div_id)


def chart_oni_full(oni: pd.DataFrame, include_plotlyjs=False, return_fig=False,
                   div_id=None) -> str:
    fig = go.Figure()

    dates = oni.index.tolist()
    vals  = oni["oni"].tolist()

    el_nino = [v if v >= 0.5 else 0.0 for v in vals]
    la_nina = [v if v <= -0.5 else 0.0 for v in vals]

    fig.add_trace(go.Scatter(
        x=dates, y=el_nino, fill="tozeroy",
        fillcolor="rgba(192,57,43,0.25)", line=dict(width=0),
        mode="none", name="El Niño (ONI >= +0.5)", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=la_nina, fill="tozeroy",
        fillcolor="rgba(30,111,165,0.25)", line=dict(width=0),
        mode="none", name="La Niña (ONI <= -0.5)", hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=dates, y=vals, mode="lines",
        name="ONI", line=dict(color="#1a2535", width=1.5),
        hovertemplate="%{x|%b %Y}: %{y:.2f} °C<extra></extra>",
    ))

    fig.add_hline(y= 0.5, line_dash="dash", line_color="rgba(192,57,43,0.5)", line_width=1)
    fig.add_hline(y=-0.5, line_dash="dash", line_color="rgba(30,111,165,0.5)", line_width=1)
    fig.add_hline(y= 0,   line_color="rgba(0,0,0,0.15)", line_width=1)

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
        title="Oceanic Niño Index: Full Record 1950 to present",
        height=360,
        plot_bgcolor="white", paper_bgcolor="white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=80, b=50),
        font=dict(family="Hanken Grotesk, Barlow Condensed, sans-serif", size=12),
    )
    if return_fig:
        return fig
    return pio.to_html(fig, full_html=False, include_plotlyjs=_pjs(include_plotlyjs),
                       div_id=div_id)


def _plume_xy(plume, onset_yr=2026):
    """Place each forecast season on the monthly x axis at its centre month."""
    start = pd.Timestamp(onset_yr - 1, 7, 1)
    xs, mu, lo, hi = [], [], [], []
    for s, m, a, b in zip(plume["seasons"], plume["oni_mean"],
                          plume["oni_q25"], plume["oni_q75"]):
        if m is None:
            continue
        c = s["centre"]
        xs.append((c.year - start.year) * 12 + c.month - start.month)
        mu.append(m); lo.append(a); hi.append(b)
    return xs, mu, lo, hi


def chart_nino34(nino34_series: pd.Series, show_events=None, include_plotlyjs=False,
                 weekly_series: pd.Series = None, plume: dict = None,
                 return_fig=False, div_id=None) -> str:
    if show_events is None:
        show_events = SHOW_EVENTS
    _now_series = nino34_series
    fig = go.Figure()

    for y, color, label in [
        ( 0.5, "rgba(192,57,43,0.55)",  "El Niño (+0.5 °C)"),
        ( 0.0, "rgba(0,0,0,0.18)",      ""),
        (-0.5, "rgba(30,111,165,0.45)", "La Niña (-0.5 °C)"),
    ]:
        fig.add_hline(y=y, line_dash="dash", line_color=color, line_width=1,
                      annotation_text=label, annotation_position="right",
                      annotation_font_size=10)

    fig.add_vline(x=_now_x(_now_series), line_dash="dot", line_color="rgba(30,111,165,0.65)",
                  line_width=1.5, annotation_text="now", annotation_position="top left")

    x = list(range(24))

    for event in show_events:
        meta = ENSO_EVENTS[event]
        vals = aligned_series(nino34_series, meta["onset_yr"], 24)
        fig.add_trace(go.Scatter(
            x=x, y=vals, mode="lines+markers", name=event,
            line=dict(color=meta["color"], width=2),
            marker=dict(size=3), opacity=0.85,
        ))

    obs_2026 = aligned_series(nino34_series, 2026, 24)
    fig.add_trace(go.Scatter(x=x, y=obs_2026, mode="lines+markers",
        name="2026-27 (monthly)", line=dict(color="#1e6fa5", width=3.5),
        marker=dict(size=5)))

    if weekly_series is not None:
        wx, wy = _aligned_weekly(weekly_series, 2026, 24)
        if wx:
            fig.add_trace(go.Scatter(x=wx, y=wy, mode="lines",
                name="2026-27 (weekly)", line=dict(color="#1e6fa5", width=1.4, dash="dot"),
                opacity=0.9, hovertemplate="weekly %{y:+.2f} °C<extra></extra>"))

    if plume is not None:
        px, pmu, plo, phi = _plume_xy(plume)
        if px:
            # Anchor the plume to the last observed month so the forecast reads as
            # a continuation rather than a floating segment.
            last_x = len(obs_2026) - 1
            while last_x >= 0 and obs_2026[last_x] is None:
                last_x -= 1
            anchor_x = [last_x] if last_x >= 0 else []
            anchor_y = [obs_2026[last_x]] if last_x >= 0 else []

            fig.add_trace(go.Scatter(
                x=px + px[::-1], y=phi + plo[::-1], fill="toself",
                fillcolor="rgba(181,23,158,0.12)", line=dict(width=0),
                mode="none", name="IRI middle 50% of models", hoverinfo="skip"))
            fig.add_trace(go.Scatter(
                x=anchor_x + px, y=anchor_y + pmu, mode="lines+markers",
                name=f"IRI plume ({plume['issue_label']})",
                line=dict(color="#b5179e", width=2.6, dash="dash"),
                marker=dict(size=5, symbol="diamond"),
                hovertemplate="forecast %{y:+.2f} °C<extra></extra>"))

    fig.update_xaxes(**_XAXIS)
    fig.update_yaxes(title_text="Niño-3.4 anomaly (°C)", range=[-1.6, 3.5], **_YAXIS)
    fig.update_layout(title="Niño-3.4 SST Anomaly: 2026 vs Historical Analogues",
                      height=490, **_LAYOUT)
    if return_fig:
        return fig
    return pio.to_html(fig, full_html=False, include_plotlyjs=_pjs(include_plotlyjs),
                       div_id=div_id)


def chart_nino12(nino12_series: pd.Series, show_events=None, include_plotlyjs=False,
                 return_fig=False, div_id=None) -> str:
    if show_events is None:
        show_events = SHOW_EVENTS
    _now_series = nino12_series
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

    fig.add_vline(x=_now_x(_now_series), line_dash="dot", line_color="rgba(30,111,165,0.65)",
                  line_width=1.5, annotation_text="now", annotation_position="top left")

    x = list(range(24))

    first_year = nino12_series.index.year.min()
    last_year  = nino12_series.index.year.max() - 1
    highlight_onsets = {ENSO_EVENTS[e]["onset_yr"] for e in show_events}

    warm_years = {}
    for yr in range(first_year, last_year + 1):
        vals = aligned_series(nino12_series, yr, 24)
        peak = max((v for v in vals if v is not None), default=None)
        if peak is not None and peak >= 1.5:
            warm_years[yr] = vals

    for yr, vals in warm_years.items():
        if yr == 2026 or yr in highlight_onsets:
            continue
        label = f"{yr}-{str(yr+1)[-2:]}"
        fig.add_trace(go.Scatter(
            x=x, y=vals, mode="lines", name=label,
            line=dict(color="rgba(150,150,150,0.35)", width=1),
            showlegend=False,
            hovertemplate=f"<b>{label}</b>: %{{y:.2f}} °C<extra></extra>",
        ))

    for event in show_events:
        meta = ENSO_EVENTS[event]
        vals = aligned_series(nino12_series, meta["onset_yr"], 24)
        fig.add_trace(go.Scatter(
            x=x, y=vals, mode="lines+markers", name=event,
            line=dict(color=meta["color"], width=2.5),
            marker=dict(size=3.5), opacity=0.9,
        ))

    obs_2026 = aligned_series(nino12_series, 2026, 24)
    fig.add_trace(go.Scatter(x=x, y=obs_2026, mode="lines+markers",
        name="2026-27 (observed)", line=dict(color="#1e6fa5", width=3.5),
        marker=dict(size=5)))

    fig.update_xaxes(**_XAXIS)
    fig.update_yaxes(title_text="Niño-1+2 anomaly (°C)", range=[-2.5, 5.5], **_YAXIS)
    fig.update_layout(
        title="Niño 1+2 SST Anomaly: 2026 vs All Events Since 1950",
        height=510, **_LAYOUT,
    )
    if return_fig:
        return fig
    return pio.to_html(fig, full_html=False, include_plotlyjs=_pjs(include_plotlyjs),
                       div_id=div_id)


def chart_mei(mei_series: pd.Series, show_events=None, include_plotlyjs=False, div_id=None) -> str:
    if show_events is None:
        show_events = SHOW_EVENTS
    _now_series = mei_series
    fig = go.Figure()

    for y, color, label in [
        ( 1.5, "rgba(192,57,43,0.55)",  "Strong El Niño (+1.5)"),
        ( 0.5, "rgba(192,57,43,0.30)",  "El Niño (+0.5)"),
        ( 0.0, "rgba(0,0,0,0.18)",      ""),
        (-0.5, "rgba(30,111,165,0.45)", "La Niña (-0.5)"),
    ]:
        fig.add_hline(y=y, line_dash="dash", line_color=color, line_width=1,
                      annotation_text=label, annotation_position="right",
                      annotation_font_size=10)

    fig.add_vline(x=_now_x(_now_series), line_dash="dot", line_color="rgba(30,111,165,0.65)",
                  line_width=1.5, annotation_text="now", annotation_position="top left")

    x = list(range(24))
    for event in show_events:
        meta = ENSO_EVENTS[event]
        vals = aligned_series(mei_series, meta["onset_yr"], 24)
        if not any(v is not None for v in vals):
            continue  # pre-1979 event, outside the MEI record
        fig.add_trace(go.Scatter(x=x, y=vals, mode="lines+markers", name=event,
            line=dict(color=meta["color"], width=2), marker=dict(size=3), opacity=0.85))

    obs_2026 = aligned_series(mei_series, 2026, 24)
    fig.add_trace(go.Scatter(x=x, y=obs_2026, mode="lines+markers",
        name="2026-27 (observed)", line=dict(color="#1e6fa5", width=3.5), marker=dict(size=5)))

    fig.update_xaxes(**_XAXIS)
    fig.update_yaxes(title_text="MEI.v2 (standard deviations)", range=[-2.8, 3.0], **_YAXIS)
    fig.update_layout(title="Multivariate ENSO Index: 2026 vs Historical Analogues",
                      height=490, **_LAYOUT)
    return pio.to_html(fig, full_html=False, include_plotlyjs=_pjs(include_plotlyjs),
                       div_id=div_id)


def chart_soi(soi_series: pd.Series, show_events=None, include_plotlyjs=False, div_id=None) -> str:
    if show_events is None:
        show_events = SHOW_EVENTS
    _now_series = soi_series
    fig = go.Figure()

    for y, color, label in [
        (-1.0, "rgba(192,57,43,0.55)",  "El Niño (-1)"),
        ( 0,   "rgba(0,0,0,0.18)",      ""),
        ( 1.0, "rgba(30,111,165,0.45)", "La Niña (+1)"),
    ]:
        fig.add_hline(y=y, line_dash="dash", line_color=color, line_width=1,
                      annotation_text=label, annotation_position="right",
                      annotation_font_size=10)

    fig.add_vline(x=_now_x(_now_series), line_dash="dot", line_color="rgba(30,111,165,0.65)",
                  line_width=1.5, annotation_text="now", annotation_position="top left")

    x = list(range(24))
    for event in show_events:
        meta = ENSO_EVENTS[event]
        vals = aligned_series(soi_series, meta["onset_yr"], 24)
        fig.add_trace(go.Scatter(x=x, y=vals, mode="lines+markers", name=event,
            line=dict(color=meta["color"], width=2), marker=dict(size=3), opacity=0.85))

    obs_2026 = aligned_series(soi_series, 2026, 24)
    fig.add_trace(go.Scatter(x=x, y=obs_2026, mode="lines+markers",
        name="2026-27 (observed)", line=dict(color="#1e6fa5", width=3.5), marker=dict(size=5)))

    fig.update_xaxes(**_XAXIS)
    fig.update_yaxes(title_text="SOI (standardised)", range=[-6.5, 4.5], **_YAXIS)
    fig.update_layout(title="Southern Oscillation Index: 2026 vs Historical Analogues",
                      height=470, **_LAYOUT)
    return pio.to_html(fig, full_html=False, include_plotlyjs=False, div_id=div_id)


def chart_nino_bar(sst: pd.DataFrame, include_plotlyjs=False, div_id=None) -> str:
    latest = sst.iloc[-1]
    date   = sst.index[-1].strftime("%B %Y")
    regions = ["Niño 1+2", "Niño 3", "Niño 3.4", "Niño 4"]
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
    return pio.to_html(fig, full_html=False, include_plotlyjs=False, div_id=div_id)


def chart_analogues(analogues: list, include_plotlyjs=False, div_id=None) -> str:
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
    return pio.to_html(fig, full_html=False, include_plotlyjs=False, div_id=div_id)


def chart_sst_comparison(sst_maps: list) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    cmap = plt.cm.RdBu_r
    norm = mcolors.TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)
    proj = ccrs.PlateCarree(central_longitude=180)
    data_crs = ccrs.PlateCarree()

    now = sst_maps[0]
    pairs = [(now, sst_maps[1]), (now, sst_maps[2])]

    fig, axes = plt.subplots(2, 2, figsize=(12, 6.4),
                             subplot_kw={"projection": proj}, facecolor="white")
    fig.subplots_adjust(wspace=0.06, hspace=0.28, bottom=0.12)

    im = None
    for row, (left, right) in enumerate(pairs):
        for col, m in enumerate([left, right]):
            ax = axes[row, col]
            ax.set_extent([110, 290, -35, 35], crs=data_crs)
            im = ax.pcolormesh(m["data"]["lon"], m["data"]["lat"], m["data"]["ssta"],
                               transform=data_crs, cmap=cmap, norm=norm,
                               rasterized=True, zorder=0)
            ax.add_feature(cfeature.LAND, facecolor="#D4C5A0", zorder=1)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.55, edgecolor="#666666", zorder=2)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="#999999", zorder=2)
            gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5, zorder=3)
            gl.top_labels = False
            gl.right_labels = False
            gl.left_labels = (col == 0)
            gl.bottom_labels = (row == 1)
            gl.xlabel_style = {"size": 8}
            gl.ylabel_style = {"size": 8}
            ax.set_title(m["label"], fontsize=10, pad=6, fontfamily="DejaVu Sans")

    cbar_ax = fig.add_axes([0.2, 0.04, 0.6, 0.03])
    cb = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cb.set_label("SST anomaly (°C)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    b64 = base64.b64encode(buf.getvalue()).decode()
    return (f'<img src="data:image/png;base64,{b64}" '
            f'style="width:100%;max-width:880px;display:block;margin:0 auto;" '
            f'alt="SST anomaly comparison">')


def _sst_maps_section(maps: list) -> str:
    if not maps or not all(m["data"] is not None for m in maps):
        return ""
    img = chart_sst_comparison(maps)
    return f"""
  <p>The maps below show the full Pacific SST anomaly pattern at the same point in the event
     cycle. Comparing the spatial footprint helps assess whether the warming is concentrated
     in the east (EP) or spread toward the dateline (CP).</p>
  <div class="cbox" style="padding:1rem .5rem .5rem;">
    {img}
    <p class="cnote">SST anomaly relative to 1991-2020 climatology. Source: NOAA ERSSTV5.</p>
  </div>"""


def build_html(sst: pd.DataFrame, soi_df: pd.DataFrame, oni: pd.DataFrame,
               analogues: list, *, cdn: bool = False,
               sst_maps_oni: list = None, sst_maps_roni: list = None,
               analogues_roni: list = None, weekly: pd.DataFrame = None,
               mei: pd.Series = None, plume: dict = None) -> str:

    latest      = sst.iloc[-1]
    latest_date = sst.index[-1].strftime("%B %Y")
    top3        = analogues[:3]

    nv = float(latest["nino34_anom"])
    if   nv >= 2.0:  phase, ph_col = "Very Strong El Niño", "#c0392b"
    elif nv >= 1.5:  phase, ph_col = "Strong El Niño",      "#e74c3c"
    elif nv >= 0.9:  phase, ph_col = "Moderate El Niño",    "#e07b39"
    elif nv >= 0.5:  phase, ph_col = "Weak El Niño",        "#f39c12"
    elif nv <= -1.5: phase, ph_col = "Strong La Niña",      "#1e6fa5"
    elif nv <= -0.5: phase, ph_col = "La Niña",             "#2980b9"
    else:            phase, ph_col = "Neutral",             "#7f8c8d"

    pjs_mode   = "cdn" if cdn else True

    # Every chart gets a fixed div id. Left to itself Plotly mints a new UUID per
    # figure on every build, so the published page differed on every run even in
    # weeks when no feed had updated anything. The ONI and RONI rankings draw the
    # same charts twice, hence the suffixes: the ids have to stay unique.
    c_map      = chart_pacific_map(include_plotlyjs=pjs_mode, div_id="pacific-map")
    c_oni_full = chart_oni_full(oni, div_id="oni-full")

    show_events_oni  = [r["event"] for r in analogues[:4]]
    show_events_roni = [r["event"] for r in analogues_roni[:4]] if analogues_roni else SHOW_EVENTS

    wk_n34         = weekly["nino34_anom"] if weekly is not None else None
    c_nino34_oni   = chart_nino34(sst["nino34_anom"], show_events=show_events_oni,
                                  weekly_series=wk_n34, plume=plume,
                                  div_id="nino34-oni")
    c_nino12_oni   = chart_nino12(sst["nino12_anom"], show_events=show_events_oni,
                                  div_id="nino12-oni")
    c_soi_oni      = chart_soi(soi_df["soi"], show_events=show_events_oni, div_id="soi-oni")
    c_mei_oni      = chart_mei(mei, show_events=show_events_oni, div_id="mei-oni") if mei is not None else ""
    c_analogue_oni = chart_analogues(analogues, div_id="analogues-oni")
    sst_sec_oni    = _sst_maps_section(sst_maps_oni)

    if analogues_roni:
        c_nino34_roni   = chart_nino34(sst["nino34_anom"], show_events=show_events_roni,
                                       weekly_series=wk_n34, plume=plume,
                                       div_id="nino34-roni")
        c_nino12_roni   = chart_nino12(sst["nino12_anom"], show_events=show_events_roni,
                                       div_id="nino12-roni")
        c_soi_roni      = chart_soi(soi_df["soi"], show_events=show_events_roni, div_id="soi-roni")
        c_mei_roni      = chart_mei(mei, show_events=show_events_roni, div_id="mei-roni") if mei is not None else ""
        c_analogue_roni = chart_analogues(analogues_roni, div_id="analogues-roni")
        sst_sec_roni    = _sst_maps_section(sst_maps_roni)
        has_roni        = True
    else:
        c_nino34_roni = c_nino12_roni = c_soi_roni = c_analogue_roni = sst_sec_roni = ""
        c_mei_roni = ""
        has_roni = False

    soi_latest = float(soi_df["soi"].iloc[-1])
    soi_date   = soi_df.index[-1].strftime("%B %Y")

    # Describe the forecast from the plume that is actually loaded. These two
    # captions previously claimed a dashed forecast line that had been removed
    # from the data pipeline, so they must stay derived rather than hardcoded.
    if plume is not None and any(v is not None for v in plume["oni_mean"]):
        pairs      = [(m, s) for m, s in zip(plume["oni_mean"], plume["seasons"])
                      if m is not None]
        peak, pks  = max(pairs, key=lambda t: t[0])
        fc_note    = (f"Solid: observed monthly. Dotted: weekly observations. "
                      f"Dashed magenta: IRI plume issued {plume['issue_label']}, "
                      f"ensemble mean of {plume['n_oni_models']} dynamical and "
                      f"statistical models by three-month season, shaded across "
                      f"the middle 50% of members.")
        pk_i    = plume["oni_mean"].index(peak)
        full_lo = plume["oni_min"][pk_i]
        full_hi = plume["oni_max"][pk_i]
        # The JSON feed also carries relative-ONI members, which have to be
        # filtered out; the figure the fallback reads never contained them, so
        # only the JSON build has an exclusion to declare.
        baseline = ("" if plume.get("source") != "plumes_json" else
                    " Relative-ONI models are excluded, since they sit on a "
                    "different baseline.")
        fc_sentence = (f"The dashed magenta line is the IRI forecast plume issued "
                       f"{plume['issue_label']}, averaged over "
                       f"{plume['n_oni_models']} dynamical and statistical models. "
                       f"It peaks at {peak:+.2f} °C in {pks['code']} "
                       f"{pks['centre']:%Y}, where individual members run from "
                       f"{full_lo:+.2f} to {full_hi:+.2f} °C. The shaded band is "
                       f"the middle 50% of members, not that full range, which "
                       f"one or two outliers would otherwise dominate." + baseline)
    else:
        fc_note     = "Solid: observed monthly. Dotted: weekly observations."
        fc_sentence = "No forecast plume is loaded for this build."

    # MEI seasons are two-month means stored against their second month, so a
    # value indexed to June is the May-June season and has to be labelled that
    # way rather than as a single month.
    mei_card = mei_sec = mei_roni_sec = ""
    if mei is not None and len(mei):
        mei_last  = float(mei.iloc[-1])
        mei_d     = mei.index[-1]
        mei_seas  = (f"{(mei_d - pd.DateOffset(months=1)).strftime('%b')}"
                     f"–{mei_d.strftime('%b %Y')}")
        if   mei_last >=  2.0: mei_phase = "very strong El Niño territory"
        elif mei_last >=  1.5: mei_phase = "strong El Niño territory"
        elif mei_last >=  1.0: mei_phase = "moderate El Niño territory"
        elif mei_last >=  0.5: mei_phase = "weak El Niño territory"
        elif mei_last <= -0.5: mei_phase = "La Niña territory"
        else:                  mei_phase = "neutral territory"

        mei_prev = float(mei.iloc[-2]) if len(mei) > 1 else mei_last
        # Either toggle set can contain a pre-1979 analogue, so name every event
        # that gets silently skipped in whichever chart the reader is looking at.
        dropped  = sorted({e for e in set(show_events_oni) | set(show_events_roni)
                           if ENSO_EVENTS[e]["onset_yr"] - 1 < MEI_START_YEAR})
        drop_note = (" MEI.v2 begins with the Dec 1978–Jan 1979 season, so "
                     + ", ".join(dropped)
                     + (" falls" if len(dropped) == 1 else " fall")
                     + " outside the record and cannot be drawn here."
                     ) if dropped else ""

        mei_card = f"""
    <div class="card"><div class="lbl">MEI.v2 ({mei_seas})</div>
      <div class="val">{mei_last:+.2f}</div>
      <div class="note">Multivariate / in std deviations</div></div>"""

        mei_sec = f"""
  <h2>The multivariate view: MEI</h2>
  <p class="takehome">Every index above measures one thing. The MEI combines five
     of them, and at {mei_last:+.2f} for {mei_seas} it puts 2026 in {mei_phase},
     confirming the ocean signal is not running ahead of the atmosphere on its own.</p>
  <p>The Multivariate ENSO Index (MEI.v2) folds sea surface temperature, sea level
     pressure, surface zonal and meridional winds, and outgoing longwave radiation
     into a single number over the tropical Pacific (30°S–30°N), expressed in
     standard deviations. Because it weighs atmosphere alongside ocean, it is
     slower to respond than Niño-3.4 but harder to fool: a warm patch of water
     that the trade winds have not yet reacted to will lift Niño-3.4 without
     moving the MEI much. The jump from {mei_prev:+.2f} in the previous season to
     {mei_last:+.2f} is the coupled system locking in.</p>
  <div class="cbox">
    {c_mei_oni}
    <p class="cnote">Two-month overlapping seasons, plotted against the second month
       of each pair. Analogues shown are the top ONI matches.{drop_note}
       Source: NOAA PSL.</p>
  </div>"""

        # The MEI section above is pinned to the ONI ranking. The RONI toggle
        # reorders which events get drawn, so it needs its own copy of the chart.
        mei_roni_sec = f"""
    <p>MEI.v2 for the RONI analogues. The index is already expressed in standard deviations
       of the coupled signal, so it shifts less between eras than Niño-3.4 does; what changes
       here is which events the detrended ranking puts on the chart.</p>
    <div class="cbox">
      {c_mei_roni}
      <p class="cnote">Two-month overlapping seasons, plotted against the second month
         of each pair. Analogues shown are the top RONI matches.{drop_note}
         Source: NOAA PSL.</p>
    </div>"""

    # The monthly indices lag by up to five weeks. When the weekly product runs
    # ahead of them, surface it so the page is not quoting a stale figure as
    # "current".
    weekly_box  = ""
    weekly_note = ""
    if weekly is not None and len(weekly):
        wk_last = weekly.iloc[-1]
        wk_date = weekly.index[-1]
        if wk_date > sst.index[-1] + pd.offsets.MonthEnd(0):
            wk_n34v = float(wk_last["nino34_anom"])
            delta   = wk_n34v - nv
            months_ahead = sorted({
                d.strftime("%B %Y") for d in weekly.index
                if d > sst.index[-1] + pd.offsets.MonthEnd(0)})
            partial = ", ".join(months_ahead)
            weekly_note = (f" The weekly product already covers {partial}; see below.")
            weekly_box = f"""
  <div class="infobox" style="border-left:3px solid #c0392b;">
    <strong>Ahead of the monthly release.</strong>
    CPC monthly indices are only current through {latest_date}, but the weekly
    product runs to the week centred on <strong>{wk_date.strftime('%d %B %Y')}</strong>
    and shows the event still strengthening. Latest weekly values:
    Niño-3.4 <strong>{wk_n34v:+.2f} °C</strong>
    ({delta:+.2f} °C vs the {latest_date} monthly mean),
    Niño-1+2 {float(wk_last['nino12_anom']):+.2f} °C,
    Niño-3 {float(wk_last['nino3_anom']):+.2f} °C,
    Niño-4 {float(wk_last['nino4_anom']):+.2f} °C.
    Weekly values are noisier than monthly means and are not used for the
    analogue ranking or the ONI, both of which require full months.
  </div>"""
    ratio      = float(latest["nino12_anom"]) / max(abs(float(latest["nino4_anom"])), 0.1)

    if nv >= 2.0:
        cond_takehome = f"At {nv:+.2f} °C Niño-3.4 is in very strong El Niño territory, tracking alongside the most intense events on record."
    elif nv >= 1.5:
        cond_takehome = f"At {nv:+.2f} °C Niño-3.4 has reached strong El Niño intensity, with the eastern Pacific (Niño-1+2 at {float(latest['nino12_anom']):+.2f} °C) running well ahead of the central Pacific."
    elif nv >= 0.9:
        cond_takehome = f"Moderate El Niño is established. Niño-3.4 sits at {nv:+.2f} °C and the eastern Pacific (Niño-1+2 at {float(latest['nino12_anom']):+.2f} °C) is running {ratio:.1f}x warmer than the central Pacific, a classic Eastern Pacific signature."
    elif nv >= 0.5:
        cond_takehome = f"Weak El Niño conditions at {nv:+.2f} °C. The eastern Pacific is tracking ahead but the event has not yet reached moderate intensity."
    else:
        cond_takehome = f"Conditions are currently neutral with Niño-3.4 at {nv:+.2f} °C. No El Niño declared."

    roni_top    = analogues_roni[:3] if analogues_roni else []
    roni1       = roni_top[0]["event"] if roni_top else "—"
    roni1_score = roni_top[0]["score"] if roni_top else 0
    shared      = {r["event"] for r in top3} & {r["event"] for r in roni_top}
    shared_note = (f" {', '.join(sorted(shared))} {'appears' if len(shared)==1 else 'appear'} in both lists."
                   if shared else " No event appears in both lists.")
    analogue_takehome = (
        f"Measured against the raw Niño-3.4 trajectory, 2026 tracks closest to {top3[0]['event']} "
        f"(r = {top3[0]['score']:.2f}). When the long-term warming background is removed "
        f"(RONI), the closest match shifts to {roni1} (r = {roni1_score:.2f}).{shared_note}"
    ) if analogues_roni else (
        f"Over the first 12 observed months, 2026 tracks most closely with {top3[0]['event']} "
        f"(r = {top3[0]['score']:.2f})."
    )

    def _cards(ranks):
        html = ""
        for r in ranks:
            html += f"""
        <div style="border:1px solid #e0e0e0;border-radius:3px;padding:1.1rem 1.25rem;background:#fff;">
          <div style="font-family:'Barlow',sans-serif;font-weight:600;font-size:.65rem;
               letter-spacing:.1em;text-transform:uppercase;padding:.15rem .5rem;
               border:1px solid {r['color']};border-radius:2px;color:{r['color']};
               display:inline-block;margin-bottom:.5rem;">#{r['rank']}</div>
          <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.1rem;
               font-weight:400;text-transform:uppercase;letter-spacing:.04em;
               color:#000;margin-bottom:.2rem;">{r['event']}</div>
          <div style="font-size:.88rem;color:#555;line-height:1.55;">
            Peak ONI: {r['peak_oni'] or '?'} °C / {r['type']}-type
          </div>
          <div style="font-size:.78rem;color:#777;margin-top:.2rem;">similarity: {r['score']:.2f}</div>
        </div>"""
        return html

    top3_html      = _cards(top3)
    top3_roni_html = _cards(roni_top)

    roni_view_html = f"""
  <div id="view-roni" style="display:none;">
    <p>The Relative ONI removes the tropical-mean warming trend before comparing events.
       The closest RONI matches share a similar ENSO forcing pattern with 2026, independent
       of the era they occurred in.</p>
    <div class="grid">{top3_roni_html}</div>
{sst_sec_roni}
    <p>Niño-3.4 trajectories for the top RONI analogues. Older events show lower absolute values
       due to the cooler mean state of their era; RONI isolates the shape of the forcing, not the level.</p>
    <div class="cbox">
      {c_nino34_roni}
      <p class="cnote">{fc_note} Month 0 = July of precursor year.</p>
    </div>
    <p>Niño 1+2 for RONI analogues. The spatial pattern of eastern Pacific warming is what RONI
       targets, so this region is particularly relevant for the comparison.</p>
    <div class="cbox">
      {c_nino12_roni}
      <p class="cnote">Observed monthly data. Source: NOAA CPC.</p>
    </div>
    <p>SOI for RONI analogues. Atmospheric coupling is captured here regardless of background SST era.</p>
    <div class="cbox">
      {c_soi_roni}
      <p class="cnote">El Niño threshold -1, La Niña +1. Standardised SOI. Source: NOAA CPC.</p>
    </div>
{mei_roni_sec}
    <h2>Full analogue ranking (RONI)</h2>
    <p class="takehome">RONI rankings often surface older events from the 1960s and 1970s. This is expected: stripped of the warming trend, those events look more like 2026 in terms of forcing pattern.</p>
    <p>All confirmed El Niño events since 1950, ordered by RONI correlation with 2026.</p>
    <div class="cbox">
      {c_analogue_roni}
      <p class="cnote">RONI removes the global warming trend from Niño-3.4 before ranking. Similarity captures forcing shape, not magnitude.</p>
    </div>
  </div>""" if has_roni else ""

    toggle_html = f"""
  <div class="toggle-row">
    <button id="btn-oni"  class="btn-toggle active" onclick="switchView('oni')">ONI analogues</button>
    <button id="btn-roni" class="btn-toggle"        onclick="switchView('roni')">RONI analogues</button>
  </div>
  <p style="font-size:.82rem;color:#666;margin-top:-.25rem;margin-bottom:1rem;">
    ONI uses raw Niño-3.4 values. RONI removes the tropical warming trend to isolate the ENSO
    forcing pattern. Toggle to compare which events each approach considers most similar to 2026.
  </p>""" if has_roni else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>ENSO 2026 Tracker</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400&family=Barlow:wght@600&family=Hanken+Grotesk:ital,wght@0,300;0,400;0,500;1,300&display=swap" rel="stylesheet"/>
<style>
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{font-family:"Hanken Grotesk",sans-serif;font-weight:300;background:#fff;color:#000;line-height:1.7;font-size:17px;}}
  .page{{max-width:920px;margin:0 auto;padding:2.5rem 1.5rem 4rem;}}
  h1,h2,h3{{font-family:"Barlow Condensed",sans-serif;font-weight:400;text-transform:uppercase;letter-spacing:.04em;color:#000;}}
  h1{{font-size:2.2rem;margin-bottom:.5rem;}}
  h2{{font-size:1.4rem;border-bottom:2px solid #5BA8B0;padding-bottom:.35rem;margin-top:2.25rem;margin-bottom:1rem;}}
  h3{{font-size:1.1rem;margin-bottom:.25rem;}}
  p{{margin:0 0 1rem;font-size:.94rem;}}
  strong{{font-weight:400;}}
  a{{color:#2B6B72;text-decoration:none;}}
  a:hover{{color:#000;}}
  .badge{{display:inline-block;padding:.15rem .5rem;font-family:"Barlow",sans-serif;font-weight:600;
          font-size:.65rem;letter-spacing:.1em;text-transform:uppercase;
          border:1px solid {ph_col};border-radius:2px;color:{ph_col};margin-bottom:1rem;}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:.75rem;margin:1rem 0 1.5rem;}}
  .card{{border:1px solid #e0e0e0;border-radius:3px;padding:1rem 1.25rem;background:#fff;}}
  .lbl{{font-family:"Barlow Condensed",sans-serif;font-size:.78rem;letter-spacing:.06em;text-transform:uppercase;color:#5BA8B0;}}
  .val{{font-family:"Barlow Condensed",sans-serif;font-size:1.5rem;font-weight:400;color:#000;line-height:1.1;margin:.1rem 0;}}
  .note{{font-size:.78rem;color:#777;margin-top:.1rem;}}
  .cbox{{border:1px solid #e0e0e0;border-radius:3px;padding:.5rem;margin:.8rem 0;background:#fff;}}
  .cnote{{font-size:.8rem;color:#777;margin-top:.35rem;line-height:1.4;padding:0 .5rem .5rem;}}
  .infobox{{border-left:3px solid #2B6B72;padding:.75rem 1rem;margin:.8rem 0;font-size:.88rem;color:#000;}}
  .alert{{border-left:3px solid #e0e0e0;padding:.65rem 1rem;margin:.4rem 0;font-size:.88rem;color:#000;}}
  .ag{{border-left-color:#2B6B72;}}
  .aw{{border-left-color:#5BA8B0;}}
  .srcs{{list-style:none;font-size:.88rem;color:#555;}}
  .srcs li{{padding:.25rem 0 .25rem 1rem;position:relative;border-bottom:1px solid #efefef;}}
  .srcs li:last-child{{border-bottom:none;}}
  .srcs li::before{{content:"›";position:absolute;left:0;color:#5BA8B0;}}
  .srcs a{{color:#2B6B72;text-decoration:none;}}
  .srcs a:hover{{color:#000;}}
  .takehome{{font-weight:400;margin:-.25rem 0 1rem;}}
  footer{{text-align:center;padding:1.5rem;font-size:.78rem;color:#777;
          border-top:1px solid #e0e0e0;margin-top:2rem;
          font-family:"Barlow Condensed",sans-serif;letter-spacing:.06em;text-transform:uppercase;}}
  .toggle-row{{display:flex;gap:.5rem;margin:1.25rem 0 .4rem;}}
  .btn-toggle{{font-family:"Barlow Condensed",sans-serif;font-size:.85rem;letter-spacing:.06em;
               text-transform:uppercase;padding:.35rem .9rem;border:1px solid #5BA8B0;
               border-radius:2px;background:#fff;color:#5BA8B0;cursor:pointer;transition:background .15s,color .15s;}}
  .btn-toggle.active{{background:#5BA8B0;color:#fff;}}
  .btn-toggle:hover:not(.active){{background:#f0f8f9;}}
</style>
</head>
<body>

<div style="background:#000;color:#fff;padding:2rem 1.5rem 1.75rem;text-align:center;">
  <div style="font-family:'Barlow Condensed',sans-serif;font-size:.78rem;letter-spacing:.1em;
       text-transform:uppercase;color:rgba(255,255,255,0.55);margin-bottom:.75rem;">
    ENSO Monitor / {latest_date}
  </div>
  <h1 style="color:#fff;font-size:clamp(1.8rem,4.5vw,2.8rem);margin-bottom:.5rem;">
    El Niño 2026: Conditions and Historical Analogues
  </h1>
  <p style="font-family:'Hanken Grotesk',sans-serif;font-weight:300;color:rgba(255,255,255,0.7);
       max-width:560px;margin:0 auto;font-size:.94rem;">
    Live NOAA data compared to every major ENSO event since 1950
  </p>
</div>

<div class="page">

  <div class="badge">{phase}</div>
  <p>Monthly data current through <strong>{latest_date}</strong>. Source: NOAA CPC.{weekly_note}</p>


  <h2>How El Niño is monitored</h2>
  <p class="takehome">Of all the indices tracked here, Niño-3.4 is the one that matters most. It needs to stay above +0.5 °C for five consecutive overlapping seasons before an El Niño is officially declared. The Southern Oscillation Index (SOI), a measure of the atmospheric pressure difference between Tahiti and Darwin, acts as a cross-check: if the atmosphere has responded to the ocean warming, it will have turned sharply negative too.</p>
  <p>The four Niño regions are fixed boxes in the equatorial Pacific, each capturing a different
     slice of surface temperature. The SOI measures the pressure difference between
     <strong>Darwin</strong> and <strong>Tahiti</strong>. When that gradient reverses, trade winds
     weaken, warm water spreads east, and what starts as an ocean anomaly gets amplified by the
     atmosphere until you have a full El Niño.</p>
  <div class="cbox" style="padding:.5rem;">
    {c_map}
  </div>


  <h2>Current conditions</h2>
  <p class="takehome">{cond_takehome}</p>
  <div class="grid">
    <div class="card"><div class="lbl">Niño-3.4 ({latest_date})</div>
      <div class="val">{latest['nino34_anom']:+.2f} °C</div>
      <div class="note">Primary index / monthly mean / El Niño above +0.5 °C</div></div>
    <div class="card"><div class="lbl">Niño-1+2</div>
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
      <div class="note">El Niño below -1 (standardised)</div></div>
    <div class="card"><div class="lbl">Prob. Very Strong</div>
      <div class="val">63 %</div>
      <div class="note">NOAA CPC / NDJ 2026-27</div></div>{mei_card}
  </div>
{weekly_box}

  <div class="infobox">
    Not all El Niños are the same. In an Eastern Pacific event the warmth sits off South America
    (Niño-1+2 and Nino-3); in a Central Pacific event it pools near the dateline (Nino-4).
    Right now Niño-1+2 ({latest['nino12_anom']:+.2f} °C) is running {ratio:.1f}x warmer than
    Nino-4 ({latest['nino4_anom']:+.2f} °C), a strong Eastern Pacific signature consistent with 1997-98.
    EP events tend to be stronger and longer-lived than CP events.
  </div>


{mei_sec}

  <h2>Historical context: 75 years of ENSO</h2>
  <p class="takehome">Strong El Niño events above +1.5 °C are rare, only four in 75 years of records. What this chart shows is that 2026 is not heading toward an ordinary event.</p>
  <p>The chart plots the Oceanic Niño Index (ONI), a 3-month running mean of Niño-3.4 sea surface
     temperature anomalies and the official NOAA benchmark for declaring El Niño or La Niña
     conditions. Every event in the observational record is here, showing how frequently they occur,
     how long they last, and how intense they get. The four annotated peaks are the events
     most comparable to 2026.</p>
  <div class="cbox">
    {c_oni_full}
    <p class="cnote">Source: NOAA CPC.</p>
  </div>


  <h2>Historical analogues</h2>
  <p class="takehome">{analogue_takehome}</p>
  <p>Each past event is aligned so that month 0 is July of its precursor year. The similarity score
     captures how closely the shape of the warming matches 2026, not how intense it ultimately got.</p>
{toggle_html}
  <div id="view-oni">
    <p>The top ONI analogues are the events whose raw Niño-3.4 trajectory most closely matched
       2026 over the first 12 months of the event cycle.</p>
    <div class="grid">{top3_html}</div>
{sst_sec_oni}
    <p>Niño-3.4 through the full 24-month window for the top ONI analogues. Through the observed
       period 2026 has tracked between 2023-24 and 1997-98. {fc_sentence}</p>
    <div class="cbox">
      {c_nino34_oni}
      <p class="cnote">Observed monthly data. Month 0 = July of precursor year. Source: NOAA CPC.</p>
    </div>
    <p>Niño 1+2 captures the far eastern Pacific, just off Ecuador and Peru. In Eastern Pacific
       events this region warms first and most sharply. The 2026 jump from -0.3 °C in March to
       +1.81 °C in May is one of the fastest accelerations on record for this index, placing 2026
       well above where 2023-24 was at the same stage.</p>
    <div class="cbox">
      {c_nino12_oni}
      <p class="cnote">Observed monthly data. Source: NOAA CPC.</p>
    </div>
    <p>The SOI provides the atmospheric side of the story. When the SOI turns sharply negative,
       trade winds weaken and warm water spreads east, amplifying the ocean anomaly through
       Bjerknes feedback. The 2026 SOI has tracked close to the 1997-98 pattern, suggesting
       coupling is already well established.</p>
    <div class="cbox">
      {c_soi_oni}
      <p class="cnote">El Niño threshold -1, La Niña +1. Standardised SOI. Source: NOAA CPC.</p>
    </div>
    <h2>Full analogue ranking (ONI)</h2>
    <p class="takehome">The top three analogues are far ahead of the rest, which tells you the 2026 trajectory is distinctive. Events ranked below 0.5 followed a different path from the start.</p>
    <p>All confirmed El Niño events since 1950, ordered by how closely their first 12 months of
       Niño-3.4 evolution match 2026.</p>
    <div class="cbox">
      {c_analogue_oni}
      <p class="cnote">Similarity captures shape, not magnitude. EP/CP type and subsurface
      precursors are separate dimensions not reflected here.</p>
    </div>
  </div>
{roni_view_html}

  <h2>What's coming</h2>
  <p class="takehome">The surface signal is now clear, but what happens next depends on heat stored below the thermocline. Two additional variables would sharpen the peak intensity forecast considerably.</p>

  <div class="alert ag">
    <strong>Kelvin waves (subsurface):</strong> three downwelling Kelvin waves in late 2025 and early
    2026 loaded the thermocline and are the primary subsurface driver of this event. Depth-section
    data from GODAS and TAO-TRITON would make this visible.
  </div>

  <div class="alert ag">
    <strong>Warm Water Volume (WWV):</strong> the volume of water above 20 °C averaged over
    5S-5N, 120E-80W predicts peak intensity months in advance and is available at NOAA PMEL.
  </div>



  <h2>Sources</h2>
  <ul class="srcs">
    <li><a href="https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices" target="_blank">NOAA CPC: Monthly Niño region SST indices</a></li>
    <li><a href="https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt" target="_blank">NOAA CPC: Oceanic Niño Index (ONI)</a></li>
    <li><a href="https://www.cpc.ncep.noaa.gov/data/indices/wksst9120.for" target="_blank">NOAA CPC: Weekly Niño region SST indices</a></li>
    <li><a href="https://www.cpc.ncep.noaa.gov/data/indices/soi" target="_blank">NOAA CPC: Standardised SOI</a></li>
    <li><a href="https://psl.noaa.gov/enso/mei/" target="_blank">NOAA PSL: Multivariate ENSO Index v2 (MEI.v2)</a></li>
    <li><a href="https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso_advisory/ensodisc.shtml" target="_blank">NOAA CPC: ENSO Diagnostic Discussion, June 2026</a></li>
    <li><a href="https://iri.columbia.edu/our-expertise/climate/forecasts/enso/current/" target="_blank">IRI Columbia: ENSO Forecasts, June 2026</a></li>
    <li><a href="https://www.ncei.noaa.gov/products/extended-reconstructed-sst" target="_blank">NOAA NCEI: Extended Reconstructed SST v5 (ERSSTV5)</a></li>
    <li><a href="https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/roni/" target="_blank">NOAA CPC: Relative Oceanic Niño Index (RONI)</a></li>
  </ul>

</div>

<footer>
  Data: NOAA CPC / {latest_date}
</footer>
<script>
function switchView(v) {{
  var oni  = document.getElementById('view-oni');
  var roni = document.getElementById('view-roni');
  var bOni  = document.getElementById('btn-oni');
  var bRoni = document.getElementById('btn-roni');
  oni.style.display  = v === 'oni'  ? '' : 'none';
  roni.style.display = v === 'roni' ? '' : 'none';
  bOni.classList.toggle('active',  v === 'oni');
  bRoni.classList.toggle('active', v === 'roni');
  var target = v === 'oni' ? oni : roni;
  if (typeof Plotly !== 'undefined') {{
    target.querySelectorAll('.plotly-graph-div').forEach(function(el) {{
      Plotly.relayout(el, {{}});
    }});
  }}
}}
</script>
</body>
</html>"""
