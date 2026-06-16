"""
Ranks historical El Niño events by similarity to the current 2026 event.
Similarity = Pearson correlation of aligned Niño-3.4 time series.
"""
import numpy as np
import pandas as pd
from enso.data import ENSO_EVENTS, aligned_series


def _pearson(a: list, b: list) -> float:
    """Pearson r ignoring None/NaN pairs. Returns 0 if insufficient data."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 4:
        return 0.0
    arr = np.array(pairs)
    r = np.corrcoef(arr[:, 0], arr[:, 1])[0, 1]
    return float(np.clip(r, -1, 1))


def rank_analogues(nino34: pd.Series, n_obs: int = 12) -> list[dict]:
    """
    Compare all historical events to 2026 over the first n_obs aligned months.

    Parameters
    ----------
    nino34 : full monthly Niño-3.4 anomaly series
    n_obs  : number of observed months available for 2026 (rest is forecast)

    Returns list of dicts sorted by similarity (descending), each with:
      event, rank, score, peak_oni, type, color
    """
    current = aligned_series(nino34, 2026, 24)[:n_obs]

    rows = []
    for event, meta in ENSO_EVENTS.items():
        if meta["onset_yr"] == 2026:
            continue
        hist = aligned_series(nino34, meta["onset_yr"], 24)[:n_obs]
        score = _pearson(current, hist)
        rows.append({
            "event":    event,
            "score":    round(score, 3),
            "peak_oni": meta["peak_oni"],
            "type":     meta["type"],
            "color":    meta["color"],
            "onset_yr": meta["onset_yr"],
        })

    rows.sort(key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(rows):
        r["rank"] = i + 1
    return rows
