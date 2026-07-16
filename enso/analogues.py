import numpy as np
import pandas as pd
from enso.data import ENSO_EVENTS, aligned_series


def _pearson(a: list, b: list) -> float:
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 4:
        return 0.0
    arr = np.array(pairs)
    r = np.corrcoef(arr[:, 0], arr[:, 1])[0, 1]
    return float(np.clip(r, -1, 1))


def rank_analogues(series: pd.Series, n_obs: int = 12) -> list[dict]:
    current = aligned_series(series, 2026, 24)[:n_obs]
    rows = []
    for event, meta in ENSO_EVENTS.items():
        if meta["onset_yr"] == 2026:
            continue
        hist = aligned_series(series, meta["onset_yr"], 24)[:n_obs]
        rows.append({
            "event":    event,
            "score":    round(_pearson(current, hist), 3),
            "peak_oni": meta["peak_oni"],
            "type":     meta["type"],
            "color":    meta["color"],
            "onset_yr": meta["onset_yr"],
        })
    rows.sort(key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(rows):
        r["rank"] = i + 1
    return rows
