"""Trade statistics in R units."""

from __future__ import annotations

import numpy as np
import pandas as pd


def summarize(trades: pd.DataFrame, col: str = "r_net", n_boot: int = 2000, seed: int = 0) -> dict:
    """Stats over filled trades. ``trades`` is the output of ``backtest.simulate``."""
    t = trades[trades["filled"]].sort_values("exit_idx")
    r = t[col].to_numpy(float)
    n = len(r)
    out = {"n_events": int(len(trades)), "n": n}
    if n == 0:
        return out | {"win_rate": np.nan, "avg_r": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                      "pf": np.nan, "total_r": 0.0, "max_dd_r": 0.0, "t": np.nan}
    rng = np.random.default_rng(seed)
    boot = r[rng.integers(0, n, size=(n_boot, n))].mean(axis=1) if n > 1 else np.array([r[0]])
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    eq = np.cumsum(r)
    dd = np.max(np.maximum.accumulate(np.r_[0.0, eq]) - np.r_[0.0, eq])
    sd = r.std(ddof=1) if n > 1 else np.nan
    return out | {
        "win_rate": float((r > 0).mean()),
        "avg_r": float(r.mean()),
        "ci_lo": float(np.percentile(boot, 2.5)),
        "ci_hi": float(np.percentile(boot, 97.5)),
        "pf": float(gains / losses) if losses > 0 else np.inf,
        "total_r": float(r.sum()),
        "max_dd_r": float(dd),
        "t": float(r.mean() / (sd / np.sqrt(n))) if n > 1 and sd > 0 else np.nan,
    }
