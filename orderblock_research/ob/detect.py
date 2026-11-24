"""Order-block detection, following the usual ICT / Smart Money Concepts definition.

Bullish order block (mirror everything for bearish):

1. Find swing highs: a bar whose high is the highest of the ``swing_len`` bars on each side.
2. Break of structure (BOS): a later bar closes above the most recent swing high.
3. The impulse leg that caused the break starts at the lowest low between the swing high and
   the break bar.
4. The order block is the *last down-close candle* at or just before that low
   (``ob_rule="last_opposite"``, the classic ICT wording) or simply the candle that made the
   low (``ob_rule="extreme"``, what most open-source indicators draw).
5. Zone = that candle's full range (``zone="wick"``) or open-to-low (``zone="body"``).
6. Optional displacement filter: the impulse leg must leave a 3-candle fair value gap (FVG).

Look-ahead
----------
A swing high at bar ``i`` is only *known* at bar ``i + swing_len``, because you need the bars
on its right. ``causal=True`` (default) respects that. ``causal=False`` lets the detector use a
swing point as soon as it happens. Several published indicators do this, and it is a source of
look-ahead bias that ``run_analysis.py`` measures.

Every event records ``bos_idx``, the bar at whose close the order block first exists. A
realistic backtest may only place orders from ``bos_idx + 1`` onwards.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

EVENT_COLUMNS = ["dir", "swing_idx", "swing_level", "ob_idx", "bos_idx", "top", "bottom", "fvg"]


@dataclass(frozen=True)
class DetectConfig:
    swing_len: int = 5
    ob_rule: str = "last_opposite"  # "last_opposite" | "extreme"
    zone: str = "wick"  # "wick" | "body"
    require_fvg: bool = False
    causal: bool = True
    max_search: int = 10  # bars to search back from the leg extreme for the opposite candle

    def label(self) -> str:
        return (f"swing{self.swing_len}/{self.ob_rule}/{self.zone}"
                f"/{'fvg' if self.require_fvg else 'nofvg'}")


def atr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
    """Simple-average true range. Value at bar t uses bars <= t only."""
    h, l, c = (df[k].to_numpy(float) for k in ("high", "low", "close"))
    prev_c = np.r_[np.nan, c[:-1]]
    tr = np.fmax(h - l, np.fmax(np.abs(h - prev_c), np.abs(l - prev_c)))
    return pd.Series(tr).rolling(n).mean().to_numpy()


def pivot_points(high: np.ndarray, low: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices of swing highs and swing lows.

    Bar i is a swing high if high[i] is the max of high[i-L : i+L+1] and strictly above every
    bar on its left (so a flat top produces one pivot, not several).
    """
    n = len(high)
    if n < 2 * length + 1:
        return np.array([], int), np.array([], int)
    L = length
    full_max = sliding_window_view(high, 2 * L + 1).max(axis=1)
    left_max = sliding_window_view(high, L).max(axis=1)[: n - 2 * L]
    ch = high[L : n - L]
    ph = np.flatnonzero((ch == full_max) & (ch > left_max)) + L

    full_min = sliding_window_view(low, 2 * L + 1).min(axis=1)
    left_min = sliding_window_view(low, L).min(axis=1)[: n - 2 * L]
    cl = low[L : n - L]
    pl = np.flatnonzero((cl == full_min) & (cl < left_min)) + L
    return ph, pl


def _build_ob(d, o, h, l, c, sw, t, cfg: DetectConfig):
    a, b = sw + 1, t - 1  # bars strictly between the swing point and the break bar
    if b < a:
        return None
    if d == 1:
        seg = l[a : b + 1]
        m = a + (len(seg) - 1 - int(np.argmin(seg[::-1])))  # last occurrence of the lowest low
    else:
        seg = h[a : b + 1]
        m = a + (len(seg) - 1 - int(np.argmax(seg[::-1])))

    k = m
    if cfg.ob_rule == "last_opposite":
        lo = max(a, m - cfg.max_search)
        for j in range(m, lo - 1, -1):
            if (d == 1 and c[j] < o[j]) or (d == -1 and c[j] > o[j]):
                k = j
                break

    if d == 1:
        top = h[k] if cfg.zone == "wick" else max(o[k], c[k])
        bottom = l[k]
        if not c[t] > top:  # price must have left the zone
            return None
    else:
        top = h[k]
        bottom = l[k] if cfg.zone == "wick" else min(o[k], c[k])
        if not c[t] < bottom:
            return None

    fvg = False
    for j in range(k + 2, t + 1):
        if (d == 1 and l[j] > h[j - 2]) or (d == -1 and h[j] < l[j - 2]):
            fvg = True
            break
    if cfg.require_fvg and not fvg:
        return None

    level = h[sw] if d == 1 else l[sw]
    return (d, sw, level, k, t, top, bottom, fvg)


def _scan(d, o, h, l, c, piv, cfg: DetectConfig):
    lag = cfg.swing_len if cfg.causal else 1
    out = []
    p, last, crossed = 0, -1, True
    for t in range(len(c)):
        while p < len(piv) and piv[p] + lag <= t:
            last, crossed = int(piv[p]), False
            p += 1
        if crossed or last < 0:
            continue
        if (d == 1 and c[t] > h[last]) or (d == -1 and c[t] < l[last]):
            crossed = True
            ev = _build_ob(d, o, h, l, c, last, t, cfg)
            if ev is not None:
                out.append(ev)
    return out


def detect_order_blocks(df: pd.DataFrame, cfg: DetectConfig = DetectConfig()) -> pd.DataFrame:
    """Return one row per order block, sorted by the bar at which it becomes known."""
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    ph, pl = pivot_points(h, l, cfg.swing_len)
    rows = _scan(1, o, h, l, c, ph, cfg) + _scan(-1, o, h, l, c, pl, cfg)
    ev = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    ev = ev.astype({"dir": int, "swing_idx": int, "ob_idx": int, "bos_idx": int, "fvg": bool})
    return ev.sort_values(["bos_idx", "dir"]).reset_index(drop=True)
