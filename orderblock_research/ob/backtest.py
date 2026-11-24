"""Bar-by-bar simulation of the standard order-block trade.

Bullish OB (mirror for bearish):
* Buy limit at the zone's proximal edge (top), or at its midpoint ("mean threshold").
* Stop just below the zone's distal edge (bottom).
* Target = entry + rr * risk.
* The order is live from the bar *after* the order block becomes known, for ``max_age`` bars.
  Only the first touch counts.

Fill and exit assumptions are conservative and stated explicitly:
* A limit buy fills at min(open, limit), so a gap down through the limit fills at the open.
* If a bar touches both the stop and the target, the stop is assumed to have hit first.
* On the fill bar itself only the stop is checked (we cannot know whether the high came
  before or after the fill).
* Gaps through the stop or target exit at the open of that bar.
* Results are in R (multiples of the planned risk), net of a round-trip cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TradeConfig:
    entry: str = "proximal"  # "proximal" | "mid"
    rr: float = 2.0
    max_age: int = 100
    stop_buffer_atr: float = 0.0
    min_risk_atr: float = 0.1  # skip zones so thin that risk is < 0.1 ATR
    available: str = "bos"  # "bos" (honest) | "ob_candle" (look-ahead, for bias tests)

    def label(self) -> str:
        return f"{self.entry}/rr{self.rr:g}"


TRADE_COLUMNS = ["filled", "fill_idx", "fill_px", "exit_idx", "exit_px", "reason",
                 "entry", "stop", "target", "risk", "r_gross", "r_net"]


def _sim_one(o, h, l, c, d, start, end, entry, stop, tp):
    n = len(c)
    if start >= n:
        return None
    touch = (l[start:end] <= entry) if d == 1 else (h[start:end] >= entry)
    if not touch.any():
        return None
    k = start + int(np.argmax(touch))

    if d == 1:
        fill = min(o[k], entry)
        if fill <= stop:
            return k, fill, k, fill, "gap"
        if l[k] <= stop:
            return k, fill, k, stop, "sl"
    else:
        fill = max(o[k], entry)
        if fill >= stop:
            return k, fill, k, fill, "gap"
        if h[k] >= stop:
            return k, fill, k, stop, "sl"

    j0 = k + 1
    if j0 >= n:
        return k, fill, n - 1, c[n - 1], "eod"
    if d == 1:
        sl_hit, tp_hit = l[j0:] <= stop, h[j0:] >= tp
    else:
        sl_hit, tp_hit = h[j0:] >= stop, l[j0:] <= tp
    hit = sl_hit | tp_hit
    if not hit.any():
        return k, fill, n - 1, c[n - 1], "eod"
    j = j0 + int(np.argmax(hit))
    oj = o[j]
    if d == 1:
        if oj <= stop:
            return k, fill, j, oj, "sl"
        if oj >= tp:
            return k, fill, j, oj, "tp"
    else:
        if oj >= stop:
            return k, fill, j, oj, "sl"
        if oj <= tp:
            return k, fill, j, oj, "tp"
    if sl_hit[j - j0]:
        return k, fill, j, stop, "sl"
    return k, fill, j, tp, "tp"


def simulate(
    df: pd.DataFrame,
    events: pd.DataFrame,
    tcfg: TradeConfig = TradeConfig(),
    cost: Callable[[float], float] = lambda px: 0.0,
    atr_arr: np.ndarray | None = None,
) -> pd.DataFrame:
    """Simulate one trade per order block. Returns events joined with trade columns."""
    from .detect import atr as _atr

    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    a = _atr(df) if atr_arr is None else atr_arr
    rows = []
    for ev in events.itertuples(index=False):
        d, top, bottom = ev.dir, ev.top, ev.bottom
        av = a[ev.bos_idx]
        nan_row = (False, -1, np.nan, -1, np.nan, "skip", np.nan, np.nan, np.nan, np.nan,
                   np.nan, np.nan)
        if not np.isfinite(av) or top <= bottom:
            rows.append(nan_row)
            continue
        buf = tcfg.stop_buffer_atr * av
        if d == 1:
            entry = top if tcfg.entry == "proximal" else (top + bottom) / 2
            stop = bottom - buf
        else:
            entry = bottom if tcfg.entry == "proximal" else (top + bottom) / 2
            stop = top + buf
        risk = abs(entry - stop)
        if risk < tcfg.min_risk_atr * av:
            rows.append(nan_row)
            continue
        tp = entry + d * tcfg.rr * risk
        start = ev.bos_idx + 1 if tcfg.available == "bos" else ev.ob_idx + 1
        res = _sim_one(o, h, l, c, d, start, start + tcfg.max_age, entry, stop, tp)
        if res is None:
            rows.append((False, -1, np.nan, -1, np.nan, "nofill", entry, stop, tp, risk,
                         np.nan, np.nan))
            continue
        k, fill, j, exit_px, reason = res
        r_gross = d * (exit_px - fill) / risk
        r_net = r_gross - cost(entry) / risk
        rows.append((True, k, fill, j, exit_px, reason, entry, stop, tp, risk, r_gross, r_net))
    tr = pd.DataFrame(rows, columns=TRADE_COLUMNS, index=events.index)
    return pd.concat([events, tr], axis=1)
