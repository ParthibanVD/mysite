"""Real historical OHLC data that ships inside PyPI packages, so no network access is needed.

Sources
-------
* S&P 500 and NASDAQ Composite, daily, 1999-2018  -> ``arch.data``
* GOOG, daily, 2004-2013                          -> ``backtesting.test``
* EURUSD, hourly, Jan 2017 - Feb 2018              -> ``gym_anytrading`` (2017, ask prices)
                                                     + ``backtesting.test`` (tail, Dec 2017 - Feb 2018)
  The two EURUSD feeds overlap for 4,356 hours and agree to within 0.2 pips on average.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import pandas as pd

COLS = ["open", "high", "low", "close"]


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns=str.lower)[COLS].astype(float)
    out = out[~out.index.duplicated()].sort_index()
    return out


def sp500_daily() -> pd.DataFrame:
    from arch.data import sp500

    return _norm(sp500.load())


def nasdaq_daily() -> pd.DataFrame:
    from arch.data import nasdaq

    return _norm(nasdaq.load())


def goog_daily() -> pd.DataFrame:
    from backtesting.test import GOOG

    return _norm(GOOG)


def eurusd_h1() -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from backtesting.test import EURUSD
        from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK

    g = FOREX_EURUSD_1H_ASK.copy()
    g.index = pd.to_datetime(g.index, format="%d.%m.%Y %H:%M:%S.%f")
    g = _norm(g)
    tail = _norm(EURUSD)
    tail = tail[tail.index > g.index[-1]]
    return pd.concat([g, tail])


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    return df.resample(rule).agg(agg).dropna()


def eurusd_h4() -> pd.DataFrame:
    return resample(eurusd_h1(), "4h")


@dataclass(frozen=True)
class Dataset:
    key: str
    label: str
    loader: Callable[[], pd.DataFrame]
    cost_bps: float = 0.0  # round-trip cost as a fraction of price, in basis points
    cost_abs: float = 0.0  # round-trip cost in price units (FX pips)
    note: str = ""

    def cost(self, price: float) -> float:
        """Round-trip trading cost in price units."""
        return self.cost_abs + price * self.cost_bps / 1e4


DATASETS: dict[str, Dataset] = {
    d.key: d
    for d in [
        Dataset("spx_d", "S&P 500 daily 1999-2018", sp500_daily, cost_bps=2,
                note="2 bps round trip (index future)"),
        Dataset("ndx_d", "NASDAQ Comp. daily 1999-2018", nasdaq_daily, cost_bps=2,
                note="2 bps round trip (index future)"),
        Dataset("goog_d", "GOOG daily 2004-2013", goog_daily, cost_bps=5,
                note="5 bps round trip (stock)"),
        Dataset("eur_h4", "EURUSD 4-hour 2017-2018", eurusd_h4, cost_abs=0.0001,
                note="1.0 pip round trip"),
        Dataset("eur_h1", "EURUSD 1-hour 2017-2018", eurusd_h1, cost_abs=0.0001,
                note="1.0 pip round trip"),
    ]
}


def load(key: str) -> pd.DataFrame:
    return DATASETS[key].loader()
