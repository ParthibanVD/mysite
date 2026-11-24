import numpy as np
import pandas as pd
import pytest

from ob import DetectConfig, TradeConfig, detect_order_blocks, simulate
from ob.backtest import _sim_one
from ob.detect import pivot_points

# A textbook bullish order block, swing_len = 2:
#   bar 2 is a swing high (12.0); bars 3-4 pull back; bar 4 is the last down candle at the low;
#   bars 5-6 displace up and bar 6 closes above 12.0 (break of structure) leaving an FVG
#   (low[6] 11.3 > high[4] 11.2); bar 8 retests the zone; bar 9 runs to the 2R target.
BARS = [
    # open  high  low   close
    (10.0, 10.5, 9.5, 10.0),
    (10.0, 11.0, 9.8, 10.8),
    (10.8, 12.0, 10.5, 11.5),  # 2 swing high
    (11.5, 11.6, 10.8, 11.0),
    (11.0, 11.2, 10.0, 10.2),  # 4 last down candle = order block
    (10.2, 11.5, 10.1, 11.4),
    (11.4, 12.5, 11.3, 12.4),  # 6 break of structure
    (12.4, 12.6, 11.9, 12.0),
    (12.0, 12.1, 11.0, 11.1),  # 8 retest -> fill at 11.2
    (11.1, 13.7, 11.05, 13.5),  # 9 target 13.6 hit
]


def frame(rows):
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"],
                        index=pd.date_range("2020-01-01", periods=len(rows), freq="D"))


def test_pivots():
    df = frame(BARS)
    ph, pl = pivot_points(df.high.to_numpy(), df.low.to_numpy(), 2)
    assert 2 in ph
    assert 4 in pl


def test_detects_textbook_bullish_ob():
    ev = detect_order_blocks(frame(BARS), DetectConfig(swing_len=2))
    assert len(ev) == 1
    e = ev.iloc[0]
    assert (e.dir, e.swing_idx, e.ob_idx, e.bos_idx) == (1, 2, 4, 6)
    assert (e.top, e.bottom) == (11.2, 10.0)
    assert bool(e.fvg)


def test_body_zone_uses_open_of_down_candle():
    ev = detect_order_blocks(frame(BARS), DetectConfig(swing_len=2, zone="body"))
    assert (ev.iloc[0].top, ev.iloc[0].bottom) == (11.0, 10.0)


def test_bearish_is_mirror_image():
    mirrored = [(-o, -l, -h, -c) for o, h, l, c in BARS]
    ev = detect_order_blocks(frame(mirrored), DetectConfig(swing_len=2))
    bear = ev[ev.dir == -1]
    assert len(bear) == 1
    e = bear.iloc[0]
    assert (e.ob_idx, e.bos_idx, e.top, e.bottom) == (4, 6, -10.0, -11.2)


def test_textbook_trade_wins_2r():
    df = frame(BARS)
    ev = detect_order_blocks(df, DetectConfig(swing_len=2))
    atr = np.full(len(df), 1.0)  # fixed ATR so the tiny sample has one
    tr = simulate(df, ev, TradeConfig(rr=2.0), atr_arr=atr).iloc[0]
    assert tr.filled and tr.fill_idx == 8 and tr.fill_px == 11.2
    assert tr.reason == "tp" and tr.exit_idx == 9
    assert tr.r_gross == pytest.approx(2.0)


def test_cost_is_charged_in_r():
    df = frame(BARS)
    ev = detect_order_blocks(df, DetectConfig(swing_len=2))
    tr = simulate(df, ev, TradeConfig(rr=2.0), cost=lambda px: 0.12,
                  atr_arr=np.full(len(df), 1.0)).iloc[0]
    assert tr.r_net == pytest.approx(2.0 - 0.12 / 1.2)


def test_orders_start_after_break_not_after_ob_candle():
    # With available="ob_candle" the order is live during the impulse leg itself (bar 5 trades
    # through 11.2), which is look-ahead: the OB did not exist until bar 6 closed.
    df = frame(BARS)
    ev = detect_order_blocks(df, DetectConfig(swing_len=2))
    atr = np.full(len(df), 1.0)
    honest = simulate(df, ev, TradeConfig(), atr_arr=atr).iloc[0]
    leaky = simulate(df, ev, TradeConfig(available="ob_candle"), atr_arr=atr).iloc[0]
    assert honest.fill_idx == 8
    assert leaky.fill_idx == 5


A = np.array


def test_same_bar_stop_and_target_counts_as_loss():
    o, h, l, c = A([10.0, 10.0, 10.0]), A([10.0, 10.0, 13.0]), A([10.0, 9.4, 8.0]), A([10.0, 9.8, 10.0])
    res = _sim_one(o, h, l, c, 1, 1, 3, entry=9.5, stop=9.0, tp=10.5)
    assert res[-1] == "sl" and res[3] == 9.0


def test_gap_through_limit_fills_at_open():
    o, h, l, c = A([10.0, 9.3, 9.6]), A([10.0, 9.4, 11.0]), A([10.0, 9.2, 9.5]), A([10.0, 9.3, 10.9])
    k, fill, j, exit_px, reason = _sim_one(o, h, l, c, 1, 1, 3, entry=9.5, stop=9.0, tp=10.5)
    assert fill == 9.3 and reason == "tp" and exit_px == 10.5


def test_gap_through_stop_exits_at_open():
    o, h, l, c = A([10.0, 9.6, 8.5]), A([10.0, 9.7, 8.6]), A([10.0, 9.4, 8.4]), A([10.0, 9.5, 8.5])
    k, fill, j, exit_px, reason = _sim_one(o, h, l, c, 1, 1, 3, entry=9.5, stop=9.0, tp=10.5)
    assert reason == "sl" and exit_px == 8.5  # worse than -1R


def test_unfilled_order_expires():
    o, h, l, c = A([10.0] * 5), A([10.2] * 5), A([9.8] * 5), A([10.0] * 5)
    assert _sim_one(o, h, l, c, 1, 1, 5, entry=9.5, stop=9.0, tp=10.5) is None
