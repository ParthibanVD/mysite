# Order-block strategy research

A recreation of the ICT / Smart Money Concepts "order block" strategy, plus a backtest that checks
whether it works. **Findings are in [REPORT.md](REPORT.md).** Short version: no edge after costs,
no better than random zones, and the high win rates that order-block tools advertise can be
reproduced only through look-ahead (repainting) and survivorship bias.

## Layout

| Path | What it is |
|---|---|
| `ob/detect.py` | Swing points, break of structure, order-block and fair-value-gap detection (no look-ahead by default) |
| `ob/backtest.py` | Limit-entry / stop / target simulation with conservative fill rules, results in R |
| `ob/stats.py` | Win rate, average R, bootstrap confidence range, profit factor, drawdown |
| `ob/data.py` | Real price data bundled in PyPI packages: S&P 500, NASDAQ, GOOG, EURUSD |
| `ob/charts.py` | The charts in `results/` |
| `run_analysis.py` | Everything: headline backtest, random-zone tests, 720-variant sweep, split-half test, bias tests |
| `pine/order_blocks_strategy.pine` | The same strategy for TradingView's strategy tester |
| `tests/` | Unit tests on hand-built candle sequences |
| `results/` | Output: CSVs, `summary.json`, charts |

## Run

```bash
pip install -r requirements.txt
python -m pytest tests
python run_analysis.py           # add --quick for fewer random-zone runs
```

## Use the detector on your own data

```python
import pandas as pd
from ob import DetectConfig, TradeConfig, detect_order_blocks, simulate, summarize

df = pd.read_csv("prices.csv", index_col=0, parse_dates=True)  # columns: open, high, low, close
events = detect_order_blocks(df, DetectConfig(swing_len=5, require_fvg=True))
trades = simulate(df, events, TradeConfig(entry="proximal", rr=2), cost=lambda px: 0.0001)
print(summarize(trades))
```

## TradingView

Open the Pine Editor, paste `pine/order_blocks_strategy.pine`, then add it to a chart. Set
commission and slippage for your market in the script's properties. Zones that fail stay on the
chart on purpose. The script was written without access to TradingView's compiler, so report any
compile error back.
