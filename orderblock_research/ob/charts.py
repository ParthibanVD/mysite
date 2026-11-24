"""Static charts for REPORT.md. Reads the files written by run_analysis.py."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from .data import DATASETS  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e8e7e3"
NEUTRAL = "#c9c8c2"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # reference order
BLUE, ORANGE = SERIES[0], SERIES[1]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False, "legend.labelcolor": INK2,
})


def _title(fig, title, sub):
    fig.text(0.012, 0.975, title, ha="left", va="top", fontsize=13, weight="bold", color=INK)
    fig.text(0.012, 0.915, sub, ha="left", va="top", fontsize=9.5, color=INK2)


def _save(fig, path):
    fig.savefig(path, dpi=200)
    plt.close(fig)


def placebo_chart(out: Path):
    draws = json.loads((out / "placebo_draws.json").read_text())
    keys = list(DATASETS)[::-1]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    fig.subplots_adjust(left=0.27, right=0.80, top=0.78, bottom=0.14)
    for y, k in enumerate(keys):
        p = np.array(draws[k]["random_time"]["avg_r"])
        lo, med, hi = np.percentile(p, [5, 50, 95])
        ax.plot([lo, hi], [y, y], color=NEUTRAL, lw=9, solid_capstyle="round", zorder=1)
        ax.plot([med, med], [y - 0.18, y + 0.18], color=MUTED, lw=1.5, zorder=2)
        ax.scatter(draws[k]["real_avg_r"], y, s=70, color=BLUE, edgecolor=SURFACE, lw=2, zorder=3)
        pt, pl = draws[k]["random_time"]["p_value_ob_better"], draws[k]["random_level"]["p_value_ob_better"]
        ax.text(1.02, y, f"p = {pt:.2f}  /  {pl:.2f}", transform=ax.get_yaxis_transform(),
                va="center", fontsize=9, color=INK2)
    ax.text(1.02, len(keys) - 0.4, "random time / level", transform=ax.get_yaxis_transform(),
            va="center", fontsize=8.5, color=MUTED)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(keys)), [DATASETS[k].label for k in keys])
    ax.set_xlabel("Average result per trade, R after costs (2R target)")
    ax.grid(axis="y", visible=False)
    ax.scatter([], [], s=60, color=BLUE, label="Order-block trades")
    ax.plot([], [], color=NEUTRAL, lw=8, label="Random zones, same shape: middle 90% of 200 runs")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, -0.38), ncol=2, fontsize=9)
    fig.subplots_adjust(bottom=0.24)
    _title(fig, "Order blocks do no better than random zones of the same shape",
           "Every blue dot sits inside the range that random limit-entry zones produce. "
           "p = share of random runs that did at least as well.")
    _save(fig, out / "placebo.png")


def bias_chart(out: Path):
    b = pd.read_csv(out / "bias.csv")
    rows = [
        ("0 honest", "Honest backtest"),
        ("1 + centred swings", "+ centred (future) swing points"),
        ("2 + orders from OB candle", "+ orders live from the OB candle (repaint)"),
        ("3 + failed zones deleted", "+ failed zones deleted (survivorship)"),
        ("smartmoneyconcepts lib, naive", "smartmoneyconcepts library, as-is"),
    ]
    vals = []
    for step, label in rows:
        g = b[b.step == step]
        n = g.n.sum()
        vals.append((label, (g.avg_r * g.n).sum() / n, (g.win_rate * g.n).sum() / n, int(n)))
    vals = vals[::-1]
    fig, ax = plt.subplots(figsize=(9, 4.0))
    fig.subplots_adjust(left=0.36, right=0.97, top=0.78, bottom=0.14)
    for y, (label, avg, win, n) in enumerate(vals):
        c = BLUE if label.startswith("Honest") else ORANGE
        ax.barh(y, avg, height=0.55, color=c, zorder=2)
        x = max(avg, 0) + 0.05
        ax.text(x, y, f"{avg:+.2f}R  ·  {win:.0%} wins  ·  {n} trades", va="center", fontsize=9, color=INK)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(vals)), [v[0] for v in vals])
    ax.set_xlim(-0.3, 3.8)
    ax.set_xlabel("Average result per trade, R after costs (all five datasets pooled)")
    ax.grid(axis="y", visible=False)
    _title(fig, "Common backtest shortcuts turn a losing rule into a 'winning' one",
           "Same detector and trade rules; each row adds one bias on top of the row above "
           "(the last row is the open-source library used as-is).")
    _save(fig, out / "bias.png")


def sweep_chart(out: Path):
    s = pd.read_csv(out / "sweep.csv").sort_values("full_avg_r").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.78, bottom=0.2)
    neg = s.full_ci_hi < 0
    x = np.arange(len(s))
    for mask, c, lab in ((~neg, NEUTRAL, "95% interval includes zero"),
                         (neg, ORANGE, "95% interval entirely below zero")):
        ax.vlines(x[mask], s.full_ci_lo[mask], s.full_ci_hi[mask], color=c, lw=0.7, alpha=0.9, zorder=1)
        ax.scatter(x[mask], s.full_avg_r[mask], s=4, color=INK2 if c == NEUTRAL else c, zorder=2,
                   label=lab)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xlim(-5, len(s) + 5)
    ax.set_xlabel("720 rule variants (24 detection variants x 6 entry/target variants x 5 datasets), sorted")
    ax.set_ylabel("Average R per trade, after costs")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", markerscale=3, fontsize=9)
    up = int((s.full_ci_lo > 0).sum())
    _title(fig, f"{up} of {len(s)} rule variants show a reliable edge",
           f"Median variant: {s.full_avg_r.median():+.2f}R per trade. {int(neg.sum())} variants "
           "lose reliably; none win reliably.")
    _save(fig, out / "sweep.png")


WF_OFFSETS = {"goog_d": (-8, 8, "right"), "spx_d": (6, 8, "left"),
              "eur_h4": (-8, 0, "right")}


def walkforward_chart(out: Path):
    s = pd.read_csv(out / "sweep.csv")
    s = s[(s.is_n >= 20) & (s.oos_n >= 10)]
    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    fig.subplots_adjust(left=0.13, right=0.96, top=0.83, bottom=0.11)
    ax.scatter(s.is_avg_r, s.oos_avg_r, s=9, color=NEUTRAL, edgecolor="none", zorder=1,
               label="Every rule variant")
    wf = pd.read_csv(out / "walk_forward.csv")
    for r in wf.itertuples():
        ax.scatter(r.is_avg_r, r.oos_avg_r, s=70, color=BLUE, edgecolor=SURFACE, lw=2, zorder=3)
        dx, dy, ha = WF_OFFSETS.get(r.dataset, (7, -3, "left"))
        ax.annotate(DATASETS[r.dataset].label.rsplit(" ", 1)[0], (r.is_avg_r, r.oos_avg_r),
                    xytext=(dx, dy), textcoords="offset points", fontsize=8.5, color=INK, ha=ha)
    ax.scatter([], [], s=60, color=BLUE, label="Best variant picked on the first half")
    ax.axhline(0, color=INK2, lw=1)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlabel("First half of the data: average R per trade")
    ax.set_ylabel("Second half: average R per trade")
    ax.legend(loc="lower left", fontsize=9)
    _title(fig, "The best-looking variant fails on unseen data",
           "Picking the top variant on the first half, then trading it on the second half,\n"
           "lost money on all five datasets.")
    _save(fig, out / "walk_forward.png")


def equity_chart(out: Path):
    fig, ax = plt.subplots(figsize=(9, 5.0))
    fig.subplots_adjust(left=0.08, right=0.94, top=0.82, bottom=0.25)
    for c, (k, ds) in zip(SERIES, DATASETS.items()):
        t = pd.read_csv(out / f"trades_{k}.csv")
        t = t[t.filled].sort_values("exit_idx")
        eq = np.r_[0, np.cumsum(t.r_net.to_numpy())]
        ax.plot(np.arange(len(eq)), eq, color=c, lw=2, label=ds.label)
        ax.text(len(eq) + 3, eq[-1], f"{eq[-1]:+.0f}R", va="center", fontsize=9, color=INK)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xlabel("Trade number")
    ax.set_ylabel("Cumulative R after costs")
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.14), ncol=3, fontsize=9)
    ax.grid(axis="x", visible=False)
    _title(fig, "Order-block trading, honest test: flat to losing on every dataset",
           "Limit at the zone edge, stop beyond the far edge, 2R target. Swing length 5, "
           "last opposite candle, full candle range.")
    _save(fig, out / "equity.png")


def example_chart(out: Path, key: str = "eur_h1", width: int = 140):
    df = DATASETS[key].loader()
    t = pd.read_csv(out / f"trades_{key}.csv")
    f = t[t.filled]
    # pick the window holding the most completed trades
    starts = np.arange(0, len(df) - width, 10)
    counts = [((f.ob_idx >= s) & (f.exit_idx < s + width)).sum() for s in starts]
    s0 = int(starts[int(np.argmax(counts))])
    s1 = s0 + width
    w = df.iloc[s0:s1]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    fig.subplots_adjust(left=0.07, right=0.96, top=0.78, bottom=0.09)
    x = np.arange(s0, s1)
    up = w.close >= w.open
    ax.vlines(x, w.low, w.high, color=INK2, lw=0.7, zorder=2)
    for m, fc in ((up, SURFACE), (~up, INK2)):
        ax.bar(x[m], (w.close - w.open)[m], bottom=w.open[m], width=0.6, color=fc,
               edgecolor=INK2, lw=0.7, zorder=3)
    vis = t[(t.bos_idx < s1) & (t.ob_idx >= s0 - 20)]
    for e in vis.itertuples():
        c = BLUE if e.dir == 1 else ORANGE
        end = e.exit_idx if e.filled else min(e.bos_idx + 100, s1)
        x0 = max(e.ob_idx, s0) - 0.4
        ax.add_patch(Rectangle((x0, e.bottom), min(end, s1) - x0, e.top - e.bottom,
                               facecolor=c, alpha=0.16, edgecolor=c, lw=0.8, zorder=1))
        if e.filled and s0 <= e.fill_idx < s1:
            ax.scatter(e.fill_idx, e.fill_px, marker="^" if e.dir == 1 else "v", s=55, color=c,
                       edgecolor=SURFACE, lw=1, zorder=4)
            if e.exit_idx < s1:
                ax.annotate(f"{e.r_net:+.1f}R", (e.exit_idx, e.exit_px), xytext=(4, 0),
                            textcoords="offset points", fontsize=8.5, color=INK, va="center")
    ax.set_xlim(s0 - 1, s1)
    ticks = np.linspace(s0 + 2, s1 - 8, 6).astype(int)
    ax.set_xticks(ticks, [df.index[i].strftime("%d %b %H:%M") for i in ticks])
    ax.set_ylabel("EURUSD")
    ax.grid(axis="x", visible=False)
    ax.add_patch(Rectangle((0, 0), 0, 0, facecolor=BLUE, alpha=0.3, label="Bullish order block"))
    ax.add_patch(Rectangle((0, 0), 0, 0, facecolor=ORANGE, alpha=0.3, label="Bearish order block"))
    ax.scatter([], [], marker="^", color=INK2, s=40, label="Limit order filled (label = result)")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.01, 1.0), fontsize=9, ncol=3)
    _title(fig, "The recreated detector on EURUSD 1-hour",
           "Zones start at the order-block candle and end where the trade closed "
           "(or after 100 bars unfilled).")
    _save(fig, out / "example.png")


def make_all(out: Path):
    for fn in (placebo_chart, bias_chart, sweep_chart, walkforward_chart, equity_chart, example_chart):
        fn(out)
    print("charts written to", out)


if __name__ == "__main__":
    make_all(Path(__file__).resolve().parents[1] / "results")
