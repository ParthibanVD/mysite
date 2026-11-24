"""Run the full order-block study and write results/ (CSV, JSON, charts).

    python run_analysis.py            # full run (~2 minutes)
    python run_analysis.py --quick    # fewer placebo iterations
"""

from __future__ import annotations

import argparse
import itertools
import json
import warnings
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from ob import DetectConfig, TradeConfig, atr, detect_order_blocks, simulate, summarize
from ob.data import DATASETS

warnings.filterwarnings("ignore")
OUT = Path(__file__).parent / "results"

HEADLINE_D = DetectConfig()  # swing 5, last down/up candle, full wick zone, no FVG filter
HEADLINE_T = TradeConfig()  # limit at proximal edge, stop at distal edge, 2R target

SWEEP_D = [
    DetectConfig(swing_len=s, ob_rule=r, zone=z, require_fvg=f)
    for s, r, z, f in itertools.product([3, 5, 10], ["last_opposite", "extreme"],
                                        ["wick", "body"], [False, True])
]
SWEEP_T = [TradeConfig(entry=e, rr=rr) for e, rr in itertools.product(["proximal", "mid"], [1, 2, 3])]


def run(df, dcfg, tcfg, cost, a=None, events=None):
    ev = detect_order_blocks(df, dcfg) if events is None else events
    return simulate(df, ev, tcfg, cost=cost, atr_arr=atr(df) if a is None else a)


# --------------------------------------------------------------------------- placebo tests
def placebo_events(df, events, a, mode, rng):
    """Random zones with the same geometry as the real order blocks (in ATR units).

    random_time : same direction, distance and width, but at a random bar.
    random_level: same break bar and width, but the zone is placed at a random depth
                  (uniform between 0 and 2x the real distance from the close).
    """
    c = df["close"].to_numpy(float)
    bos = events["bos_idx"].to_numpy()
    d = events["dir"].to_numpy()
    av = a[bos]
    prox = np.where(d == 1, events["top"], events["bottom"])
    dist = d * (c[bos] - prox) / av
    width = (events["top"] - events["bottom"]).to_numpy() / av
    if mode == "random_time":
        valid = np.flatnonzero(np.isfinite(a))
        valid = valid[valid < len(c) - 2]
        t2 = rng.choice(valid, size=len(events))
        dist2 = dist
    else:
        t2 = bos
        dist2 = dist * rng.uniform(0, 2, size=len(events))
    a2 = a[t2]
    prox2 = c[t2] - d * dist2 * a2
    top = np.where(d == 1, prox2, prox2 + width * a2)
    bottom = np.where(d == 1, prox2 - width * a2, prox2)
    return pd.DataFrame({"dir": d, "swing_idx": t2, "swing_level": np.nan, "ob_idx": t2,
                         "bos_idx": t2, "top": top, "bottom": bottom, "fvg": False})


def placebo_test(df, events, tcfg, cost, a, mode, n_iter, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_iter):
        pe = placebo_events(df, events, a, mode, rng)
        t = simulate(df, pe, tcfg, cost=cost, atr_arr=a)
        f = t[t.filled]
        out.append((f.r_net.mean(), (f.r_net > 0).mean(), len(f)))
    return np.array(out)


# --------------------------------------------------------------------------- bias tests
def deleted_after_failure(df, trades):
    """True where a zone was later broken and then traded back through.

    This mirrors indicators (e.g. the `smartmoneyconcepts` library) that erase such zones,
    so a chart, or a backtest on the indicator's output, only shows survivors.
    """
    h, l = df["high"].to_numpy(float), df["low"].to_numpy(float)
    gone = np.zeros(len(trades), bool)
    for i, e in enumerate(trades.itertuples(index=False)):
        s = e.bos_idx + 1
        broken = (l[s:] < e.bottom) if e.dir == 1 else (h[s:] > e.top)
        if not broken.any():
            continue
        b = s + int(np.argmax(broken))
        back = (h[b + 1 :] > e.top) if e.dir == 1 else (l[b + 1 :] < e.bottom)
        gone[i] = back.any()
    return gone


def bias_ladder(df, cost, a):
    steps = {}
    honest = run(df, HEADLINE_D, HEADLINE_T, cost, a)
    steps["0 honest"] = honest
    nc = run(df, replace(HEADLINE_D, causal=False), HEADLINE_T, cost, a)
    steps["1 + centred swings"] = nc
    rep = run(df, replace(HEADLINE_D, causal=False), replace(HEADLINE_T, available="ob_candle"), cost, a)
    steps["2 + orders from OB candle"] = rep
    steps["3 + failed zones deleted"] = rep[~deleted_after_failure(df, rep)]
    # each bias on its own, for reference
    solo = {
        "centred swings only": nc,
        "orders from OB candle only": run(df, HEADLINE_D, replace(HEADLINE_T, available="ob_candle"), cost, a),
        "failed zones deleted only": honest[~deleted_after_failure(df, honest)],
    }
    return steps, solo


def smc_library_backtest(df, cost, a):
    """Backtest the open-source `smartmoneyconcepts` OB output the way a user typically would."""
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        from smartmoneyconcepts import smc
    ohlc = df.copy()
    ohlc["volume"] = 1.0
    sw = smc.swing_highs_lows(ohlc, swing_length=5)
    ob = smc.ob(ohlc, sw)
    rows = ob[ob["OB"].fillna(0) != 0]
    idx = np.flatnonzero(ob["OB"].fillna(0).to_numpy() != 0)
    ev = pd.DataFrame({"dir": rows["OB"].astype(int).to_numpy(), "swing_idx": idx,
                       "swing_level": np.nan, "ob_idx": idx, "bos_idx": idx,
                       "top": rows["Top"].astype(float).to_numpy(),
                       "bottom": rows["Bottom"].astype(float).to_numpy(), "fvg": False})
    return simulate(df, ev, replace(HEADLINE_T, available="ob_candle"), cost=cost, atr_arr=a)


# --------------------------------------------------------------------------- main
def main(quick=False):
    OUT.mkdir(exist_ok=True)
    n_placebo = 40 if quick else 200
    summary = {"headline_config": {**asdict(HEADLINE_D), **asdict(HEADLINE_T)}, "datasets": {}}
    headline_rows, sweep_rows, wf_rows, bias_rows, placebo_store = [], [], [], [], {}
    pooled = []

    for key, ds in DATASETS.items():
        print(f"== {ds.label}")
        df = ds.loader()
        a = atr(df)
        summary["datasets"][key] = {"label": ds.label, "bars": len(df), "start": str(df.index[0]),
                                    "end": str(df.index[-1]), "cost": ds.note}

        # 1. headline config at 1R / 2R / 3R, net and gross, plus long/short split
        ev = detect_order_blocks(df, HEADLINE_D)
        for rr in (1, 2, 3):
            t = run(df, HEADLINE_D, replace(HEADLINE_T, rr=rr), ds.cost, a, events=ev)
            for side, sub in (("both", t), ("long", t[t.dir == 1]), ("short", t[t.dir == -1])):
                s = summarize(sub)
                headline_rows.append({"dataset": key, "rr": rr, "side": side, **s,
                                      "avg_r_gross": summarize(sub, "r_gross")["avg_r"],
                                      "breakeven_win": 1 / (1 + rr)})
            if rr == 2:
                t.to_csv(OUT / f"trades_{key}.csv")
                pooled.append(t[t.filled].assign(dataset=key))
                head = t

        # 2. placebo tests on the headline config
        real = summarize(head)
        placebo_store[key] = {"real_avg_r": real["avg_r"], "real_win": real["win_rate"]}
        for mode in ("random_time", "random_level"):
            p = placebo_test(df, ev, HEADLINE_T, ds.cost, a, mode, n_placebo)
            pval = (1 + np.sum(p[:, 0] >= real["avg_r"])) / (1 + len(p))
            placebo_store[key][mode] = {
                "avg_r": p[:, 0].tolist(), "win": p[:, 1].tolist(),
                "median_avg_r": float(np.median(p[:, 0])),
                "p5": float(np.percentile(p[:, 0], 5)), "p95": float(np.percentile(p[:, 0], 95)),
                "p_value_ob_better": float(pval),
            }
            print(f"   placebo {mode:12s} real {real['avg_r']:+.3f}R vs placebo median "
                  f"{np.median(p[:, 0]):+.3f}R  p={pval:.2f}")

        # 3. full sweep + walk-forward (first half picks, second half tests)
        half = len(df) // 2
        parts = {"full": df, "is": df.iloc[:half], "oos": df.iloc[half:]}
        atrs = {k: atr(v) for k, v in parts.items()}
        for dcfg in SWEEP_D:
            evs = {k: detect_order_blocks(v, dcfg) for k, v in parts.items()}
            for tcfg in SWEEP_T:
                row = {"dataset": key, "detect": dcfg.label(), "trade": tcfg.label(), "rr": tcfg.rr,
                       "swing_len": dcfg.swing_len, "ob_rule": dcfg.ob_rule, "zone": dcfg.zone,
                       "fvg": dcfg.require_fvg, "entry": tcfg.entry}
                for k, v in parts.items():
                    s = summarize(simulate(v, evs[k], tcfg, cost=ds.cost, atr_arr=atrs[k]), n_boot=500)
                    row |= {f"{k}_{m}": s[m] for m in ("n", "win_rate", "avg_r", "ci_lo", "ci_hi")}
                sweep_rows.append(row)

        # 4. bias ladder + library backtest
        steps, solo = bias_ladder(df, ds.cost, a)
        lib = smc_library_backtest(df, ds.cost, a)
        for name, t in {**steps, **{f"solo: {k}": v for k, v in solo.items()},
                        "smartmoneyconcepts lib, naive": lib}.items():
            s = summarize(t)
            bias_rows.append({"dataset": key, "step": name, **s})
            print(f"   {name:34s} n={s['n']:4d} win={s['win_rate']:.3f} avg={s['avg_r']:+.3f}R")

    headline = pd.DataFrame(headline_rows)
    sweep = pd.DataFrame(sweep_rows)
    bias = pd.DataFrame(bias_rows)
    headline.to_csv(OUT / "headline.csv", index=False)
    sweep.to_csv(OUT / "sweep.csv", index=False)
    bias.to_csv(OUT / "bias.csv", index=False)

    # walk-forward: best in-sample config per dataset, scored out-of-sample
    for key, g in sweep.groupby("dataset"):
        g = g[g.is_n >= 20]
        best = g.sort_values("is_avg_r", ascending=False).iloc[0]
        rho = g[["is_avg_r", "oos_avg_r"]].corr(method="spearman").iloc[0, 1]
        wf_rows.append({"dataset": key, "best_config": f"{best.detect} {best.trade}",
                        "is_avg_r": best.is_avg_r, "is_n": best.is_n,
                        "oos_avg_r": best.oos_avg_r, "oos_n": best.oos_n,
                        "oos_median_all_configs": g.oos_avg_r.median(),
                        "is_oos_rank_corr": rho,
                        "share_configs_positive_both_halves": float(((g.is_avg_r > 0) & (g.oos_avg_r > 0)).mean())})
    wf = pd.DataFrame(wf_rows)
    wf.to_csv(OUT / "walk_forward.csv", index=False)

    pooled_t = pd.concat(pooled)
    ps = summarize(pooled_t)
    sig_pos = int((sweep.full_ci_lo > 0).sum())
    sig_neg = int((sweep.full_ci_hi < 0).sum())
    summary |= {
        "pooled_headline": ps,
        "pooled_headline_gross": summarize(pooled_t, "r_gross"),
        "sweep": {"runs": len(sweep), "ci_above_zero": sig_pos, "ci_below_zero": sig_neg,
                  "share_avg_r_positive": float((sweep.full_avg_r > 0).mean()),
                  "median_avg_r": float(sweep.full_avg_r.median()),
                  "median_avg_r_by_dataset": sweep.groupby("dataset").full_avg_r.median().to_dict()},
        "walk_forward": wf.to_dict(orient="records"),
        "placebo": {k: {kk: (vv if not isinstance(vv, dict) else {m: vv[m] for m in vv if m not in ("avg_r", "win")})
                        for kk, vv in v.items()} for k, v in placebo_store.items()},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    (OUT / "placebo_draws.json").write_text(json.dumps(placebo_store, default=float))
    print("\npooled headline:", {k: round(v, 3) if isinstance(v, float) else v for k, v in ps.items()})
    print(f"sweep: {sig_pos}/{len(sweep)} runs with 95% CI above 0, {sig_neg} below 0")
    print(wf.to_string())

    try:
        from ob.charts import make_all
        make_all(OUT)
    except ImportError as e:  # charts are optional
        print("charts skipped:", e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
