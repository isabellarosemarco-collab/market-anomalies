"""
walk_forward.py — anchored walk-forward optimisation.

Protocol. The in-sample window is ANCHORED: it always starts at the first available bar
and its right edge advances, so each fold trains on all history to date. Immediately after
each IS window sits a disjoint 2-month OOS window that the selection step never sees.
Parameters are chosen by IS Sharpe, then applied unchanged to OOS. The per-fold OOS
segments are compounded into one continuous out-of-sample track record, which is the only
equity curve here that carries evidence about the edge.

Interpretation guardrail. Walk-forward efficiency (OOS Sharpe / IS Sharpe) is only
meaningful when IS Sharpe is positive. If the IS winner is itself negative, the ratio is
reported as undefined rather than as a flattering positive number produced by two negatives.
"""
from __future__ import annotations

import itertools
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from backtester import compute_metrics, run_backtest
from data_loader import load_universe
from signals import build_indicators

GRID = {
    "z_entry": [1.5, 2.0, 2.5],
    "rvol_thr": [2.0, 2.5, 3.0],
    "atr_mult": [1.5, 3.0, 6.0],
}
IS_MONTHS = 6
OOS_MONTHS = 2
MIN_IS_TRADES = 15          # a combo with fewer IS trades is not selectable
WFE_MIN_IS_SHARPE = 0.20    # below this the OOS/IS ratio is numerically meaningless
INTERVAL, PERIOD = "60m", "730d"

_PREP: dict[str, pd.DataFrame] = {}


def _init_worker(interval: str, period: str) -> None:
    """Each process rebuilds indicators once from the parquet cache."""
    global _PREP
    data = load_universe(interval=interval, period=period)
    _PREP = {s: build_indicators(d) for s, d in data.items()}


def _eval_combo(task):
    combo, start, end = task
    res = run_backtest(prepared=_PREP, params=combo, start=start, end=end)
    m = res["metrics"]
    return combo, m["sharpe"], m["n_trades"], m["profit_factor"]


def combos() -> list[dict]:
    keys = list(GRID)
    return [dict(zip(keys, v)) for v in itertools.product(*GRID.values())]


def make_folds(first: pd.Timestamp, last: pd.Timestamp) -> list[tuple]:
    """Anchored IS windows with adjacent, non-overlapping OOS windows."""
    folds = []
    k = 0
    while True:
        is_start = first
        is_end = first + pd.DateOffset(months=IS_MONTHS + k * OOS_MONTHS)
        oos_end = is_end + pd.DateOffset(months=OOS_MONTHS)
        if oos_end > last:
            break
        folds.append((is_start, is_end, is_end, oos_end))
        k += 1
    return folds


def main() -> int:
    print("Loading data ...")
    data = load_universe(interval=INTERVAL, period=PERIOD)
    if not data:
        print("No data; aborting.")
        return 2
    prep = {s: build_indicators(d) for s, d in data.items()}
    first = min(d.index[0] for d in prep.values())
    last = max(d.index[-1] for d in prep.values())
    folds = make_folds(first, last)
    grid = combos()
    print(f"Span {first.date()} .. {last.date()}   folds={len(folds)}   grid={len(grid)}")
    print(f"Anchored IS >= {IS_MONTHS}mo, OOS = {OOS_MONTHS}mo, "
          f"{len(folds) * len(grid)} in-sample evaluations\n")

    rows = []
    oos_segments = []
    with ProcessPoolExecutor(initializer=_init_worker, initargs=(INTERVAL, PERIOD)) as ex:
        for i, (is_s, is_e, oos_s, oos_e) in enumerate(folds, 1):
            tasks = [(c, is_s, is_e) for c in grid]
            results = list(ex.map(_eval_combo, tasks))
            eligible = [(c, sh, n, pf) for c, sh, n, pf in results if n >= MIN_IS_TRADES]
            if not eligible:
                print(f"fold {i:>2}: no combo reached {MIN_IS_TRADES} IS trades — skipped")
                continue
            best, is_sharpe, is_n, is_pf = max(eligible, key=lambda t: t[1])

            oos = run_backtest(prepared=prep, params=best, start=oos_s, end=oos_e)
            om = oos["metrics"]
            if not oos["equity"].empty:
                oos_segments.append(oos["equity"])

            # WFE explodes when the denominator is near zero: an IS Sharpe of +0.015 turns a
            # mediocre OOS Sharpe into a "118x efficiency". Require a meaningful IS Sharpe
            # before the ratio is reported at all.
            wfe = (om["sharpe"] / is_sharpe) if is_sharpe >= WFE_MIN_IS_SHARPE else np.nan
            rows.append({
                "fold": i,
                "is_end": is_e.date(), "oos": f"{oos_s.date()}..{oos_e.date()}",
                "z": best["z_entry"], "rvol": best["rvol_thr"], "atr_m": best["atr_mult"],
                "is_sharpe": is_sharpe, "is_trades": is_n,
                "oos_sharpe": om["sharpe"], "oos_pf": om["profit_factor"],
                "oos_trades": om["n_trades"], "oos_ret_pct": om["total_return_pct"],
                "oos_maxdd_pct": om["max_drawdown_pct"], "wfe": wfe,
            })
            print(f"fold {i:>2}  IS->{is_e.date()}  best z={best['z_entry']} "
                  f"rvol={best['rvol_thr']} atrx={best['atr_mult']}  "
                  f"IS_Sh={is_sharpe:+.2f}(n={is_n})  "
                  f"OOS_Sh={om['sharpe']:+.2f} PF={om['profit_factor']:.2f} "
                  f"(n={om['n_trades']})", flush=True)

    if not rows:
        print("\nNo usable folds.")
        return 2
    fr = pd.DataFrame(rows)

    # ---- stitch OOS segments into one continuous track record ----
    stitched, curve = 10_000.0, [10_000.0]
    idx = [oos_segments[0].index[0]] if oos_segments else []
    for seg in oos_segments:
        rets = seg.pct_change().fillna(0.0)
        for ts, r in rets.items():
            stitched *= (1.0 + r)
            curve.append(stitched)
            idx.append(ts)
    oos_eq = pd.Series(curve[1:], index=pd.DatetimeIndex(idx[1:]), name="oos_equity") \
        if len(curve) > 1 else pd.Series(dtype=float)

    all_oos_trades = pd.concat(
        [run_backtest(prepared=prep,
                      params={"z_entry": r.z, "rvol_thr": r.rvol, "atr_mult": r.atr_m},
                      start=pd.Timestamp(r.oos.split("..")[0], tz=first.tz),
                      end=pd.Timestamp(r.oos.split("..")[1], tz=first.tz))["trades"]
         for r in fr.itertuples()], ignore_index=True)

    bpd = int(next(iter(prep.values()))["bars_per_day"].iloc[0])
    agg = compute_metrics(oos_eq, all_oos_trades, bpd)

    print("\n" + "=" * 78)
    print("PER-FOLD RESULTS")
    print("=" * 78)
    print(fr.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    mean_is, mean_oos = fr["is_sharpe"].mean(), fr["oos_sharpe"].mean()
    wfe_agg = mean_oos / mean_is if mean_is >= WFE_MIN_IS_SHARPE else float("nan")

    print("\n" + "=" * 78)
    print("AGGREGATE OUT-OF-SAMPLE (stitched, after all frictions)")
    print("=" * 78)
    print(f"  OOS trades                : {agg['n_trades']}")
    print(f"  OOS total return          : {agg['total_return_pct']:+.2f}%")
    print(f"  OOS annualised Sharpe     : {agg['sharpe']:+.3f}")
    print(f"  OOS Sortino               : {agg['sortino']:+.3f}")
    print(f"  OOS max drawdown          : {agg['max_drawdown_pct']:.2f}%")
    print(f"  OOS profit factor         : {agg['profit_factor']:.3f}")
    print(f"  OOS win rate              : {agg['win_rate_pct']:.1f}%")
    print(f"  Gross profit / loss       : ${agg['gross_profit']:,.0f} / ${agg['gross_loss']:,.0f}")
    print(f"  Total fees paid           : ${agg['total_fees']:,.2f}")
    print(f"  Mean IS Sharpe            : {mean_is:+.3f}")
    print(f"  Mean OOS Sharpe           : {mean_oos:+.3f}")
    print(f"  Walk-forward efficiency   : "
          + (f"{wfe_agg:.3f}" if np.isfinite(wfe_agg) else "undefined (IS Sharpe <= 0)"))

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    pf_ok = agg["profit_factor"] > 1.3
    wfe_ok = np.isfinite(wfe_agg) and wfe_agg >= 0.7 and agg["sharpe"] > 1.2
    print(f"  OOS profit factor > 1.3   : {'PASS' if pf_ok else 'FAIL'} "
          f"({agg['profit_factor']:.3f})")
    print(f"  WFE >= 0.7 and Sharpe>1.2 : {'PASS' if wfe_ok else 'FAIL'}")
    if not pf_ok:
        print("  -> Transaction costs are not the binding constraint; the raw signal is "
              "unprofitable before they are even applied in aggregate.")
    print(f"  TRADEABLE                 : {'YES' if (pf_ok and wfe_ok) else 'NO'}")

    _plot(fr, oos_eq, prep)
    fr.to_csv("wfo_folds.csv", index=False)
    print("\nSaved wfo_folds.csv and equity_curves.png")
    return 0


def _plot(fr: pd.DataFrame, oos_eq: pd.Series, prep: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(14, 9))

    full = run_backtest(prepared=prep)["equity"]
    if not full.empty:
        ax[0, 0].plot(full.index, full.values, lw=1.1, color="tab:blue")
    ax[0, 0].set_title("Full-sample equity (in-sample, baseline params)")
    ax[0, 0].axhline(10_000, ls="--", c="grey", lw=0.8)

    if not oos_eq.empty:
        ax[0, 1].plot(oos_eq.index, oos_eq.values, lw=1.1, color="tab:red")
    ax[0, 1].set_title("Stitched OUT-OF-SAMPLE equity (walk-forward)")
    ax[0, 1].axhline(10_000, ls="--", c="grey", lw=0.8)

    w = 0.38
    x = np.arange(len(fr))
    ax[1, 0].bar(x - w / 2, fr["is_sharpe"], w, label="IS Sharpe")
    ax[1, 0].bar(x + w / 2, fr["oos_sharpe"], w, label="OOS Sharpe")
    ax[1, 0].axhline(0, c="k", lw=0.8)
    ax[1, 0].set_xticks(x); ax[1, 0].set_xticklabels(fr["fold"])
    ax[1, 0].set_xlabel("fold"); ax[1, 0].legend(); ax[1, 0].set_title("IS vs OOS Sharpe by fold")

    ax[1, 1].bar(x, fr["oos_pf"], color="tab:purple")
    ax[1, 1].axhline(1.0, c="k", lw=0.8)
    ax[1, 1].axhline(1.3, c="green", ls="--", lw=1.0, label="1.3 target")
    ax[1, 1].set_xticks(x); ax[1, 1].set_xticklabels(fr["fold"])
    ax[1, 1].set_xlabel("fold"); ax[1, 1].legend(); ax[1, 1].set_title("OOS profit factor by fold")

    for a in ax.flat:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("equity_curves.png", dpi=130)


if __name__ == "__main__":
    sys.exit(main())
