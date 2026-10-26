"""
walk_forward_xsec.py — anchored WFO for cross-sectional momentum.

Two universes, same machinery, selected by CLI arg:
    python walk_forward_xsec.py sectors     # 9 broad SPDR sectors (the narrow-dispersion test)
    python walk_forward_xsec.py industry    # 41 sub-industry / thematic ETFs (the wide test)

Windows are years, not months: a monthly-rebalanced model sees ~12 observations a year, so the
6-month/2-month scheme used for the intraday test would select parameters on a handful of
rebalances. IS is anchored from 2015 with a minimum of 4 years; OOS is the following 12 months.

Benchmarking discipline. A long-only rotation is mostly a bet on equities, so "OOS Sharpe > 0.8"
is not evidence of an anomaly. Every window is scored against two controls on identical dates —
SPY buy-and-hold, and equal-weight-the-whole-universe (does RANKING beat not ranking?) — and
against a regression alpha. The regression is the load-bearing test: a long-only book carries
beta by construction, and only the intercept of

    r_strategy,t = alpha + beta * r_benchmark,t + e_t

is return the benchmark does not already explain. t-statistics use Newey-West (HAC, 5 lags)
because monthly rebalancing induces autocorrelation in daily strategy returns, which would make
plain OLS standard errors too small and the alpha look more significant than it is.
"""
from __future__ import annotations

import itertools
import sys

import numpy as np
import pandas as pd

from backtester import compute_metrics
from backtester_xsec import run_xsec_backtest
from data_loader import BENCH, CASH, INDUSTRY, SECTORS, load_daily
from signals_xsec import build_panel

# short_book stays in the grid rather than being pinned to 0. It was rejected in all 8 folds of
# the 9-sector run, but that universe had almost no cross-sectional spread to harvest
# (see dispersion_check.py: 0.25%/mo top-minus-bottom vs 1.51%/mo across 41 industries), so the
# rejection was a property of the universe, not of shorting. Gross exposure is held at 1.0 either
# way so the two configurations are compared at equal risk budget, not equal capital.
GRID = {
    "lookback": [126, 252],
    "n_sel": [4, 8],
    "use_regime": [False, True],
    "short_book": [0.0, 0.5],
}
BASE = {"skip": 21, "require_positive": True, "max_weight": 0.35}

IS_MIN_YEARS = 4
OOS_MONTHS = 12
START = pd.Timestamp("2015-01-01")
DATA_START = "2014-01-01"


def combos() -> list[dict]:
    keys = list(GRID)
    out = []
    for v in itertools.product(*GRID.values()):
        c = {**BASE, **dict(zip(keys, v))}
        # Hold gross exposure at 1.0: long-only spends the whole budget long, dollar-neutral
        # splits it 0.5/0.5. Otherwise the long/short variant would be compared at half the risk.
        c["long_book"] = 1.0 if c["short_book"] == 0.0 else 0.5
        out.append(c)
    return out


def make_folds(last: pd.Timestamp) -> list[tuple]:
    folds, k = [], 0
    while True:
        is_end = START + pd.DateOffset(years=IS_MIN_YEARS, months=k * OOS_MONTHS)
        if is_end >= last:
            break
        oos_end = is_end + pd.DateOffset(months=OOS_MONTHS)
        folds.append((START, is_end, is_end, min(oos_end, last)))
        k += 1
    return folds


def _bench_returns(px: pd.DataFrame, universe: list[str],
                   s: pd.Timestamp, e: pd.Timestamp) -> tuple[pd.Series, pd.Series]:
    """SPY and equal-weight-universe daily returns over [s, e).

    Equal weight only averages names that already have price history at each date, so it is not
    granted foresight about ETFs that had not listed yet.
    """
    win = px[(px.index >= s) & (px.index < e)]
    spy = win[BENCH].pct_change()
    ew = win[[c for c in universe if c in win]].pct_change().mean(axis=1, skipna=True)
    return spy, ew


def _metrics_from_ret(r: pd.Series) -> dict:
    if r.dropna().empty:
        return compute_metrics(pd.Series(dtype=float), pd.DataFrame(), 1)
    eq = 10_000 * (1 + r.fillna(0.0)).cumprod()
    return compute_metrics(eq, pd.DataFrame(), 1)


def _alpha(strat: pd.Series, bench: pd.Series) -> dict:
    """Newey-West CAPM-style regression of strategy on benchmark returns."""
    import statsmodels.api as sm

    df = pd.concat([strat.rename("s"), bench.rename("b")], axis=1).dropna()
    if len(df) < 60:
        return {"alpha_ann_pct": np.nan, "beta": np.nan, "t": np.nan}
    fit = sm.OLS(df["s"], sm.add_constant(df["b"])).fit(
        cov_type="HAC", cov_kwds={"maxlags": 5}
    )
    return {"alpha_ann_pct": float(fit.params.iloc[0]) * 252 * 100,
            "beta": float(fit.params.iloc[1]),
            "t": float(fit.tvalues.iloc[0])}


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "industry"
    universe = INDUSTRY if mode == "industry" else SECTORS

    data = load_daily(universe + [BENCH, CASH], start=DATA_START)
    held_names = [s for s in universe if s in data]
    missing = sorted(set(universe) - set(held_names))
    px = build_panel(data)
    last = px.index[-1]
    folds, grid = make_folds(last), combos()

    print(f"Mode: {mode}   universe loaded: {len(held_names)}/{len(universe)}")
    if missing:
        print(f"  unavailable: {missing}")
    first = {s: px[s].first_valid_index() for s in held_names}
    late = {s: d.date() for s, d in first.items() if d is not None and d > pd.Timestamp("2014-06-01")}
    if late:
        print(f"  listed after 2014-06 (ineligible until history accrues): {late}")
    print(f"Span {START.date()} .. {last.date()}   folds={len(folds)}   grid={len(grid)}")
    print(f"Anchored IS >= {IS_MIN_YEARS}y, OOS = {OOS_MONTHS}mo, max_weight={BASE['max_weight']}\n")

    rows, seg_rets = [], []
    for i, (is_s, is_e, oos_s, oos_e) in enumerate(folds, 1):
        scored = []
        for c in grid:
            m = run_xsec_backtest(data, held_names, params=c, start=is_s, end=is_e)["metrics"]
            if m["n_trades"] >= 20:
                scored.append((c, m["sharpe"]))
        if not scored:
            print(f"fold {i}: no eligible combo")
            continue
        best, is_sh = max(scored, key=lambda t: t[1])

        oos = run_xsec_backtest(data, held_names, params=best, start=oos_s, end=oos_e)
        om, r = oos["metrics"], oos["net_ret"]
        if not r.dropna().empty:
            seg_rets.append(r)
        spy_r, ew_r = _bench_returns(px, held_names, oos_s, oos_e)
        spy_m, ew_m = _metrics_from_ret(spy_r), _metrics_from_ret(ew_r)
        a_spy, a_ew = _alpha(r, spy_r), _alpha(r, ew_r)

        rows.append({
            "fold": i, "oos": f"{oos_s.date()}..{oos_e.date()}",
            "lb": best["lookback"], "n": best["n_sel"], "reg": best["use_regime"],
            "shrt": best["short_book"],
            "is_sh": is_sh, "oos_sh": om["sharpe"], "oos_ret": om["total_return_pct"],
            "oos_dd": om["max_drawdown_pct"],
            "spy_sh": spy_m["sharpe"], "spy_ret": spy_m["total_return_pct"],
            "spy_dd": spy_m["max_drawdown_pct"],
            "ew_sh": ew_m["sharpe"], "ew_ret": ew_m["total_return_pct"],
            "a_spy": a_spy["alpha_ann_pct"], "b_spy": a_spy["beta"],
            "a_ew": a_ew["alpha_ann_pct"],
            "beat_spy": om["total_return_pct"] > spy_m["total_return_pct"],
            "beat_ew": om["total_return_pct"] > ew_m["total_return_pct"],
        })
        print(f"fold {i:>2} OOS {oos_s.date()}..{oos_e.date()}  "
              f"lb={best['lookback']} n={best['n_sel']} reg={str(best['use_regime']):<5} "
              f"short={best['short_book']}  "
              f"IS={is_sh:+.2f}  OOS_Sh={om['sharpe']:+.2f} ret={om['total_return_pct']:+6.1f}% "
              f"dd={om['max_drawdown_pct']:+6.1f}%  | SPY {spy_m['sharpe']:+.2f}/"
              f"{spy_m['total_return_pct']:+6.1f}%  EW {ew_m['sharpe']:+.2f}/"
              f"{ew_m['total_return_pct']:+6.1f}%  | a_spy={a_spy['alpha_ann_pct']:+6.2f}%",
              flush=True)

    if not rows:
        print("No usable folds.")
        return 2
    fr = pd.DataFrame(rows)

    # Stitch OOS returns into one continuous out-of-sample track record.
    oos_r = pd.concat(seg_rets).sort_index()
    oos_r = oos_r[~oos_r.index.duplicated(keep="first")]
    oos_eq = 10_000 * (1 + oos_r.fillna(0.0)).cumprod()
    s0, s1 = oos_r.index[0], oos_r.index[-1] + pd.Timedelta(days=1)

    all_tr = pd.concat([
        run_xsec_backtest(
            data, held_names,
            params={**BASE, "lookback": r.lb, "n_sel": r.n, "use_regime": r.reg,
                    "short_book": r.shrt,
                    "long_book": 1.0 if r.shrt == 0.0 else 0.5},
            start=pd.Timestamp(r.oos.split("..")[0]),
            end=pd.Timestamp(r.oos.split("..")[1]))["trades"]
        for r in fr.itertuples()], ignore_index=True)
    agg = compute_metrics(oos_eq, all_tr, 1)

    spy_r, ew_r = _bench_returns(px, held_names, s0, s1)
    spy_all, ew_all = _metrics_from_ret(spy_r), _metrics_from_ret(ew_r)
    A_spy, A_ew = _alpha(oos_r, spy_r), _alpha(oos_r, ew_r)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 118)
    print("PER-FOLD RESULTS")
    print("=" * 118)
    print(fr.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    print("\n" + "=" * 118)
    print(f"AGGREGATE OUT-OF-SAMPLE {s0.date()} .. {oos_r.index[-1].date()} (after frictions)")
    print("=" * 118)
    print(f"  {'':<24}{'STRATEGY':>12}{'SPY B&H':>12}{'EW UNIVERSE':>14}")
    for label, key in (("annualised Sharpe", "sharpe"), ("Sortino", "sortino"),
                       ("total return %", "total_return_pct"),
                       ("max drawdown %", "max_drawdown_pct")):
        print(f"  {label:<24}{agg[key]:>12.3f}{spy_all[key]:>12.3f}{ew_all[key]:>14.3f}")
    print(f"  {'profit factor':<24}{agg['profit_factor']:>12.3f}")
    print(f"  {'win rate %':<24}{agg['win_rate_pct']:>12.1f}")
    print(f"  {'position-months':<24}{agg['n_trades']:>12}")

    print("\n  ANNUALISED ALPHA (Newey-West HAC, 5 lags)")
    for name, a in (("vs SPY", A_spy), ("vs EW universe", A_ew)):
        sig = "significant" if abs(a["t"]) > 1.96 else "NOT significant at 5%"
        print(f"    {name:<16} alpha={a['alpha_ann_pct']:+7.2f}%/yr  beta={a['beta']:.2f}  "
              f"t={a['t']:+.2f}  ({sig})")

    mean_is, mean_oos = fr["is_sh"].mean(), fr["oos_sh"].mean()
    wfe = mean_oos / mean_is if mean_is >= 0.20 else float("nan")
    print(f"\n  mean IS Sharpe          : {mean_is:+.3f}")
    print(f"  mean OOS Sharpe         : {mean_oos:+.3f}")
    print("  walk-forward efficiency : "
          + (f"{wfe:.3f}" if np.isfinite(wfe) else "undefined (mean IS Sharpe < 0.20)"))
    print(f"  beat_spy ratio          : {int(fr['beat_spy'].sum())}/{len(fr)} "
          f"({100*fr['beat_spy'].mean():.0f}%)")
    print(f"  beat_ew ratio           : {int(fr['beat_ew'].sum())}/{len(fr)} "
          f"({100*fr['beat_ew'].mean():.0f}%)")

    print("\n" + "=" * 118)
    print("VERDICT")
    print("=" * 118)
    checks = {
        "OOS Sharpe > 0.8": agg["sharpe"] > 0.8,
        "WFE >= 0.7": np.isfinite(wfe) and wfe >= 0.7,
        "Risk-adj. beats SPY": agg["sharpe"] > spy_all["sharpe"],
        "Risk-adj. beats EW universe": agg["sharpe"] > ew_all["sharpe"],
        "Alpha vs SPY > 0, t > 1.96": A_spy["alpha_ann_pct"] > 0 and A_spy["t"] > 1.96,
        "Alpha vs EW > 0, t > 1.96": A_ew["alpha_ann_pct"] > 0 and A_ew["t"] > 1.96,
    }
    for k, v in checks.items():
        print(f"  {k:<30}: {'PASS' if v else 'FAIL'}")
    survives = checks["OOS Sharpe > 0.8"] and checks["Alpha vs SPY > 0, t > 1.96"] \
        and checks["Alpha vs EW > 0, t > 1.96"]
    print(f"\n  ANOMALY SURVIVES              : {'YES' if survives else 'NO'}")

    _plot(fr, oos_eq, spy_r, ew_r, mode)
    fr.to_csv(f"wfo_{mode}_folds.csv", index=False)
    print(f"\nSaved wfo_{mode}_folds.csv and equity_curves_{mode}.png")
    return 0


def _plot(fr, oos_eq, spy_r, ew_r, mode) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(19, 5))
    spy_eq = 10_000 * (1 + spy_r.fillna(0)).cumprod()
    ew_eq = 10_000 * (1 + ew_r.fillna(0)).cumprod()

    ax[0].plot(oos_eq.index, oos_eq.values, lw=1.5, label="Strategy (OOS, stitched)")
    ax[0].plot(spy_eq.index, spy_eq.values, lw=1.1, label="SPY buy & hold", alpha=0.85)
    ax[0].plot(ew_eq.index, ew_eq.values, lw=1.1, label="Equal-weight universe", alpha=0.85)
    ax[0].axhline(10_000, ls="--", c="grey", lw=0.8)
    ax[0].set_yscale("log"); ax[0].set_title(f"OOS equity vs controls ({mode}, log)")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    x = np.arange(len(fr)); w = 0.27
    ax[1].bar(x - w, fr["oos_sh"], w, label="Strategy")
    ax[1].bar(x, fr["spy_sh"], w, label="SPY")
    ax[1].bar(x + w, fr["ew_sh"], w, label="EW universe")
    ax[1].axhline(0, c="k", lw=0.8); ax[1].set_xticks(x); ax[1].set_xticklabels(fr["fold"])
    ax[1].set_xlabel("fold"); ax[1].set_title("OOS Sharpe by fold")
    ax[1].legend(); ax[1].grid(alpha=0.3)

    ax[2].bar(x, fr["a_spy"], color=["tab:green" if v > 0 else "tab:red" for v in fr["a_spy"]])
    ax[2].axhline(0, c="k", lw=0.8); ax[2].set_xticks(x); ax[2].set_xticklabels(fr["fold"])
    ax[2].set_xlabel("fold"); ax[2].set_ylabel("%/yr")
    ax[2].set_title("Annualised alpha vs SPY by fold"); ax[2].grid(alpha=0.3)

    fig.tight_layout(); fig.savefig(f"equity_curves_{mode}.png", dpi=130)


if __name__ == "__main__":
    raise SystemExit(main())
