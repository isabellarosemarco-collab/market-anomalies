"""
overnight_drift.py — decomposing the equity premium into overnight and intraday sessions.

WHY UNADJUSTED PRICES PLUS AN EXPLICIT DIVIDEND TERM

The obvious implementation loads auto_adjust=True bars and computes Open_t / Close_{t-1}. That is
subtly wrong in a way that flatters the hypothesis. yfinance's adjustment factor steps on the
ex-dividend date, so the ex-date price drop is removed from the overnight return and the dividend
is silently credited to the overnight bucket — while the intraday bucket (Close_t / Open_t, both
scaled by the same factor) receives none of it. A naive adjusted decomposition therefore hands the
overnight session the entire dividend yield, roughly 1.3-1.8%/yr on SPY, and reports it as drift.

Economically the credit is real: a holder who buys at the close of t-1 and sells at the open of the
ex-date t is the holder of record and does receive the dividend. So it belongs in the overnight
return. It just is not a demand-shock anomaly, and it must be reported separately, which requires
raw prices and a dividend column:

    R_night_total = (Open_t + Div_t) / Close_{t-1} - 1      <- what an overnight holder earns
    R_night_price = Open_t / Close_{t-1} - 1                <- drift only, dividend stripped out
    R_day         = Close_t / Open_t - 1                    <- no dividend accrues intraday
    R_bh          = (Close_t + Div_t) / Close_{t-1} - 1     <- total-return control

The exact compounding identity needs care about when the dividend is reinvested. Chaining the two
sessions implicitly reinvests it at Open_t, where it then earns R_day, whereas the conventional
total-return formula above holds it as cash. So the identity that holds to floating-point precision
is against a reinvested control,

    (1 + R_night_total)(1 + R_day) = 1 + R_bh_reinv,  R_bh_reinv = Close_t(1 + Div_t/Open_t)/Close_{t-1} - 1

and that is what gets asserted at load time, because if it fails the decomposition is meaningless.
The gap between R_bh and R_bh_reinv is ~3bp on ~110 ex-dates out of ~8,000 sessions — immaterial to
every conclusion here, but the check has to be stated correctly to be worth running at all.

EXECUTION CAVEAT ON THE CONDITIONAL STRATEGY

"Go long overnight if today's intraday return was negative" needs the intraday return at the moment
the MOC order is submitted, but MOC cutoff is 15:50, not 16:00. The last ten minutes are unknown.
Two variants are therefore reported: `cond_t1` uses the same-day intraday return (the literature's
formulation, mildly optimistic about execution) and `cond_t2` uses a fully lagged one (no ambiguity
whatsoever). If the edge lives only in cond_t1, that is an execution-risk finding, not an anomaly.

t-statistics are Newey-West (HAC) throughout.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

SYMBOLS = ["SPY", "QQQ", "IWM"]
START = "1993-01-01"   # SPY inception; QQQ 1999-03, IWM 2000-05 start at their own first bar
ANN = 252

COSTS = {
    "slippage_pct": 0.0002,     # 2bp per side, on MOO and MOC auction fills
    "fee_per_share": 0.005,
    "notional": 10_000.0,
}


# ----------------------------------------------------------------------------- data

def load_ohlc_div(symbols: list[str], start: str = START) -> dict[str, pd.DataFrame]:
    """Raw (unadjusted) OHLC plus the dividend column, keyed by ex-date."""
    import yfinance as yf

    out = {}
    for sym in symbols:
        df = yf.Ticker(sym).history(start=start, interval="1d",
                                    auto_adjust=False, actions=True)
        if df is None or df.empty:
            print(f"  {sym}: no data")
            continue
        df = df.rename(columns=str.lower)
        cols = ["open", "high", "low", "close"]
        df["dividends"] = df.get("dividends", 0.0)
        df = df[cols + ["volume", "dividends"]].copy()
        # Splits are handled by yfinance's raw series being split-adjusted already; guard anyway
        # against zero/NaN prices which would create infinite returns.
        df = df.dropna(subset=cols)
        df = df[(df[cols] > 0).all(axis=1)]
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        out[sym] = df
    return out


def decompose(df: pd.DataFrame) -> pd.DataFrame:
    """Split each day into non-overlapping overnight and intraday components."""
    o, c, d = df["open"], df["close"], df["dividends"].fillna(0.0)
    pc = c.shift(1)

    r = pd.DataFrame(index=df.index)
    r["r_night"] = (o + d) / pc - 1.0          # overnight holder receives the dividend
    r["r_night_px"] = o / pc - 1.0             # price-only, dividend stripped
    r["r_day"] = c / o - 1.0
    r["r_bh"] = (c + d) / pc - 1.0                        # cash dividend, conventional control
    r["r_bh_reinv"] = c * (1 + d / o) / pc - 1.0          # dividend reinvested at the open
    r["div_yield"] = d / pc
    r["close"] = c
    r = r.dropna()

    # Integrity: the two sessions must reconstruct the full day exactly.
    resid = (1 + r["r_night"]) * (1 + r["r_day"]) - (1 + r["r_bh_reinv"])
    worst = float(resid.abs().max())
    if worst > 1e-12:
        raise AssertionError(f"decomposition identity violated, max resid {worst:.3e}")
    return r


# ------------------------------------------------------------------- stats helpers

def hac_t(x: pd.Series, lags: int = 5) -> float:
    """Newey-West t-statistic for H0: mean = 0."""
    x = x.dropna()
    if len(x) < 30:
        return float("nan")
    fit = sm.OLS(x.values, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(fit.tvalues[0])


def max_dd(r: pd.Series) -> float:
    eq = (1 + r.fillna(0.0)).cumprod()
    return float((eq / eq.cummax() - 1).min() * 100)


def stats(r: pd.Series, label: str, exposure: pd.Series | None = None) -> dict:
    r = r.dropna()
    if r.empty:
        return {"strategy": label}
    yrs = len(r) / ANN
    eq = (1 + r).cumprod()
    dn = r[r < 0]
    return {
        "strategy": label,
        "days": len(r),
        "exposed_%": 100.0 if exposure is None else float(100 * exposure.reindex(r.index).fillna(0).mean()),
        "ann_ret_%": 100 * (eq.iloc[-1] ** (1 / yrs) - 1),
        "ann_vol_%": 100 * r.std(ddof=1) * np.sqrt(ANN),
        "sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(ANN)) if r.std(ddof=1) > 0 else np.nan,
        "sortino": float(r.mean() / dn.std(ddof=1) * np.sqrt(ANN)) if len(dn) > 2 else np.nan,
        "maxdd_%": max_dd(r),
        "t_stat": hac_t(r),
        "mean_bp/day": 1e4 * r.mean(),
    }


# ------------------------------------------------------------------------ frictions

def cost_per_day(price: pd.Series, slippage: float, costs: dict) -> pd.Series:
    """Round-trip cost as a return fraction: two auction fills per active day."""
    fee_frac = costs["fee_per_share"] / price
    return 2.0 * (slippage + fee_frac)


def apply_costs(gross: pd.Series, active: pd.Series, price: pd.Series,
                slippage: float, costs: dict) -> pd.Series:
    return gross - active.reindex(gross.index).fillna(0.0) * cost_per_day(price, slippage, costs)


# ----------------------------------------------------------------------- strategies

def signals(r: pd.DataFrame, win: int = 20, q: float = 0.0) -> dict[str, pd.Series]:
    """Conditional overnight entries. q=0 means 'strictly negative intraday return';
    q>0 means 'below the rolling q-quantile of intraday returns'.

    Every threshold is computed on a rolling window ending at the observation used, then shifted,
    so no future information enters the entry decision.
    """
    rd = r["r_day"]
    if q <= 0:
        trigger = rd < 0
    else:
        thr = rd.rolling(win, min_periods=win).quantile(q)
        trigger = rd < thr
    return {
        "cond_t1": trigger.shift(1).fillna(False),   # decided at close t-1 (MOC-cutoff risk)
        "cond_t2": trigger.shift(2).fillna(False),   # fully lagged, no ambiguity
    }


def baseline_table(r: pd.DataFrame, slippage: float, costs: dict) -> pd.DataFrame:
    on = pd.Series(1.0, index=r.index)
    rows = [
        stats(r["r_night"], "A. overnight (gross, incl. div)"),
        stats(r["r_night_px"], "A2. overnight (gross, price only)"),
        stats(r["r_day"], "B. intraday (gross)"),
        stats(r["r_bh"], "C. buy & hold (total return)"),
        stats(apply_costs(r["r_night"], on, r["close"], slippage, costs),
              f"A. overnight NET @{slippage*1e4:.0f}bp/side"),
        stats(apply_costs(r["r_day"], on, r["close"], slippage, costs),
              f"B. intraday NET @{slippage*1e4:.0f}bp/side"),
    ]
    return pd.DataFrame(rows).set_index("strategy")


def conditional_table(r: pd.DataFrame, slippage: float, costs: dict,
                      grid=((20, 0.0), (20, 0.2), (20, 0.3), (60, 0.2))) -> pd.DataFrame:
    rows = []
    for win, q in grid:
        sig = signals(r, win, q)
        for name, s in sig.items():
            act = s.astype(float)
            gross = r["r_night"] * act
            net = apply_costs(gross, act, r["close"], slippage, costs)
            tag = f"q={q:.1f} win={win} {name}"
            g = stats(gross, f"{tag} GROSS", act)
            n = stats(net, f"{tag} NET", act)
            rows += [g, n]
    return pd.DataFrame(rows).set_index("strategy")


def friction_sweep(r: pd.DataFrame, costs: dict, active: pd.Series | None = None,
                   label: str = "overnight") -> pd.DataFrame:
    """Sharpe and net return as a function of per-side slippage, with breakevens."""
    act = pd.Series(1.0, index=r.index) if active is None else active.astype(float)
    gross = r["r_night"] * act
    rows = []
    for bp in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]:
        net = apply_costs(gross, act, r["close"], bp / 1e4, costs)
        st = stats(net, f"{label} @{bp}bp", act)
        rows.append({"slippage_bp/side": bp, "sharpe": st["sharpe"],
                     "ann_ret_%": st["ann_ret_%"], "t_stat": st["t_stat"],
                     "maxdd_%": st["maxdd_%"]})
    return pd.DataFrame(rows).set_index("slippage_bp/side")


def breakevens(r: pd.DataFrame, costs: dict, active: pd.Series | None = None) -> dict:
    """Solve for the per-side slippage at which Sharpe hits 0.8 and at which return hits 0."""
    act = pd.Series(1.0, index=r.index) if active is None else active.astype(float)
    gross = r["r_night"] * act

    def f(bp, target):
        net = apply_costs(gross, act, r["close"], bp / 1e4, costs)
        s = net.std(ddof=1)
        sh = net.mean() / s * np.sqrt(ANN) if s > 0 else 0.0
        return sh - target

    def solve(target):
        lo, hi = 0.0, 50.0
        if f(lo, target) < 0:
            return 0.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if f(mid, target) > 0:
                lo = mid
            else:
                hi = mid
        return lo

    return {"bp_for_sharpe_0.8": solve(0.8), "bp_for_sharpe_0.0": solve(0.0),
            "days_traded": float(act.sum())}


# --------------------------------------------------------------------- walk-forward

def wfo(r: pd.DataFrame, costs: dict, slippage: float,
        is_years: int = 5, oos_months: int = 12) -> tuple[pd.DataFrame, pd.Series]:
    """Anchored 5y IS / 1y OOS. The grid selects the conditional threshold; 'always on' is
    included so the optimizer can reject conditioning entirely."""
    grid = [("always", None, None)] + [(n, w, q) for w, q in ((20, 0.0), (20, 0.2), (20, 0.3), (60, 0.2))
                                       for n in ("cond_t1", "cond_t2")]
    start = r.index[0]
    last = r.index[-1]
    rows, segs = [], []
    k = 0
    while True:
        is_end = start + pd.DateOffset(years=is_years, months=k * oos_months)
        if is_end >= last:
            break
        oos_end = min(is_end + pd.DateOffset(months=oos_months), last)

        def build(name, w, q, sl):
            if name == "always":
                act = pd.Series(1.0, index=r.index)
            else:
                act = signals(r, w, q)[name].astype(float)
            return apply_costs(r["r_night"] * act, act, r["close"], sl, costs), act

        best, best_sh = None, -np.inf
        for name, w, q in grid:
            net, act = build(name, w, q, slippage)
            seg = net[(net.index >= start) & (net.index < is_end)]
            if len(seg) < 250 or act[(act.index >= start) & (act.index < is_end)].sum() < 60:
                continue
            sd = seg.std(ddof=1)
            sh = seg.mean() / sd * np.sqrt(ANN) if sd > 0 else -np.inf
            if sh > best_sh:
                best_sh, best = sh, (name, w, q)
        if best is None:
            k += 1
            continue

        net, act = build(*best, slippage)
        oos = net[(net.index >= is_end) & (net.index < oos_end)]
        aoos = act.reindex(oos.index).fillna(0.0)
        if oos.empty:
            k += 1
            continue
        segs.append(oos)
        st = stats(oos, "oos", aoos)
        rows.append({"fold": len(rows) + 1, "oos": f"{is_end.date()}..{oos_end.date()}",
                     "cfg": best[0] if best[0] == "always" else f"{best[0]} q={best[2]} w={best[1]}",
                     "is_sharpe": best_sh, "oos_sharpe": st["sharpe"], "oos_sortino": st["sortino"],
                     "oos_ret_%": st["ann_ret_%"], "oos_dd_%": st["maxdd_%"],
                     "oos_t": st["t_stat"], "exposed_%": st["exposed_%"]})
        k += 1
    stitched = pd.concat(segs).sort_index() if segs else pd.Series(dtype=float)
    stitched = stitched[~stitched.index.duplicated(keep="first")]
    return pd.DataFrame(rows), stitched


# ---------------------------------------------------------------------------- main

def fmt(df: pd.DataFrame) -> str:
    return df.to_string(float_format=lambda x: f"{x:,.3f}")


def main() -> int:
    print("Loading raw (unadjusted) daily bars with dividends...")
    data = load_ohlc_div(SYMBOLS)
    dec = {s: decompose(d) for s, d in data.items()}
    print("Decomposition identity (1+r_night)(1+r_day) == (1+r_bh): OK for all symbols\n")

    for sym, r in dec.items():
        span = f"{r.index[0].date()}..{r.index[-1].date()}"
        dy = 100 * r["div_yield"].sum() / ((r.index[-1] - r.index[0]).days / 365.25)
        print("=" * 108)
        print(f"{sym}   {span}   {len(r):,} sessions   avg dividend yield {dy:.2f}%/yr")
        print("=" * 108)

        print("\n--- 1. BASELINE DECOMPOSITION (t-stat vs H0: mean = 0, Newey-West) ---")
        print(fmt(baseline_table(r, COSTS["slippage_pct"], COSTS)))

        print("\n--- 2. CONDITIONAL OVERNIGHT (long overnight only after intraday weakness) ---")
        print(fmt(conditional_table(r, COSTS["slippage_pct"], COSTS)))

        print("\n--- 3a. FRICTION SWEEP, unconditional overnight ---")
        print(fmt(friction_sweep(r, COSTS)))
        be = breakevens(r, COSTS)
        print(f"    breakeven: Sharpe 0.8 at {be['bp_for_sharpe_0.8']:.2f}bp/side, "
              f"Sharpe 0.0 at {be['bp_for_sharpe_0.0']:.2f}bp/side")

        act = signals(r, 20, 0.0)["cond_t2"].astype(float)
        print("\n--- 3b. FRICTION SWEEP, conditional (cond_t2, q=0) ---")
        print(fmt(friction_sweep(r, COSTS, act, "cond")))
        be2 = breakevens(r, COSTS, act)
        print(f"    breakeven: Sharpe 0.8 at {be2['bp_for_sharpe_0.8']:.2f}bp/side, "
              f"Sharpe 0.0 at {be2['bp_for_sharpe_0.0']:.2f}bp/side, "
              f"{be2['days_traded']:,.0f} nights traded")

        print("\n--- 4. SUBPERIOD STABILITY, gross overnight vs intraday ---")
        cuts = [("2000-2007", "2000-01-01", "2008-01-01"), ("2008-2012", "2008-01-01", "2013-01-01"),
                ("2013-2019", "2013-01-01", "2020-01-01"), ("2020-2026", "2020-01-01", "2027-01-01")]
        rows = []
        for lab, a, b in cuts:
            w = r[(r.index >= a) & (r.index < b)]
            if len(w) < 120:
                continue
            nt, dy_, bh = stats(w["r_night"], "n"), stats(w["r_day"], "d"), stats(w["r_bh"], "b")
            net = stats(apply_costs(w["r_night"], pd.Series(1.0, index=w.index),
                                    w["close"], COSTS["slippage_pct"], COSTS), "x")
            rows.append({"period": lab, "days": len(w),
                         "night_ret_%": nt["ann_ret_%"], "night_sharpe": nt["sharpe"],
                         "night_t": nt["t_stat"], "night_NET_ret_%": net["ann_ret_%"],
                         "day_ret_%": dy_["ann_ret_%"], "day_sharpe": dy_["sharpe"],
                         "day_t": dy_["t_stat"], "bh_ret_%": bh["ann_ret_%"]})
        print(fmt(pd.DataFrame(rows).set_index("period")))

        print("\n--- 5. ANCHORED WALK-FORWARD (5y IS / 1y OOS, net of 2bp/side) ---")
        folds, stitched = wfo(r, COSTS, COSTS["slippage_pct"])
        if folds.empty:
            print("    insufficient history for 5y IS folds")
        else:
            print(fmt(folds.set_index("fold")))
            agg = stats(stitched, "aggregate OOS")
            post = stitched[stitched.index >= "2020-01-01"]
            print(f"\n    AGGREGATE OOS  sharpe={agg['sharpe']:+.3f}  sortino={agg['sortino']:+.3f}  "
                  f"ann_ret={agg['ann_ret_%']:+.2f}%  maxdd={agg['maxdd_%']:.2f}%  "
                  f"t={agg['t_stat']:+.2f}")
            if len(post) > 120:
                ap = stats(post, "post2020")
                print(f"    POST-2020 OOS  sharpe={ap['sharpe']:+.3f}  "
                      f"ann_ret={ap['ann_ret_%']:+.2f}%  maxdd={ap['maxdd_%']:.2f}%  "
                      f"t={ap['t_stat']:+.2f}  ({ap['days']} days)")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
