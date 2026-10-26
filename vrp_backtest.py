"""
vrp_backtest.py — the volatility risk premium, tested against its own 30-year live track record.

WHY THIS DOES NOT USE OPTION CHAINS. The plan was to download EOD SPY/SPX chains and simulate short
straddles or put spreads. That approach has a flaw that no amount of data fixes: a short-vol position's
P&L is a path integral over gamma, so "VIX minus realised vol was positive" does NOT imply the trade
made money unless you delta-hedge continuously. Retail cannot hedge continuously, and every hedge pays
the equity spread. A chain backtest at an assumed 2bp per leg would therefore produce a large, entirely
fictional Sharpe — SPY options quote 1-3 cents wide on a $1-4 premium, which is 50-300bp per leg, 25 to
150 times the assumed cost. Cost assumption, not signal quality, is what decides this question.

WHAT IT USES INSTEAD. CBOE's PUT index (^PUT, daily since 1996-08) IS this strategy, run live for 30
years: sell a cash-secured at-the-money SPX put each month, hold the collateral in T-bills, roll at
expiry. Prices come from real quoted option prices, not a model, and the T-bill income on collateral is
included — which is the correct construction, because the collateral yield is a real part of the return
and omitting it understates the strategy. If VRP harvesting were a Sharpe-1.5 opportunity, thirty years
of the mechanical version of it would say so.

CAVEAT, STATED NOT BURIED. The index rolls at quoted prices a large institution can hit; a retail
account rolling one SPX contract would pay somewhat worse, so treat these numbers as an UPPER bound.
They are still far closer to reality than a chain simulation at 2bp. And cash-secured means unlevered:
that is the right comparison for this account, and it is why the returns look modest rather than
spectacular.

WHAT IS GENUINELY BEING TESTED:
  V1  Does the premium exist?         VIX minus subsequent realised vol, measured causally.
  V2  Does harvesting it beat SPY?    ^PUT vs buy & hold, full sample and 2013+ out of sample.
  V3  Does timing it help?            Hold ^PUT only when the premium is wide / term structure calm.
  V4  Where does it actually hurt?    The tail, sized honestly against SPY's own tail.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")
CACHE = Path(__file__).resolve().parent / ".xsec_cache"
CACHE.mkdir(exist_ok=True)

SPLIT = pd.Timestamp("2013-01-01")
SWITCH_BP = 2.0          # per side, switching between ^PUT-equivalent and cash (an ETF-level cost)


def load(sym: str) -> pd.Series | None:
    f = CACHE / f"vrp_{sym.replace('^','_')}.csv"
    if f.exists():
        return pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0]
    try:
        import yfinance as yf
        d = yf.Ticker(sym).history(start="1990-01-01", auto_adjust=True)
    except Exception:
        return None
    if d.empty or len(d) < 500:
        return None
    d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
    s = d["Close"].dropna()
    s.to_frame("close").to_csv(f)
    return s


def repair(s: pd.Series, jump: float = 0.15, cancel: float = 0.03) -> tuple[pd.Series, list]:
    """Repair one-day SPIKE-AND-REVERSE ticks only, leaving genuine volatility clustering alone.

    Yahoo's ^PUT series contains such a tick: 2020-03-13 prints 2050.28 between 1515.21 and 1468.31,
    a +35% spike that reverses -28% the next session. Left in, that single print dominates every tail
    statistic here — it alone produced the -28.4% "worst day", a -37.1% max drawdown, kurtosis of 358,
    and a POSITIVE skew for a short-volatility strategy, which is not physically possible.

    THE FIRST VERSION OF THIS FUNCTION WAS WORSE THAN THE BUG. Comparing each print to the geometric
    mean of its neighbours flagged three consecutive March 2020 sessions and "repaired" 1515.21 UP to
    1856.25 — it deleted real crash days, understating exactly the drawdown this study exists to
    measure. A neighbour test cannot distinguish a bad tick from real volatility clustering, because
    during a crash consecutive 10% moves ARE the data.

    The test used instead needs no tuned threshold on the reversal: a bad print sits more than `jump`
    away from the day before AND more than `jump` away from the day after, IN THE SAME DIRECTION — a
    spike standing clear of both its neighbours. A real crash day is far from the day before but close
    to the day after, since the level stays down. Repairs are printed, never silent.
    """
    lp = np.log(s)
    back, fwd = lp - lp.shift(1), lp - lp.shift(-1)
    bad = ((back.abs() > jump) & (fwd.abs() > jump) &
           (np.sign(back) == np.sign(fwd))).fillna(False)
    v = s.copy()
    gm = np.sqrt(s.shift(1) * s.shift(-1))
    fixed = [(d.date(), round(s[d], 2), round(gm[d], 2)) for d in s.index[bad]]
    v[bad] = gm[bad]
    return v, fixed


def stats(r: pd.Series, per_year: int = 252, label: str = "") -> dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    yrs = len(r) / per_year
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    dd = (eq / eq.cummax() - 1)
    o = r[r.index >= SPLIT]
    oeq = (1 + o).cumprod()
    return {
        "series": label, "yrs": round(yrs, 1),
        "CAGR%": round(100 * cagr, 2),
        "OOS_CAGR%": round(100 * (oeq.iloc[-1] ** (per_year / len(o)) - 1), 2) if len(o) > 250 else np.nan,
        "vol%": round(100 * r.std(ddof=1) * np.sqrt(per_year), 1),
        "sharpe": round(r.mean() / r.std(ddof=1) * np.sqrt(per_year), 2),
        "OOS_sharpe": round(o.mean() / o.std(ddof=1) * np.sqrt(per_year), 2) if len(o) > 250 else np.nan,
        "maxDD%": round(100 * dd.min(), 1),
        "worst_day%": round(100 * r.min(), 1),
        "skew": round(float(r.skew()), 2),
        "kurt": round(float(r.kurtosis()), 1),
        "yrs_to_100x": round(np.log(100) / np.log(1 + cagr), 1) if cagr > 0 else np.nan,
    }


def main() -> int:
    put, spy, vix = load("^PUT"), load("SPY"), load("^VIX")
    vix9d, vix3m = load("^VIX9D"), load("^VIX3M")
    if put is None or spy is None:
        print("missing ^PUT or SPY")
        return 1
    print(f"^PUT {len(put)} rows {put.index[0]:%Y-%m-%d}..{put.index[-1]:%Y-%m-%d}")
    put, fixed = repair(put)
    print(f"  repaired {len(fixed)} isolated bad print(s): " +
          ", ".join(f"{d} {was}->{now}" for d, was, now in fixed) if fixed else "  no bad prints")

    # ---------------- V1: does the premium exist? --------------------------------
    # Compare today's VIX to the realised vol that ACTUALLY FOLLOWED over the next 21 sessions.
    # Comparing VIX to *trailing* realised vol is the common error: it measures nothing, because the
    # option seller is paid for future variance, not past variance.
    rs = spy.pct_change()
    fwd_rv = 100 * rs.shift(-1).rolling(21).std().shift(-20) * np.sqrt(252)
    v = pd.concat({"vix": vix, "fwd_rv": fwd_rv}, axis=1).dropna()
    v["vrp"] = v.vix - v.fwd_rv
    print("\n" + "=" * 104)
    print("V1  DOES THE PREMIUM EXIST?   VIX(t) vs realised vol over the FOLLOWING 21 sessions")
    print("=" * 104)
    print(f"  mean VIX {v.vix.mean():.1f}   mean subsequent realised {v.fwd_rv.mean():.1f}   "
          f"mean premium {v.vrp.mean():+.2f} vol points")
    print(f"  premium positive on {100*(v.vrp>0).mean():.1f}% of days   "
          f"t-stat (HAC, 21 lags) {sm.OLS(v.vrp.to_numpy(), np.ones(len(v))).fit(cov_type='HAC', cov_kwds={'maxlags':21}).tvalues[0]:.1f}")
    print(f"  by VIX regime (the premium is not uniform — this is what makes timing tempting):")
    for lo, hi in ((0, 15), (15, 20), (20, 30), (30, 100)):
        m = v[(v.vix >= lo) & (v.vix < hi)]
        if len(m) > 50:
            print(f"    VIX {lo:>2}-{hi:<3} n={len(m):5}  premium {m.vrp.mean():+6.2f}  "
                  f"positive {100*(m.vrp>0).mean():5.1f}%")

    # ---------------- V2: does harvesting it beat SPY? ---------------------------
    rp, rspy = put.pct_change(), spy.pct_change()
    both = pd.concat({"put": rp, "spy": rspy}, axis=1).dropna()
    rows = [stats(both.spy, label="BUY & HOLD SPY"),
            stats(both.put, label="^PUT cash-secured putwrite")]

    # ---------------- V3: does timing it help? ----------------------------------
    # Every signal is shifted one day: the decision to hold tomorrow uses only data through today.
    sig_defs = {}
    vixa = vix.reindex(both.index).ffill()
    trail_rv = (100 * rs.rolling(21).std() * np.sqrt(252)).reindex(both.index).ffill()
    sig_defs["V3a VIX<30 gate"] = (vixa < 30)
    sig_defs["V3b wide premium (VIX-trailRV>2)"] = ((vixa - trail_rv) > 2.0)
    sig_defs["V3c both"] = (vixa < 30) & ((vixa - trail_rv) > 2.0)
    if vix9d is not None:
        n9 = vix9d.reindex(both.index).ffill()
        sig_defs["V3d term structure calm (VIX9D<VIX)"] = (n9 < vixa)
    if vix3m is not None:
        n3 = vix3m.reindex(both.index).ffill()
        sig_defs["V3e contango (VIX<VIX3M)"] = (vixa < n3)

    for name, sig in sig_defs.items():
        s = sig.shift(1).fillna(False).astype(float)
        sw = s.diff().abs().fillna(0)
        r = (both.put * s) - sw * SWITCH_BP / 1e4
        st = stats(r.loc[s.index[s.notna()]], label=name)
        st["in_mkt%"] = round(100 * s.mean(), 1)
        rows.append(st)

    t = pd.DataFrame(rows)
    cols = ["series", "yrs", "CAGR%", "OOS_CAGR%", "vol%", "sharpe", "OOS_sharpe",
            "maxDD%", "worst_day%", "skew", "kurt", "yrs_to_100x"]
    print("\n" + "=" * 104)
    print(f"V2/V3  HARVESTING AND TIMING IT   (OOS = {SPLIT:%Y}+, switch cost {SWITCH_BP}bp/side)")
    print("=" * 104)
    print(t[cols].to_string(index=False, na_rep="-"))
    if "in_mkt%" in t:
        print("\n  time in market for the timed variants: " + ", ".join(
            f"{r.series.split()[0]} {r['in_mkt%']:.0f}%" for _, r in t.iterrows()
            if pd.notna(r.get("in_mkt%"))))

    # ---------------- V4: the tail --------------------------------------------
    print("\n" + "=" * 104)
    print("V4  THE TAIL — short vol's whole risk is here, so compare it to SPY's own tail")
    print("=" * 104)
    for nm, r in (("SPY", both.spy), ("^PUT", both.put)):
        print(f"  {nm:5} worst 5 days: " + ", ".join(f"{100*x:.1f}%" for x in r.nsmallest(5)))
    crisis = {"2008 GFC": ("2008-09-01", "2009-03-31"),
              "2018 Volmageddon": ("2018-01-26", "2018-02-28"),
              "2020 COVID": ("2020-02-19", "2020-03-23"),
              "2022 bear": ("2022-01-03", "2022-10-14")}
    print(f"\n  {'episode':20} {'SPY':>9} {'^PUT':>9}")
    for nm, (a, b) in crisis.items():
        w = both.loc[a:b]
        if len(w) < 5:
            continue
        print(f"  {nm:20} {100*((1+w.spy).prod()-1):8.1f}% {100*((1+w.put).prod()-1):8.1f}%")

    t.to_csv("vrp_backtest_results.csv", index=False)
    print("\nsaved -> vrp_backtest_results.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
