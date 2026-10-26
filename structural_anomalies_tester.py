"""
structural_anomalies_tester.py — seven pre-registered hypotheses, one honest scoreboard.

WHAT THIS IS FOR. `fed_drift_suite` found one real but small edge (FOMC eve, +14.8bp, 8 trades/yr,
~1.2%/yr). The question here is whether there are more *special days* worth trading. The hypotheses
were named in writing before any of them was run, and every one is reported below whether it passed
or failed, because a battery that only prints its winners is not evidence of anything.

THE MULTIPLE-TESTING PROBLEM IS THE WHOLE PROBLEM. Test enough ideas at p<0.05 and one passes on
luck alone. So:
  * The hypothesis list is fixed (7 hypotheses x up to 3 assets = 17 tests).
  * The significance bar is Bonferroni-adjusted: 0.05/17 = 0.0029, not 0.05.
  * Full-sample significance is necessary but not sufficient. A survivor must ALSO hold up on
    1996-2017 alone (or 2000-2013 where data starts later) and in walk-forward.
  * Anything that needs a conditional filter to work is flagged, because a filter is a free
    parameter and free parameters are how noise gets dressed up as a strategy.

WHY DAILY CLOSE-TO-CLOSE USES ADJUSTED PRICES HERE. The overnight research deliberately avoided
auto-adjusted prices: a dividend paid on the ex-date lands entirely in the close-to-open gap and
inflates "overnight drift" by 1.3-1.8%/yr, which is larger than the effect being measured. These
hypotheses hold multi-day close-to-close positions instead, where the dividend belongs inside the
holding period, so adjusted prices are the correct instrument rather than a source of bias.

FRICTION. 2.0bp round trip: 0.5bp/side auction slippage, SEC Section 31 on the sell (~0.28bp),
FINRA TAF, and commission. That is the same model the FOMC work used, where it reproduced against
real fills. Every result is also reported under a +2bp cost shock, because a strategy that only
works at my assumed cost is not a strategy.
"""
from __future__ import annotations

import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")

SUITE = Path(__file__).resolve().parent / "fed_drift_suite"
if str(SUITE) not in sys.path:
    sys.path.insert(0, str(SUITE))
from commercial.market_days import holidays, is_trading_day   # noqa: E402

CACHE = Path(__file__).resolve().parent / ".anomaly_cache"
CACHE.mkdir(exist_ok=True)

START = "1999-01-01"
COST_BP = 2.0              # round trip
COST_SHOCK_BP = 2.0        # additional, for the sensitivity column
N_PERM = 5000
SEED = 20260924
N_TESTS = 17               # pre-registered; sets the Bonferroni bar
BAR = 0.05 / N_TESTS

rng = np.random.default_rng(SEED)


# ------------------------------------------------------------------ data

def fetch(ticker: str) -> pd.DataFrame:
    """Adjusted daily bars, cached to disk so a re-run does not re-download."""
    f = CACHE / f"{ticker.replace('^', '_')}.csv"
    if f.exists():
        return pd.read_csv(f, index_col=0, parse_dates=True)
    import yfinance as yf
    df = yf.Ticker(ticker).history(start=START, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"no data for {ticker}")
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.to_csv(f)
    return df


# ------------------------------------------------------------------ statistics

def hac_t(x: np.ndarray, lags: int = 5) -> float:
    """Newey-West t against zero. Overlapping/multi-day holds need lags >= hold length."""
    if len(x) < 10:
        return np.nan
    return float(sm.OLS(x, np.ones(len(x))).fit(
        cov_type="HAC", cov_kwds={"maxlags": max(lags, 1)}).tvalues[0])


def perm_p(event: np.ndarray, pool: np.ndarray) -> float:
    """One-sided: how often does a same-size random draw from non-event windows beat the event mean?

    The pool deliberately EXCLUDES event windows, so this asks "is this calendar special" rather
    than "is the market up", which is the question a t-test against zero cannot answer.
    """
    if len(event) < 5 or len(pool) < len(event):
        return np.nan
    obs = event.mean()
    draws = rng.choice(pool, size=(N_PERM, len(event)), replace=True).mean(axis=1)
    return float((draws >= obs).mean())


def maxdd(per_trade: np.ndarray) -> float:
    eq = np.cumprod(1 + per_trade / 1e4)
    return float(100 * (eq / np.maximum.accumulate(eq) - 1).min())


def sharpe_ann(per_trade: np.ndarray, per_year: float) -> float:
    """Sparse-book identity: annualized Sharpe = per-trade Sharpe x sqrt(trades per year).

    This is why trade count matters more than edge size. 8 trades/yr caps annualized Sharpe near
    1.2 no matter how good each trade is.
    """
    s = per_trade.std(ddof=1)
    return float(per_trade.mean() / s * np.sqrt(per_year)) if s > 0 else np.nan


def walk_forward(dates: pd.Series, r: np.ndarray, is_yrs: int = 5, oos_yrs: int = 2) -> float:
    """WFE = mean OOS expectancy / mean IS expectancy over rolling 5y-in / 2y-out blocks.

    ~1.0 means the edge did not depend on the window it was measured in. Well under 1 means it did.
    """
    d = pd.to_datetime(dates)
    y0, y1 = d.dt.year.min(), d.dt.year.max()
    is_m, oos_m = [], []
    y = y0
    while y + is_yrs + oos_yrs - 1 <= y1:
        i = r[(d.dt.year >= y) & (d.dt.year < y + is_yrs)]
        o = r[(d.dt.year >= y + is_yrs) & (d.dt.year < y + is_yrs + oos_yrs)]
        if len(i) >= 8 and len(o) >= 4:
            is_m.append(i.mean())
            oos_m.append(o.mean())
        y += oos_yrs
    if not is_m or np.mean(is_m) <= 0:
        return np.nan
    return float(np.mean(oos_m) / np.mean(is_m))


RESULTS: list[dict] = []


def record(name: str, asset: str, dates, gross_bp, pool_bp, per_year: float,
           hold_days: int, note: str = "") -> None:
    g = np.asarray(gross_bp, dtype=float)
    net = g - COST_BP
    pool = np.asarray(pool_bp, dtype=float) - COST_BP
    dts = pd.Series(pd.to_datetime(list(dates)))
    split = pd.Timestamp("2018-01-01")
    early = net[dts.values < split.to_datetime64()]
    RESULTS.append({
        "hypothesis": name, "asset": asset, "n": len(net), "n_per_yr": round(per_year, 1),
        "hold_d": hold_days,
        "gross_bp": round(g.mean(), 2), "NET_bp": round(net.mean(), 2),
        "early_bp": round(early.mean(), 2) if len(early) >= 10 else np.nan,
        "shocked_bp": round(net.mean() - COST_SHOCK_BP, 2),
        "win%": round(100 * (net > 0).mean(), 1),
        "sharpe": round(sharpe_ann(net, per_year), 2),
        "maxDD%": round(maxdd(net), 1),
        "ret_yr%": round(per_year * net.mean() / 100, 2),
        "t_HAC": round(hac_t(net, lags=max(5, hold_days)), 2),
        "perm_p": round(perm_p(net, pool), 4) if len(pool) else np.nan,
        "WFE": round(walk_forward(dts, net), 2),
        "note": note,
    })


# ------------------------------------------------------------------ window helpers

def fwd_bp(px: pd.Series, i: int, k: int) -> float | None:
    """Close-to-close return in bp from bar i to bar i+k. Causal by construction."""
    if i + k >= len(px):
        return None
    return float(1e4 * (px.iloc[i + k] / px.iloc[i] - 1))


def all_k_day_windows(px: pd.Series, k: int, exclude: set[int]) -> np.ndarray:
    """Every k-day close-to-close window not starting on an excluded bar — the permutation pool."""
    return np.array([1e4 * (px.iloc[i + k] / px.iloc[i] - 1)
                     for i in range(len(px) - k) if i not in exclude])


def nth_weekday(y: int, m: int, weekday: int, n: int) -> date:
    d = date(y, m, 1)
    shift = (weekday - d.weekday()) % 7
    return date(y, m, 1 + shift + 7 * (n - 1))


# ------------------------------------------------------------------ H1  turn of the month

def h1_totm(assets=("SPY", "QQQ", "IWM")) -> None:
    """Enter MOC on the last trading day of the month, exit MOC on the 3rd trading day of the next.

    Mechanism if real: 401k payroll contributions, pension distributions and DRIP reinvestment are
    price-insensitive flows that arrive on a payroll calendar, not when stocks are cheap.
    """
    for a in assets:
        px = fetch(a)["Close"]
        idx = px.index
        month = pd.Series(idx.to_period("M"), index=range(len(idx)))
        last_of_month = [i for i in range(len(idx) - 1) if month[i] != month[i + 1]]

        ev, ds, used = [], [], set()
        for i in last_of_month:
            r = fwd_bp(px, i, 4)                       # T-0 close -> 3rd trading day close
            if r is None:
                continue
            ev.append(r); ds.append(idx[i]); used.update(range(i, i + 5))
        pool = all_k_day_windows(px, 4, used)
        record("H1 turn-of-month", a, ds, ev, pool, 12.0, 4)

        # The benchmark that matters: the rest of the month, same asset, same cost model.
        rest, rds = [], []
        for i in range(len(idx) - 4):
            if i not in used:
                r = fwd_bp(px, i, 4)
                if r is not None:
                    rest.append(r); rds.append(idx[i])
        record("  └ rest-of-month (bench)", a, rds, rest, [], 12.0, 4, "overlapping, context only")


# ------------------------------------------------------------------ H2  post-OpEx unpinning

def h2_opex(assets=("SPY", "QQQ")) -> None:
    """Close of OpEx Friday -> close of the following Wednesday (3 trading days).

    Tested twice on purpose: unconditional, and filtered to the lowest quintile of 5-day realized
    vol into expiry. The filter is the interesting version but it is also a fitted parameter and it
    cuts the sample by 80%, so both are shown side by side.
    """
    for a in assets:
        px = fetch(a)["Close"]
        idx = px.index
        rv = px.pct_change().rolling(5).std().shift(1)      # shift(1): known before the Friday close

        ev, ds, rvs, used = [], [], [], set()
        for y in range(idx[0].year, idx[-1].year + 1):
            for m in range(1, 13):
                f = pd.Timestamp(nth_weekday(y, m, 4, 3))   # 3rd Friday
                while f.date() in holidays(f.year):         # Good Friday etc -> Thursday
                    f -= pd.Timedelta(days=1)
                loc = idx.searchsorted(f)
                if loc >= len(idx) or idx[loc] != f:
                    continue
                r = fwd_bp(px, loc, 3)
                if r is None or np.isnan(rv.iloc[loc]):
                    continue
                ev.append(r); ds.append(f); rvs.append(rv.iloc[loc]); used.update(range(loc, loc + 4))

        pool = all_k_day_windows(px, 3, used)
        n_yr = len(ev) / ((idx[-1] - idx[0]).days / 365.25)
        record("H2 post-OpEx (all)", a, ds, ev, pool, n_yr, 3)

        ev_a, rv_a = np.array(ev), np.array(rvs)
        cut = np.quantile(rv_a, 0.20)
        m = rv_a <= cut
        record("  └ H2 low-vol filter", a, np.array(ds)[m], ev_a[m], pool, n_yr * 0.2, 3,
               "filter = 1 fitted parameter")


# ------------------------------------------------------------------ H3  short-vol carry overlay

def h3_shortvol() -> None:
    """SPY + 10% cash-collateralized SVXY while VIX term structure is calm; all cash on a spike.

    READ THE NOTE. This is the one hypothesis where the backtest is structurally untrustworthy and
    no amount of statistics fixes it:
      * XIV, the most popular vehicle for this exact trade, was TERMINATED on 2018-02-05 after
        losing ~96% in one session. It is absent from every price series you can download, so any
        short-vol backtest is survivorship-biased by construction.
      * SVXY survived only because it existed; ProShares then cut its mandate from -1.0x to -0.5x
        VIX futures in Feb 2018. Pre-2018 and post-2018 SVXY are different instruments, so a
        continuous series splices two products and understates the risk of the earlier one.
    Short vol has genuine positive carry and a left tail that ends the account. The carry shows up
    in the mean; the tail shows up once.
    """
    try:
        spy, svxy, vix = fetch("SPY")["Close"], fetch("SVXY")["Close"], fetch("^VIX")["Close"]
    except RuntimeError as e:
        RESULTS.append({"hypothesis": "H3 short-vol overlay", "asset": "SVXY", "n": 0,
                        "note": f"UNTESTABLE: {e}"})
        return

    df = pd.concat({"spy": spy, "svxy": svxy, "vix": vix}, axis=1).dropna()
    # All signals shifted: the Friday decision uses data through Thursday's close.
    sig = ((df.vix < df.vix.rolling(50).mean()) & (df.vix < df.vix.rolling(200).mean())).shift(1)
    rs, rv = df.spy.pct_change(), df.svxy.pct_change()

    fri = df.index[df.index.weekday == 4]
    ev, ds, bench = [], [], []
    for j in range(len(fri) - 1):
        a, b = fri[j], fri[j + 1]
        w = df.loc[a:b]
        if len(w) < 3 or pd.isna(sig.loc[a]):
            continue
        spy_w = float(1e4 * (df.spy.loc[b] / df.spy.loc[a] - 1))
        vol_w = float(1e4 * (df.svxy.loc[b] / df.svxy.loc[a] - 1))
        ev.append(spy_w + 0.10 * vol_w if sig.loc[a] else 0.0)
        bench.append(spy_w)
        ds.append(a)
    n_yr = 52.0
    record("H3 short-vol overlay", "SPY+10%SVXY", ds, ev, [], n_yr, 5,
           "SURVIVORSHIP: XIV delisted -96%; SVXY mandate changed 2018")
    record("  └ SPY weekly (bench)", "SPY", ds, bench, [], n_yr, 5, "buy & hold, same weeks")


# ------------------------------------------------------------------ B1  weekend

def b1_weekend(assets=("SPY", "QQQ", "IWM")) -> None:
    """Friday close -> Monday open. My own best guess going in: most trades, most likely arbitraged.

    Mechanism if real: a risk premium for holding 65 unhedgeable hours, since no one can trade out
    of a weekend headline.
    """
    for a in assets:
        d = fetch(a)
        c, o, idx = d["Close"], d["Open"], d.index
        ev, ds, pool = [], [], []
        for i in range(len(idx) - 1):
            r = 1e4 * (o.iloc[i + 1] / c.iloc[i] - 1)
            gap_days = (idx[i + 1] - idx[i]).days
            if gap_days >= 3:
                ev.append(r); ds.append(idx[i])
            elif gap_days == 1:
                pool.append(r)
        record("B1 weekend gap", a, ds, ev, pool, len(ev) / ((idx[-1] - idx[0]).days / 365.25), 1)


# ------------------------------------------------------------------ B2  pre-holiday

def b2_preholiday(assets=("SPY", "QQQ", "IWM")) -> None:
    """The full session immediately before a market holiday. ~9 events a year, which is the problem."""
    for a in assets:
        px = fetch(a)["Close"]
        idx = px.index
        ev, ds, used = [], [], set()
        for i in range(len(idx) - 1):
            nxt = idx[i] + pd.Timedelta(days=1)
            while nxt.weekday() >= 5:
                nxt += pd.Timedelta(days=1)
            if not is_trading_day(nxt.date()) and nxt.date() in holidays(nxt.year):
                r = fwd_bp(px, i, 1)
                if r is not None:
                    ev.append(r); ds.append(idx[i]); used.add(i)
        pool = all_k_day_windows(px, 1, used)
        record("B2 pre-holiday", a, ds, ev, pool, len(ev) / ((idx[-1] - idx[0]).days / 365.25), 1)


# ------------------------------------------------------------------ B4  FOMC decision day session

def b4_decision_day(assets=("SPY", "QQQ", "IWM")) -> None:
    """Open -> close on the decision day itself (statement 14:00 ET). A free add-on if it works."""
    from config import Config
    from core.calendar import FomcCalendar
    cal = FomcCalendar(Config().cache_path, auto_fetch=False)
    decisions = {pd.Timestamp(x) for x in cal.dates()}
    for a in assets:
        d = fetch(a)
        c, o, idx = d["Close"], d["Open"], d.index
        ev, ds, pool = [], [], []
        for i in range(len(idx)):
            r = 1e4 * (c.iloc[i] / o.iloc[i] - 1)
            (ev if idx[i] in decisions else pool).append(r)
            if idx[i] in decisions:
                ds.append(idx[i])
        record("B4 FOMC decision day", a, ds, ev, pool,
               len(ev) / ((idx[-1] - idx[0]).days / 365.25), 1)


# ------------------------------------------------------------------ report

def main() -> int:
    h1_totm(); h2_opex(); h3_shortvol(); b1_weekend(); b2_preholiday(); b4_decision_day()

    df = pd.DataFrame(RESULTS)
    cols = ["hypothesis", "asset", "n", "n_per_yr", "NET_bp", "early_bp", "shocked_bp",
            "win%", "sharpe", "maxDD%", "ret_yr%", "t_HAC", "perm_p", "WFE"]
    print("\n" + "=" * 118)
    print(f"PRE-REGISTERED BATTERY — {N_TESTS} tests, Bonferroni bar p < {BAR:.4f} "
          f"(not 0.05), friction {COST_BP}bp round trip")
    print("=" * 118)
    print(df[cols].to_string(index=False, na_rep="—"))

    print(f"\nNET_bp = per-trade expectancy after {COST_BP}bp friction")
    print(f"early_bp = same, 1999-2017 only (the closest thing to out-of-sample here)")
    print(f"shocked_bp = expectancy if my cost model is {COST_SHOCK_BP}bp too optimistic")
    print("WFE = out-of-sample expectancy / in-sample, rolling 5y-in 2y-out. ~1.0 is stable.")

    print("\n" + "=" * 118)
    print("VERDICT")
    print("=" * 118)
    real = df[~df.hypothesis.str.startswith("  ")]
    for _, r in real.iterrows():
        if not r.get("n"):
            print(f"  REJECT   {r.hypothesis:24s} {r.asset:12s} {r.get('note','')}")
            continue
        p, t = r.get("perm_p"), r.get("t_HAC")
        fails = []
        if pd.isna(p) or p > BAR:
            fails.append(f"perm p {p} > {BAR:.4f}")
        if pd.notna(t) and abs(t) < 3.0:
            fails.append(f"|t| {t} < 3.0")
        if pd.notna(r.early_bp) and r.early_bp <= 0:
            fails.append(f"pre-2018 negative ({r.early_bp}bp)")
        if r.shocked_bp <= 0:
            fails.append("dies under cost shock")
        if r.NET_bp < 15:
            fails.append(f"expectancy {r.NET_bp}bp < 15bp target")
        tag = "SURVIVES" if not fails else "REJECT  "
        print(f"  {tag} {r.hypothesis:24s} {r.asset:12s} " +
              ("; ".join(fails) if fails else f"{r.NET_bp}bp, {r.n} trades, p={p}"))
        if r.get("note"):
            print(f"           ^ {r.note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
