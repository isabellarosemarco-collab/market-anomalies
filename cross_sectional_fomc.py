"""
cross_sectional_fomc.py — on the 8 nights we already know are special, which assets drift most?

THE DESIGN PROBLEM, STATED FIRST. Searching ~40 assets for the best FOMC-eve performer is exactly
the procedure that manufactures fake edges: with 40 candidates, the best in-sample result is mostly
the luckiest, not the strongest. Three guards, fixed before running:

  1. SELECT IN SAMPLE, SCORE OUT OF SAMPLE. Assets are ranked on 1996-2017 only. The winner is then
     evaluated on 2018+ alone, with the IS number shown next to it. The gap between those two
     columns IS the selection bias, measured rather than argued about.
  2. LIQUIDITY-TIERED COSTS. `build_dataset.py` uses 0.5bp/side, justified for a >$40bn/day name.
     Applying that to a thin sector ETF would hand the search a fake winner whose real spread eats
     the edge. Slippage here scales with median dollar volume, so illiquid names must clear a higher
     bar to win — which is what happens in reality.
  3. THE PORTFOLIO QUESTION IS THE REAL ONE. "Which single asset is best" is badly powered on 69
     out-of-sample nights. "Does holding several assets on the same night beat holding one" is much
     better powered, because it is a variance question, not a mean question. If FOMC-eve returns are
     less than perfectly correlated, a basket keeps the mean and cuts the standard deviation, which
     raises Sharpe without requiring any new anomaly. That is the outcome worth wanting.

INTERNATIONAL ETFs ARE FLAGGED, NOT TRUSTED. For EFA/EEM/EWJ/FXI the US close-to-open window is
when their underlying markets are actually trading, so the "overnight gap" is a real foreign session,
not US pre-positioning. They are priced here for completeness and excluded from selection.

RETURN CONSTRUCTION matches core/build_dataset.py exactly: raw prices (auto_adjust=False) with an
explicit dividend term, because auto-adjustment silently credits the entire dividend to the
close-to-open gap and would fabricate 1.3-1.8%/yr of drift.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")

SUITE = Path(__file__).resolve().parent / "fed_drift_suite"
if str(SUITE) not in sys.path:
    sys.path.insert(0, str(SUITE))
from config import Config                    # noqa: E402
from core.calendar import FomcCalendar       # noqa: E402
from commercial.market_days import is_trading_day   # noqa: E402

CACHE = Path(__file__).resolve().parent / ".xsec_cache"
CACHE.mkdir(exist_ok=True)

SEC_FEE_FRAC = 27.80 / 1e6
FINRA_TAF_PER_SHARE = 0.000166
SPLIT = pd.Timestamp("2018-01-01")
MIN_IS, MIN_OOS = 40, 30
N_PERM, SEED = 5000, 20260924
rng = np.random.default_rng(SEED)

# Long-only, unlevered, reasonably liquid. No leveraged or inverse products: daily rebalancing
# decay and borrow costs make them a different instrument, not a higher-beta version of this one.
UNIVERSE = [
    "SPY", "IVV", "VTI", "QQQ", "IWM", "DIA", "MDY", "IJH", "IJR",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU", "XLY", "XLB",
    "IWF", "IWD", "IWN", "IWO", "IWB", "OEF",
    "SMH", "SOXX", "IBB", "XBI", "IYW", "IYF", "IYR", "VNQ",
    "TLT", "IEF", "LQD", "HYG", "GLD", "SLV",
    "EFA", "EEM", "EWJ", "FXI",
]
INTERNATIONAL = {"EFA", "EEM", "EWJ", "FXI"}


def slip_bp_per_side(dollar_vol: float) -> float:
    """Half-spread + impact by liquidity tier. The search must not be able to win by being thin."""
    if dollar_vol > 5e9:
        return 0.5
    if dollar_vol > 1e9:
        return 1.0
    if dollar_vol > 2e8:
        return 2.0
    if dollar_vol > 5e7:
        return 4.0
    return 8.0


def load(symbol: str) -> pd.DataFrame | None:
    f = CACHE / f"{symbol}.csv"
    if f.exists():
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        return df if len(df) > 500 else None
    try:
        import yfinance as yf
        df = yf.Ticker(symbol).history(start="1995-01-01", interval="1d",
                                       auto_adjust=False, actions=True)
    except Exception:
        return None
    if df.empty or len(df) < 500:
        return None
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    if "dividends" not in df:
        df["dividends"] = 0.0
    df = df[["open", "high", "low", "close", "volume", "dividends"]].dropna(
        subset=["open", "close"])
    df.to_csv(f)
    return df


def eve_returns(symbol: str, eves: set[pd.Timestamp]) -> pd.DataFrame | None:
    """Net close-to-open bp labelled by the ENTRY date, dividend explicit, liquidity-tiered cost.

    THE LABEL IS THE WHOLE TRICK. The trade is close of the eve E -> open of the decision day E+1,
    so the return lives on the row for E+1 (its `open`, against E's `close`). Labelling that row with
    its own date and then asking "is this date an FOMC eve" silently tests the night BEFORE the eve
    — a different, worthless night. The label here is the entry date, so `is_eve` means what it says.

    Liquidity is measured on a TRAILING 252-day window rather than over all history: IYW and SOXX
    were genuinely thin in 2003 and are not now, and charging today's trade for 2003's spread would
    penalise names that grew rather than names that are expensive to trade.
    """
    df = load(symbol)
    if df is None:
        return None

    o, c, d = df.open, df.close, df.dividends
    prev_c = c.shift(1)
    # Entry is the prior close; the holder is on the record date and receives the dividend.
    gross = (o + d) / prev_c - 1.0

    dvol = (df.close * df.volume).rolling(252, min_periods=60).median().shift(1)
    slip = dvol.map(lambda v: slip_bp_per_side(v) if pd.notna(v) else np.nan)
    cost = 2.0 * (slip / 1e4) + SEC_FEE_FRAC + FINRA_TAF_PER_SHARE / prev_c

    out = pd.DataFrame({
        "date": df.index.to_series().shift(1).to_numpy(),      # <- the ENTRY (eve) date
        "exit_date": df.index,
        "gross_bp": 1e4 * gross.to_numpy(),
        "cost_bp": 1e4 * cost.to_numpy(),
        "slip_bp": slip.to_numpy(),
        "dollar_vol": dvol.to_numpy(),
    }).dropna(subset=["date", "gross_bp", "cost_bp"])
    out["net_bp"] = out.gross_bp - out.cost_bp
    out["is_eve"] = out.date.isin(eves)
    out["symbol"] = symbol
    return out


def hac_t(x: np.ndarray) -> float:
    if len(x) < 10:
        return np.nan
    return float(sm.OLS(x, np.ones(len(x))).fit(
        cov_type="HAC", cov_kwds={"maxlags": 5}).tvalues[0])


def perm_p(ev: np.ndarray, pool: np.ndarray) -> float:
    if len(ev) < 5 or len(pool) < len(ev):
        return np.nan
    draws = rng.choice(pool, size=(N_PERM, len(ev)), replace=True).mean(axis=1)
    return float((draws >= ev.mean()).mean())


def sharpe(x: np.ndarray, per_year: float = 8.0) -> float:
    s = x.std(ddof=1)
    return float(x.mean() / s * np.sqrt(per_year)) if s > 0 else np.nan


def maxdd(x: np.ndarray) -> float:
    eq = np.cumprod(1 + x / 1e4)
    return float(100 * (eq / np.maximum.accumulate(eq) - 1).min())


def main() -> int:
    cal = FomcCalendar(Config().cache_path, auto_fetch=False)
    eves = set()
    for dec in cal.dates():
        e = cal.eve_of(dec, is_trading_day)
        if e:
            eves.add(pd.Timestamp(e))
    print(f"FOMC eves on the calendar: {len(eves)}  ({min(eves).date()} .. {max(eves).date()})")

    frames = []
    for s in UNIVERSE:
        r = eve_returns(s, eves)
        if r is not None:
            frames.append(r)
        else:
            print(f"  skipped {s} (no usable data)")
    allr = pd.concat(frames, ignore_index=True)

    # ---- VALIDATION GATE -------------------------------------------------------
    # Reproduce the committed trade file before trusting anything new. This is what caught the
    # entry-date labelling bug: the pipeline looked fine and quietly measured the wrong night.
    # A ~1bp shortfall is expected and correct — the tiered cost model charges QQQ 1.0bp/side where
    # build_dataset.py charges 0.5bp/side.
    ref = pd.read_csv(SUITE / "core" / "data" / "fomc_eve_returns.csv", parse_dates=["eve_date"])
    print("\n  validation vs core/data/fomc_eve_returns.csv (pre-2018, expect ~1bp lower here):")
    ok = True
    for sym in ("SPY", "QQQ", "IWM"):
        want = ref[(ref.symbol == sym) & (ref.eve_date < SPLIT)].net_bp.mean()
        g = allr[(allr.symbol == sym) & allr.is_eve]
        got = g[g.date < SPLIT].net_bp.mean()
        flag = "ok" if abs(got - want) < 2.5 else "MISMATCH"
        if flag != "ok":
            ok = False
        print(f"    {sym}: suite {want:+6.2f}bp   here {got:+6.2f}bp   {flag}")
    if not ok:
        print("\n  ABORTING: the pipeline does not reproduce known-good numbers.")
        return 1

    rows = []
    for s, g in allr.groupby("symbol"):
        ev = g[g.is_eve]
        pool = g[~g.is_eve].net_bp.to_numpy()
        is_ = ev[ev.date < SPLIT].net_bp.to_numpy()
        oos = ev[ev.date >= SPLIT].net_bp.to_numpy()
        if len(is_) < MIN_IS or len(oos) < MIN_OOS:
            continue
        rows.append({
            "symbol": s, "intl": s in INTERNATIONAL,
            "slip": round(float(ev.slip_bp.median()), 1),
            "$vol_M": round(float(ev.dollar_vol.median()) / 1e6),
            "n_is": len(is_), "IS_bp": round(is_.mean(), 2),
            "n_oos": len(oos), "OOS_bp": round(oos.mean(), 2),
            "all_bp": round(ev.net_bp.mean(), 2),
            "sharpe": round(sharpe(ev.net_bp.to_numpy()), 2),
            "maxDD%": round(maxdd(ev.net_bp.to_numpy()), 1),
            "t_HAC": round(hac_t(ev.net_bp.to_numpy()), 2),
            "perm_p": round(perm_p(ev.net_bp.to_numpy(), pool), 4),
            "base_bp": round(pool.mean(), 2),
        })
    t = pd.DataFrame(rows).sort_values("IS_bp", ascending=False)

    print("\n" + "=" * 112)
    print("RANKED BY IN-SAMPLE (pre-2018) ONLY. OOS_bp is the honest number — it had no say in the ranking.")
    print("=" * 112)
    print(t.to_string(index=False))

    dom = t[~t.intl]
    print("\n" + "=" * 112)
    print("DOES SELECTION SURVIVE?")
    print("=" * 112)
    bar = 0.05 / len(dom)
    print(f"  {len(dom)} domestic candidates -> Bonferroni bar p < {bar:.5f}")
    top = dom.iloc[0]
    qqq = dom[dom.symbol == "QQQ"].iloc[0]
    print(f"  best in-sample:      {top.symbol}  IS {top.IS_bp:+.2f}bp  ->  OOS {top.OOS_bp:+.2f}bp"
          f"   (selection bias: {top.OOS_bp - top.IS_bp:+.2f}bp)")
    print(f"  incumbent QQQ:       IS {qqq.IS_bp:+.2f}bp  ->  OOS {qqq.OOS_bp:+.2f}bp")
    print(f"  mean OOS-minus-IS across all domestic names: "
          f"{(dom.OOS_bp - dom.IS_bp).mean():+.2f}bp")
    passers = dom[(dom.perm_p < bar)]
    print(f"  names clearing the Bonferroni bar full-sample: "
          f"{', '.join(passers.symbol) if len(passers) else 'NONE'}")

    # ---- the portfolio question -------------------------------------------------
    print("\n" + "=" * 112)
    print("THE PORTFOLIO TEST — same nights, several assets, equal weight")
    print("=" * 112)
    wide = (allr[allr.is_eve]
            .pivot_table(index="date", columns="symbol", values="net_bp"))
    liquid = [s for s in dom.symbol if s in wide.columns]
    core = [s for s in ("SPY", "QQQ", "IWM") if s in wide.columns]

    corr = wide[core].corr().values
    print(f"  correlation among SPY/QQQ/IWM on FOMC eves: "
          f"{corr[np.triu_indices(len(core), 1)].mean():.2f}  "
          f"(1.00 would mean a basket buys you nothing)")

    def score(cols, label, restrict_oos=False):
        w = wide[cols].dropna()
        if restrict_oos:
            w = w[w.index >= SPLIT]
        if len(w) < 20:
            return
        r = w.mean(axis=1).to_numpy()
        print(f"  {label:34s} n={len(r):3d}  {r.mean():+6.2f}bp  "
              f"sharpe {sharpe(r):5.2f}  maxDD {maxdd(r):6.1f}%  t {hac_t(r):5.2f}")

    print("\n  full sample:")
    score(["QQQ"], "QQQ alone (incumbent)")
    score(["IWM"], "IWM alone (our pick)")
    score(core, "SPY+QQQ+IWM equal weight")
    is_top3 = list(dom.nlargest(3, "IS_bp").symbol)
    is_top5 = list(dom.nlargest(5, "IS_bp").symbol)
    score(is_top3, f"top-3 by IS: {'+'.join(is_top3)}")
    score(is_top5, f"top-5 by IS: {'+'.join(is_top5)}")
    score(liquid, f"all {len(liquid)} domestic, equal weight")

    print("\n  2018+ only (out of sample for the IS-selected baskets):")
    score(["QQQ"], "QQQ alone", True)
    score(["IWM"], "IWM alone", True)
    score(core, "SPY+QQQ+IWM equal weight", True)
    score(is_top3, f"top-3 by IS: {'+'.join(is_top3)}", True)
    score(liquid, f"all {len(liquid)} domestic, equal weight", True)

    t.to_csv("cross_sectional_fomc_results.csv", index=False)
    print("\nsaved -> cross_sectional_fomc_results.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
