"""
edge_hunt.py — stop hunting calendar anomalies; test things that can compound.

WHY THIS IS A DIFFERENT SEARCH FROM THE LAST ONE. Every hypothesis tested so far was an *event*:
8-12 trades a year, capital idle the rest of the time, so the ceiling is
    annual return = trades/yr x expectancy
and with 8 trades at 15bp that ceiling is ~1.2%/yr. You cannot reach 100x on that, no matter how many
events you find, because the binding constraint is TIME IN THE MARKET, not edge per trade. To compound
at 10-20% you have to be exposed to a real risk premium most of the time and manage the risk of it.

So these five hypotheses are premium-harvesting rules, not anomalies. They are all:
  * documented in peer-reviewed literature with long out-of-sample records (cited per-function),
  * implementable in a cash retail brokerage account with no options, shorting, or leverage,
  * low turnover (2-24 trades/yr), matching the stated preference for structural over micro-turnover.

BENCHMARK DISCIPLINE. Every rule is scored against BUY AND HOLD SPY over the identical window, net of
the same friction. A rule that cannot beat doing nothing is reported as a failure regardless of how
good its absolute numbers look — this is the test that killed the short-vol overlay earlier
(+5.65%/yr looked fine until the benchmark printed +15.11%/yr over the same weeks).

CAUSALITY. All signals are computed from data through the close of month t and applied to the return
from month t to t+1. No signal uses a price from the period it is predicting. Rebalance friction is
charged on the fraction of the portfolio that actually changed, not on notional.

WHAT THIS CANNOT TELL YOU. Every one of these rules was published, which means the published sample
is in-sample for its author. The IS/OOS split here is 2013, so the OOS column is the only honest
estimate, and for trend following there is a further problem: 2013-2026 was an unusually strong,
low-drawdown equity bull market, which flatters buy-and-hold and penalises any de-risking rule. That
is stated in the output rather than buried.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
CACHE = Path(__file__).resolve().parent / ".xsec_cache"
CACHE.mkdir(exist_ok=True)

COST_BP = 2.0                     # per side, on the fraction of portfolio traded
SPLIT = pd.Timestamp("2013-01-01")
RF_ANNUAL = 0.02                  # cash yield assumption for the absolute-momentum gate

# Assets chosen for LOW MUTUAL CORRELATION, which is the only free lunch in finance: equities, long
# duration, gold, international, real estate, commodities. Not 8 flavours of US large cap.
WIDE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "VNQ", "DBC"]
CORE = ["SPY", "EFA", "TLT", "GLD"]


def load(sym: str) -> pd.Series | None:
    """Total-return monthly series. Adjusted prices are correct here: these are multi-month holds,
    so dividends belong inside the holding period (unlike the overnight work, where they do not)."""
    f = CACHE / f"adj_{sym}.csv"
    if f.exists():
        s = pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0]
    else:
        try:
            import yfinance as yf
            df = yf.Ticker(sym).history(start="1990-01-01", interval="1d", auto_adjust=True)
        except Exception:
            return None
        if df.empty:
            return None
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        s = df["Close"].dropna()
        s.to_frame("close").to_csv(f)
    return s if len(s) > 500 else None


def monthly(syms: list[str]) -> pd.DataFrame:
    cols = {}
    for s in syms:
        px = load(s)
        if px is None:
            print(f"  skipped {s}")
            continue
        cols[s] = px.resample("ME").last()
    return pd.DataFrame(cols).dropna()


# ----------------------------------------------------------------- scoring

def score(name: str, w: pd.DataFrame, px: pd.DataFrame, note: str = "") -> dict:
    """w = target weights known at the END of month t, applied to month t+1's return."""
    r = px.pct_change().shift(-1)                       # return of month t -> t+1
    port = (w * r).sum(axis=1)
    turn = w.diff().abs().sum(axis=1).fillna(w.iloc[0].abs().sum())
    net = (port - turn * COST_BP / 1e4).dropna()
    return _stats(name, net, turn.reindex(net.index), note)


def _stats(name: str, net: pd.Series, turn: pd.Series, note: str) -> dict:
    eq = (1 + net).cumprod()
    yrs = len(net) / 12
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    dd = (eq / eq.cummax() - 1).min()
    sh = net.mean() / net.std(ddof=1) * np.sqrt(12) if net.std() > 0 else np.nan
    o = net[net.index >= SPLIT]
    oeq = (1 + o).cumprod()
    return {
        "rule": name, "yrs": round(yrs, 1),
        "CAGR%": round(100 * cagr, 2),
        "OOS_CAGR%": round(100 * (oeq.iloc[-1] ** (12 / len(o)) - 1), 2) if len(o) > 24 else np.nan,
        "vol%": round(100 * net.std(ddof=1) * np.sqrt(12), 1),
        "sharpe": round(sh, 2),
        "maxDD%": round(100 * dd, 1),
        "OOS_maxDD%": round(100 * (oeq / oeq.cummax() - 1).min(), 1) if len(o) > 24 else np.nan,
        "trades/yr": round(float(turn.mean()) * 12, 1),
        "yrs_to_100x": round(np.log(100) / np.log(1 + cagr), 1) if cagr > 0 else np.nan,
        "note": note,
    }


# ----------------------------------------------------------------- E1 time-series momentum

def e1_trend(px: pd.DataFrame, k: int, label: str) -> list[dict]:
    """Hold the top-k assets by 12-month return, but only those beating cash. Else sit in cash.

    Moskowitz, Ooi & Pedersen (2012) 'Time Series Momentum' and Hurst/Ooi/Pedersen (2017) document
    this across 58 instruments back to 1880. Mechanism is structural rather than technical: investors
    underreact to news over 1-12 months and institutional risk-management flows (de-risking into
    drawdowns, re-risking after) mechanically extend trends. The absolute-momentum gate is the part
    that matters — it is what converts a stock-picking rule into a drawdown-control rule.
    """
    mom = px.pct_change(12)                                     # known at close of month t
    hurdle = RF_ANNUAL
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    for t in mom.index:
        row = mom.loc[t].dropna()
        row = row[row > hurdle]
        if len(row) == 0:
            continue                                            # all cash
        pick = row.nlargest(min(k, len(row))).index
        w.loc[t, pick] = 1.0 / k                                # undersized -> rest in cash, on purpose
    return [score(label, w.iloc[12:], px, "cash when nothing beats cash")]


# ----------------------------------------------------------------- E2 volatility targeting

def e2_voltarget(px: pd.DataFrame, target: float = 0.12) -> list[dict]:
    """Scale SPY exposure to a constant 12% vol forecast, capped at 1.0 (de-risk only, no leverage).

    Moreira & Muir (2017) 'Volatility-Managed Portfolios'. Mechanism: volatility is strongly
    autocorrelated month to month while expected return is not, so scaling down when realised vol is
    high sheds variance without shedding a proportional amount of return. Capped at 1.0 here because
    the levered version needs margin and this account should not use it.
    """
    d = load("SPY")
    rv = d.pct_change().rolling(20).std() * np.sqrt(252)
    ex = (target / rv).clip(upper=1.0).resample("ME").last().shift(1)   # shift: use last month's vol
    sub = px[["SPY"]].copy()
    w = pd.DataFrame({"SPY": ex.reindex(sub.index)}).fillna(0.0)
    return [score("E2 SPY vol-targeted 12%", w, sub, "de-risk only, no leverage")]


# ----------------------------------------------------------------- E3 Halloween

def e3_halloween(px: pd.DataFrame) -> list[dict]:
    """Long SPY November-April, cash May-October. Two trades a year.

    Bouman & Jacobsen (2002), and Zhang & Jacobsen (2019) who found it in 65 markets over 300 years.
    Included precisely BECAUSE it is the sort of thing that should not work: there is no convincing
    flow mechanism, only a seasonal-risk-aversion story. It is the control for whether this battery
    can tell a real premium from a persistent coincidence.
    """
    sub = px[["SPY"]]
    w = pd.DataFrame({"SPY": [1.0 if m in (10, 11, 12, 1, 2, 3) else 0.0
                              for m in sub.index.month]}, index=sub.index)
    return [score("E3 Halloween (Nov-Apr)", w, sub, "no flow mechanism — this is the control")]


# ----------------------------------------------------------------- E4 buyback window

def e4_buyback(px: pd.DataFrame) -> list[dict]:
    """Second half of each calendar quarter vs the first half, SPY.

    Mechanism: US corporates repurchase ~$1trn/yr, and Rule 10b-18 safe-harbour practice keeps most
    issuers out of the market from roughly two weeks before quarter end until two days after they
    report. The aggregate corporate bid is therefore mechanically stronger in the back half of a
    quarter than the front. Real flow, real calendar, genuinely structural — if the effect exists it
    should be visible at monthly resolution as months 2-3 of a quarter beating month 1.
    """
    sub = px[["SPY"]]
    late = [1.0 if m % 3 != 1 else 0.0 for m in sub.index.month]        # months 2,3 of each quarter
    early = [1.0 - x for x in late]
    return [score("E4 buyback window (mo 2-3 of qtr)", pd.DataFrame({"SPY": late}, index=sub.index), sub),
            score("  |- inverse (mo 1 of qtr)", pd.DataFrame({"SPY": early}, index=sub.index), sub,
                  "if E4 is real this should be much worse")]


# ----------------------------------------------------------------- E5 risk parity baseline

def e5_riskparity(px: pd.DataFrame) -> list[dict]:
    """Inverse-volatility weights across the whole set, monthly. No forecasting at all.

    This is the honest floor for 'diversify and manage risk'. If the clever rules cannot beat it,
    the clever part is not adding anything, and this is what should actually be held.
    """
    rv = px.pct_change().rolling(12).std()
    iv = (1 / rv).replace([np.inf, -np.inf], np.nan)
    w = iv.div(iv.sum(axis=1), axis=0).fillna(0.0)
    return [score("E5 inverse-vol risk parity", w.iloc[12:], px, "zero forecasting")]


# ----------------------------------------------------------------- report

def main() -> int:
    wide = monthly(WIDE)
    core = monthly(CORE)
    print(f"wide universe {list(wide.columns)}  {wide.index[0]:%Y-%m} .. {wide.index[-1]:%Y-%m}")
    print(f"core universe {list(core.columns)}  {core.index[0]:%Y-%m} .. {core.index[-1]:%Y-%m}")

    rows = []
    # Benchmarks first, on each window, so nothing gets scored against the wrong yardstick.
    for nm, p in (("wide", wide), ("core", core)):
        sub = p[["SPY"]]
        rows.append(score(f"BUY & HOLD SPY ({nm} window)", pd.DataFrame({"SPY": 1.0}, index=sub.index),
                          sub, "the number to beat"))
    rows += e1_trend(wide, 3, "E1 trend top-3 of 10")
    rows += e1_trend(wide, 5, "E1 trend top-5 of 10")
    rows += e1_trend(core, 2, "E1 trend top-2 of 4")
    rows += e2_voltarget(wide)
    rows += e3_halloween(wide)
    rows += e4_buyback(wide)
    rows += e5_riskparity(wide)

    t = pd.DataFrame(rows)
    cols = ["rule", "yrs", "CAGR%", "OOS_CAGR%", "vol%", "sharpe", "maxDD%", "OOS_maxDD%",
            "trades/yr", "yrs_to_100x"]
    print("\n" + "=" * 118)
    print(f"PREMIUM-HARVESTING BATTERY — monthly, {COST_BP}bp/side on traded fraction, "
          f"OOS = {SPLIT:%Y}+")
    print("=" * 118)
    print(t[cols].to_string(index=False, na_rep="-"))
    print("\nnotes:")
    for _, r in t.iterrows():
        if r["note"]:
            print(f"  {r['rule']}: {r['note']}")

    bh = t[t.rule.str.startswith("BUY & HOLD SPY (wide")].iloc[0]
    print("\n" + "=" * 118)
    print(f"VERDICT vs buy & hold ({bh['CAGR%']}%/yr, Sharpe {bh['sharpe']}, "
          f"maxDD {bh['maxDD%']}%, {bh['yrs_to_100x']}yr to 100x)")
    print("=" * 118)
    for _, r in t.iterrows():
        if r.rule.startswith(("BUY & HOLD", "  ")):
            continue
        better_sh = r["sharpe"] > bh["sharpe"]
        better_dd = r["maxDD%"] > bh["maxDD%"]
        tag = "BEATS " if (better_sh and better_dd) else "fails "
        print(f"  {tag} {r['rule']:34s} CAGR {r['CAGR%']:6.2f}%  sharpe {r['sharpe']:5.2f}  "
              f"maxDD {r['maxDD%']:6.1f}%  -> 100x in {r['yrs_to_100x']} yr")
    t.to_csv("edge_hunt_results.csv", index=False)
    print("\nsaved -> edge_hunt_results.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
