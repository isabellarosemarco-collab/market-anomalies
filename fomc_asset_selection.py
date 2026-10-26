"""
fomc_asset_selection.py — the event gate stays locked to the 8 verified FOMC eves. Only the asset
choice varies. Question: does any selection rule clear +35bp/trade net out of sample without
degrading Sharpe versus 100% QQQ?

PRE-REGISTERED PREDICTION, WRITTEN BEFORE RUNNING. The cross-sectional run showed the whole
universe strengthening post-2018 by a similar amount (+8.5bp mean), not just the winners. If FOMC-eve
drift is a market-wide beta event rather than an asset-specific one, then:
    expectancy(asset) ~= beta(asset) x expectancy(market)
    volatility(asset) ~= beta(asset) x volatility(market)
so a high-beta basket should raise bp/trade AND raise variance proportionally, leaving Sharpe
approximately flat. "+35bp at the same Sharpe" would then be unreachable by beta alone, and the
bp/trade column would be measuring leverage, not skill. A rule that beats this — higher expectancy at
equal or better Sharpe — falsifies the prediction and is genuinely interesting. The `bp_per_beta`
column below is the discriminator: if it is flat across rules, it is all beta.

WHY LEVERAGE IS NOT THE ANSWER EITHER. If the only lever is beta, buying a 1.5-beta basket is
economically the same as buying 1.5x the QQQ position. That is a position-sizing decision, not a
strategy, and it should be made deliberately against drawdown tolerance rather than smuggled in
through asset choice.

CAUSALITY. Signals are computed from data through the close of the day BEFORE the eve, which is
strictly causal and needs no argument. Her spec allows a 15:55 calculation on the eve itself; the
`(near-close)` rows use the eve's own close as a proxy for the 15:45 price, which is realistic in
practice but embeds a ~15-minute approximation. Both are reported so the choice can be seen not to
matter.

SURVIVORSHIP. The single-stock universe is today's mega-caps, chosen with hindsight. AAPL/NVDA/TSLA
are in the list because they won. Any single-stock result is therefore biased upward and cannot be
compared to an ETF result at face value. Flagged in the output rather than silently averaged in.
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
from config import Config                             # noqa: E402
from core.calendar import FomcCalendar                # noqa: E402
from commercial.market_days import is_trading_day     # noqa: E402

CACHE = Path(__file__).resolve().parent / ".xsec_cache"
CACHE.mkdir(exist_ok=True)

START = "2009-01-01"                # a year of burn-in before the 2010 test window
IS_FROM, OOS_FROM = pd.Timestamp("2010-01-01"), pd.Timestamp("2019-01-01")
SEC_FEE_FRAC = 27.80 / 1e6
FINRA_TAF_PER_SHARE = 0.000166
SLIP_STOCK, SLIP_ETF = 1.0, 0.5     # bp per side, per her spec
TOP_N = 3
BETA_MIN = 1.2

ETFS = ["SMH", "XLK", "XBI", "XME", "ARKK", "IWM", "QQQ", "SPY"]
STOCKS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD"]


def load(sym: str) -> pd.DataFrame | None:
    f = CACHE / f"{sym}.csv"
    if f.exists():
        df = pd.read_csv(f, index_col=0, parse_dates=True)
    else:
        try:
            import yfinance as yf
            df = yf.Ticker(sym).history(start="1995-01-01", interval="1d",
                                        auto_adjust=False, actions=True)
        except Exception:
            return None
        if df.empty:
            return None
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        if "dividends" not in df:
            df["dividends"] = 0.0
        df = df[["open", "high", "low", "close", "volume", "dividends"]].dropna(
            subset=["open", "close"])
        df.to_csv(f)
    df = df[df.index >= START]
    return df if len(df) > 400 else None


def panel(symbols: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for s in symbols:
        d = load(s)
        if d is None:
            print(f"  skipped {s} (no usable history)")
            continue
        out[s] = d
    return out


def night_bp(df: pd.DataFrame, eve: pd.Timestamp, slip: float) -> float | None:
    """Net close(eve) -> open(next session) in bp. Raw prices + explicit dividend."""
    idx = df.index
    i = idx.searchsorted(eve)
    if i >= len(idx) or idx[i] != eve or i + 1 >= len(idx):
        return None
    entry = float(df.close.iloc[i])
    exit_ = float(df.open.iloc[i + 1]) + float(df.dividends.iloc[i + 1])
    if entry <= 0:
        return None
    cost = 2.0 * (slip / 1e4) + SEC_FEE_FRAC + FINRA_TAF_PER_SHARE / entry
    return 1e4 * ((exit_ / entry - 1.0) - cost)


def signals(df: pd.DataFrame, spy: pd.DataFrame, eve: pd.Timestamp, strict: bool) -> dict | None:
    """Beta, 5-day momentum and vol ratio as known before the entry decision."""
    idx = df.index
    i = idx.searchsorted(eve)
    if i >= len(idx) or idx[i] != eve:
        return None
    end = i if strict else i + 1          # strict: through the prior close
    if end < 70:
        return None
    r = df.close.pct_change().iloc[:end]
    rs = spy.close.pct_change().reindex(df.index).iloc[:end]
    j = pd.concat([r, rs], axis=1).dropna().tail(60)
    if len(j) < 40 or j.iloc[:, 1].var() == 0:
        return None
    beta = float(j.cov().iloc[0, 1] / j.iloc[:, 1].var())
    px = df.close.iloc[:end]
    if len(px) < 65:
        return None
    mom5 = float(px.iloc[-1] / px.iloc[-6] - 1.0)
    v5 = float(r.tail(5).std())
    v60 = float(r.tail(60).std())
    if not np.isfinite(v5) or not np.isfinite(v60) or v60 == 0:
        return None
    return {"beta": beta, "mom5": mom5, "volratio": v5 / v60}


def hac_t(x: np.ndarray) -> float:
    if len(x) < 8:
        return np.nan
    return float(sm.OLS(x, np.ones(len(x))).fit(
        cov_type="HAC", cov_kwds={"maxlags": 5}).tvalues[0])


def stats(x: np.ndarray, per_year: float = 8.0) -> dict:
    x = np.asarray(x, dtype=float)
    if len(x) < 5:
        return {}
    s = x.std(ddof=1)
    eq = np.cumprod(1 + x / 1e4)
    return {"n": len(x), "bp": x.mean(),
            "sharpe": x.mean() / s * np.sqrt(per_year) if s > 0 else np.nan,
            "dd": 100 * (eq / np.maximum.accumulate(eq) - 1).min(),
            "win": 100 * (x > 0).mean(), "t": hac_t(x)}


def run(universe: dict[str, pd.DataFrame], spy: pd.DataFrame, eves: list[pd.Timestamp],
        label: str, strict: bool) -> pd.DataFrame:
    is_etf = {s: s in ETFS for s in universe}
    rules = ["QQQ (control)", f"high beta>{BETA_MIN}", "5d momentum", "low vol ratio"]
    per_rule: dict[str, list] = {k: [] for k in rules}
    beta_of: dict[str, list] = {k: [] for k in rules}
    dates_of: dict[str, list] = {k: [] for k in rules}

    for eve in eves:
        cand = {}
        for s, df in universe.items():
            sig = signals(df, spy, eve, strict)
            if sig is None:
                continue
            r = night_bp(df, eve, SLIP_ETF if is_etf[s] else SLIP_STOCK)
            if r is None:
                continue
            cand[s] = {**sig, "ret": r}
        if len(cand) < TOP_N + 2:
            continue
        t = pd.DataFrame(cand).T.astype(float)

        if "QQQ" in t.index:
            per_rule["QQQ (control)"].append(t.loc["QQQ", "ret"])
            beta_of["QQQ (control)"].append(t.loc["QQQ", "beta"])
            dates_of["QQQ (control)"].append(eve)

        hb = t[t.beta > BETA_MIN].nlargest(TOP_N, "beta")
        for name, pick in [(f"high beta>{BETA_MIN}", hb),
                           ("5d momentum", t.nlargest(TOP_N, "mom5")),
                           ("low vol ratio", t.nsmallest(TOP_N, "volratio"))]:
            if len(pick) < TOP_N:
                continue
            per_rule[name].append(float(pick.ret.mean()))
            beta_of[name].append(float(pick.beta.mean()))
            dates_of[name].append(eve)

    rows = []
    for k in rules:
        r = np.array(per_rule[k])
        d = pd.to_datetime(dates_of[k])
        if len(r) < 8:
            continue
        b = float(np.mean(beta_of[k]))
        full, oos = stats(r), stats(r[d >= OOS_FROM])
        ins = stats(r[(d >= IS_FROM) & (d < OOS_FROM)])
        rows.append({
            "rule": k, "avg_beta": round(b, 2), "n": full["n"],
            "IS_bp": round(ins.get("bp", np.nan), 1),
            "OOS_bp": round(oos.get("bp", np.nan), 1),
            "full_bp": round(full["bp"], 1),
            "OOS_sharpe": round(oos.get("sharpe", np.nan), 2),
            "OOS_dd%": round(oos.get("dd", np.nan), 1),
            "OOS_win%": round(oos.get("win", np.nan), 1),
            "OOS_t": round(oos.get("t", np.nan), 2),
            "bp_per_beta": round(oos.get("bp", np.nan) / b, 1) if b else np.nan,
        })
    out = pd.DataFrame(rows)
    print(f"\n--- {label} " + "-" * (100 - len(label)))
    print(out.to_string(index=False))
    return out


def main() -> int:
    cal = FomcCalendar(Config().cache_path, auto_fetch=False)
    eves = sorted({pd.Timestamp(e) for d in cal.dates()
                   if (e := cal.eve_of(d, is_trading_day))} )
    eves = [e for e in eves if IS_FROM <= e <= pd.Timestamp("2026-09-30")]
    print(f"FOMC eves in window: {len(eves)}  ({eves[0].date()} .. {eves[-1].date()})")

    spy = load("SPY")
    etf_only = panel(ETFS)
    both = {**etf_only, **panel(STOCKS)}

    print("\n" + "=" * 104)
    print("ETFs ONLY — no survivorship problem, so these numbers are the trustworthy ones")
    print("=" * 104)
    a = run(etf_only, spy, eves, "ETF universe (strictly causal signals)", strict=True)
    run(etf_only, spy, eves, "ETF universe (near-close signals, 15:45 proxy)", strict=False)

    print("\n" + "=" * 104)
    print("ETFs + MEGA-CAP STOCKS — upward biased: this list is today's winners, chosen with hindsight")
    print("=" * 104)
    b = run(both, spy, eves, "ETF + single stocks (strictly causal)", strict=True)

    print("\n" + "=" * 104)
    print("VERDICT — target was +35bp OOS without degrading Sharpe vs the QQQ control")
    print("=" * 104)
    for tag, tbl in (("ETF only", a), ("ETF+stocks", b)):
        ctl = tbl[tbl.rule.str.startswith("QQQ")]
        if ctl.empty:
            continue
        c = ctl.iloc[0]
        print(f"\n  [{tag}]  control QQQ: {c.OOS_bp:+.1f}bp OOS, sharpe {c.OOS_sharpe:.2f}, "
              f"dd {c['OOS_dd%']:.1f}%")
        for _, r in tbl[~tbl.rule.str.startswith("QQQ")].iterrows():
            hit35 = r.OOS_bp >= 35
            keeps = r.OOS_sharpe >= c.OOS_sharpe
            tag2 = "PASS" if (hit35 and keeps) else "FAIL"
            why = []
            if not hit35:
                why.append(f"{r.OOS_bp:+.1f}bp < +35bp")
            if not keeps:
                why.append(f"sharpe {r.OOS_sharpe:.2f} < control {c.OOS_sharpe:.2f}")
            print(f"    {tag2}  {r.rule:18s} beta {r.avg_beta:.2f}  "
                  f"{r.OOS_bp:+6.1f}bp  sharpe {r.OOS_sharpe:5.2f}  dd {r['OOS_dd%']:5.1f}%"
                  + ("   " + "; ".join(why) if why else ""))
        print(f"    bp_per_beta across rules: "
              + ", ".join(f"{r.rule.split()[0]} {r.bp_per_beta:.1f}" for _, r in tbl.iterrows()))
        print("      ^ flat across rules => the differences are beta (leverage), not selection skill")

    a.to_csv("fomc_asset_selection_etf.csv", index=False)
    b.to_csv("fomc_asset_selection_all.csv", index=False)
    print("\nsaved -> fomc_asset_selection_etf.csv, fomc_asset_selection_all.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
