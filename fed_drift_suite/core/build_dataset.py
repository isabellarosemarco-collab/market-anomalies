"""
Builds the frozen research dataset that `analytics.py` serves to clients.

WHY A COMMITTED DATASET INSTEAD OF HARDCODED SUMMARY STATISTICS

`/v1/research/metrics` is a due-diligence endpoint. An allocator who asks "where does this Sharpe
come from" must be able to get the per-trade series behind it, recompute every published number, and
find no free parameters. Typing `"sharpe": 1.16` into a dict would make the endpoint unfalsifiable
and worthless for that purpose. So this script writes one row per (symbol, meeting) to
`data/fomc_eve_returns.csv`, `analytics.py` computes every statistic from that file at import, and
the API exposes the rows themselves at `/v1/research/trades`.

METHOD

For each scheduled FOMC decision date, the trade is:  buy the previous session's close, sell the
decision session's open. On a daily bar series the previous session is simply the preceding bar, so
the trade return is the decision day's overnight leg.

Prices are raw (`auto_adjust=False`) with an explicit dividend term. This matters more than it
sounds: yfinance's adjustment factor steps on the ex-dividend date, which removes the ex-date price
drop from the overnight return and silently credits the whole dividend to the overnight bucket. A
naive adjusted decomposition hands the overnight session ~1.3-1.8%/yr of SPY's dividend yield and
reports it as drift. The correct overnight total return is

    r_night = (Open_t + Div_t) / Close_{t-1} - 1

because a holder who buys at the close of t-1 and sells at the open of the ex-date t is the holder
of record and does receive the dividend. Both that and the dividend-stripped price-only version are
written out, so a reader can see the effect is not a dividend artifact. (It is not: only a handful
of ex-dates land on decision days.)

COSTS are charged in full, per trade, and stored per row rather than assumed:
0.5bp slippage per side, IBKR $0.0035/share both ways, SEC Section 31 at 27.80/1e6 of sell notional,
FINRA TAF at $0.000166/share on the sell. That is ~4bp round trip against a ~20bp gross edge, so it
is a fifth of the thesis and cannot be waved away.

Run:  python -m core.build_dataset            (from the fed_drift_suite directory)
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from core.calendar import DATA_DIR, FomcCalendar

SYMBOLS = ["SPY", "QQQ", "IWM"]
START = "1993-01-01"                    # SPY inception; QQQ and IWM start at their own first bar

SLIP_BP_PER_SIDE = 0.5                  # auction fills, retail size in a >$40bn/day name
COMMISSION_PER_SHARE = 0.0035           # IBKR fixed tier
SEC_FEE_FRAC = 27.80 / 1e6              # 0.278bp of sell notional (FY2025 rate)
FINRA_TAF_PER_SHARE = 0.000166          # sells only

# The rule was designed on 2018+ data. Everything before that was never looked at while choosing it.
HOLDOUT_END = date(2017, 12, 31)

RETURNS_CSV = DATA_DIR / "fomc_eve_returns.csv"
NIGHTS_CSV = DATA_DIR / "all_nights.csv"
PROVENANCE_JSON = DATA_DIR / "provenance.json"


def load_raw(symbol: str, start: str = START) -> pd.DataFrame:
    import yfinance as yf

    df = yf.Ticker(symbol).history(start=start, interval="1d", auto_adjust=False, actions=True)
    if df is None or df.empty:
        raise RuntimeError(f"{symbol}: yfinance returned no data")
    df = df.rename(columns=str.lower)
    df["dividends"] = df.get("dividends", 0.0).fillna(0.0)
    df = df[["open", "high", "low", "close", "volume", "dividends"]].dropna(
        subset=["open", "close"])
    df = df[(df[["open", "close"]] > 0).all(axis=1)]
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def overnight_legs(df: pd.DataFrame) -> pd.DataFrame:
    """Overnight and intraday decomposition, with the identity asserted."""
    o, c, d = df["open"], df["close"], df["dividends"]
    pc = c.shift(1)
    r = pd.DataFrame(index=df.index)
    r["r_night"] = (o + d) / pc - 1.0                   # what an overnight holder earns
    r["r_night_px"] = o / pc - 1.0                      # dividend stripped out
    r["r_day"] = c / o - 1.0
    r["r_bh_reinv"] = c * (1 + d / o) / pc - 1.0
    r["prev_close"] = pc
    r["open"] = o
    r["dividend"] = d
    r = r.dropna()

    resid = (1 + r["r_night"]) * (1 + r["r_day"]) - (1 + r["r_bh_reinv"])
    worst = float(resid.abs().max())
    if worst > 1e-12:
        raise AssertionError(f"decomposition identity violated, max resid {worst:.3e}")
    return r


def round_trip_cost_frac(entry_price: float) -> float:
    """Total round-trip friction as a fraction of notional. Buy the close, sell the open."""
    slip = 2.0 * (SLIP_BP_PER_SIDE / 1e4)
    comm = 2.0 * COMMISSION_PER_SHARE / entry_price
    reg = SEC_FEE_FRAC + FINRA_TAF_PER_SHARE / entry_price       # sell side only
    return slip + comm + reg


def build() -> pd.DataFrame:
    cal = FomcCalendar()
    decisions = cal.dates()
    today = date.today()
    rows, nights = [], []

    for sym in SYMBOLS:
        raw = load_raw(sym)
        legs = overnight_legs(raw)
        idx = legs.index
        bars = {d.date(): d for d in idx}

        # Every night in the sample, FOMC or not. This is the comparison population for the
        # permutation test: without it, "is +19bp unusual?" has no denominator a client can audit.
        dec_set = {d.isoformat() for d in decisions}
        nights.append(pd.DataFrame({
            "symbol": sym,
            "date": [d.date().isoformat() for d in idx],
            "gross_bp": (1e4 * legs["r_night"].to_numpy()).round(4),
            "cost_bp": (1e4 * legs["prev_close"].map(round_trip_cost_frac).to_numpy()).round(4),
        }).assign(is_fomc_eve_night=lambda f: f["date"].isin(dec_set)))

        for dec in decisions:
            if dec >= today:
                continue                                 # no realized return yet
            bar = bars.get(dec)
            if bar is None:
                continue                                 # decision predates this ETF's inception
            pos = idx.get_loc(bar)
            if pos == 0:
                continue
            eve = idx[pos - 1].date()
            row = legs.loc[bar]
            cost = round_trip_cost_frac(float(row["prev_close"]))
            rows.append({
                "symbol": sym,
                "decision_date": dec.isoformat(),
                "eve_date": eve.isoformat(),
                "entry_close": round(float(row["prev_close"]), 4),
                "exit_open": round(float(row["open"]), 4),
                "dividend": round(float(row["dividend"]), 6),
                "gross_bp": round(1e4 * float(row["r_night"]), 4),
                "gross_px_bp": round(1e4 * float(row["r_night_px"]), 4),
                "cost_bp": round(1e4 * cost, 4),
                "net_bp": round(1e4 * (float(row["r_night"]) - cost), 4),
                "sample": "holdout" if dec <= HOLDOUT_END else "design",
            })

    out = pd.DataFrame(rows).sort_values(["symbol", "decision_date"]).reset_index(drop=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(RETURNS_CSV, index=False)

    allnights = pd.concat(nights, ignore_index=True)
    allnights["net_bp"] = (allnights["gross_bp"] - allnights["cost_bp"]).round(4)
    allnights.to_csv(NIGHTS_CSV, index=False)

    prov = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "price_source": "yfinance (Yahoo Finance) daily bars, auto_adjust=False, actions=True",
        "calendar_source": "federalreserve.gov fomccalendars.htm + fomchistorical{year}.htm",
        "symbols": SYMBOLS,
        "n_decision_dates_cached": len(decisions),
        "n_trades": int(len(out)),
        "n_baseline_nights": int(len(allnights)),
        "first_trade": out["decision_date"].min(),
        "last_trade": out["decision_date"].max(),
        "holdout_end": HOLDOUT_END.isoformat(),
        "holdout_note": ("The rule was specified on 2018+ data. Rows marked 'holdout' "
                         "(<= 2017-12-31) were never examined while choosing the rule and are the "
                         "only genuinely out-of-sample evidence here."),
        "cost_model": {
            "slippage_bp_per_side": SLIP_BP_PER_SIDE,
            "commission_per_share": COMMISSION_PER_SHARE,
            "sec_section31_frac_of_sell_notional": SEC_FEE_FRAC,
            "finra_taf_per_share": FINRA_TAF_PER_SHARE,
            "charged": "full round trip, every trade, computed per-row from the entry price",
        },
        "trade_definition": ("Buy at the close of the trading session immediately preceding a "
                            "scheduled FOMC decision; sell at that decision session's open. "
                            "No overnight position is held through the 14:00 ET statement."),
    }
    PROVENANCE_JSON.write_text(json.dumps(prov, indent=2))
    return out


if __name__ == "__main__":
    df = build()
    print(f"wrote {RETURNS_CSV}  ({len(df)} rows)")
    for sym, g in df.groupby("symbol"):
        hold = g[g["sample"] == "holdout"]["net_bp"]
        des = g[g["sample"] == "design"]["net_bp"]
        print(f"  {sym}: {len(g):3d} trades  {g['decision_date'].min()} .. "
              f"{g['decision_date'].max()}  "
              f"net mean {g['net_bp'].mean():+6.2f}bp  "
              f"(holdout n={len(hold):3d} {hold.mean():+6.2f}  "
              f"design n={len(des):2d} {des.mean():+6.2f})")
    print(f"wrote {NIGHTS_CSV} and {PROVENANCE_JSON}")
