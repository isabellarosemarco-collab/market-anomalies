"""
cboe.py — real dealer gamma and real expiration-day option spreads, free and without an API key.

WHY THIS FILE EXISTS: THE SEARCH FOR HISTORICAL OPEN INTEREST, AND WHAT IT ACTUALLY FOUND. The brief's K*
is the strike with maximal gamma-weighted OPEN INTEREST, measured on past Fridays back to 2022. Six
sources were checked before concluding anything:

  yfinance option_chain      has OI and strikes, but serves UNEXPIRED expirations only. No history.
  DoltHub post-no-preference has real per-strike gamma, IV and quotes back past 2022, free and with no
                             key -- and it is still unusable here for two reasons: there is no
                             open_interest column at all, and each daily snapshot carries only three
                             expirations beginning about two weeks out, so the expiring front week (the
                             only chain that pins) is absent by construction.
  CBOE cdn delayed_quotes    THE FIND. Full chain for every listed expiration with CBOE's OWN computed
                             gamma and delta, open_interest, bid, ask, AND bid_size/ask_size. Free, no
                             key, no rate limit encountered. Current snapshot only.
  OCC marketdata             400/403 without credentials.
  Kaggle / Polygon / ThetaData / Tradier / IBKR / Alpaca -- every one needs an account or key, and none
                             is present on this machine (checked: no ~/.kaggle/kaggle.json, no key
                             env vars, and fed_drift_suite/.env still has all four Alpaca fields empty).

So historical per-strike OI is genuinely unavailable at zero cost, and that fact shapes the study rather
than being worked around silently. What CBOE gives instead is better than a proxy for the ONE question
where precision matters most: today is 2026-09-25, a real expiration Friday, and these are real bids,
real asks and real sizes on options that expire in hours. Question 3 -- do retail spreads eat the decay --
is therefore answered with live market data rather than a slippage assumption.

GAMMA HERE IS CBOE'S, NOT MINE, AND THAT MATTERS. A first pass computed gamma from Black-Scholes using
yfinance implied vols and produced nonsense, because yfinance reports impliedVolatility of 1e-5 for deep
in-the-money contracts. Garbage IV gives garbage gamma gives a garbage K*. CBOE publishes its own greeks
computed from its own vol surface, so the gamma concentration below is the exchange's number.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import requests

UA = {"User-Agent": "BellaMarco-Research/1.0 (bellamarco.research@gmail.com)"}
CDN = "https://cdn.cboe.com/api/global/delayed_quotes/options/{sym}.json"
BAND = 0.015          # the brief's +-1.5% search window for K*
GRID = 5.0


def chain(sym: str) -> tuple[pd.DataFrame, float, str]:
    """Full option chain with CBOE greeks and OI. Returns (df, spot, timestamp)."""
    r = requests.get(CDN.format(sym=sym), headers=UA, timeout=60)
    r.raise_for_status()
    j = r.json()
    d = j["data"]
    rows = []
    for o in d["options"]:
        # OCC symbol: ROOT + YYMMDD + C/P + strike*1000, right-packed. Parsing this rather than trusting
        # a separate field because the JSON exposes the contract only as this string.
        s = o["option"]
        body = s[-15:]
        rows.append({"expiry": pd.Timestamp(f"20{body[0:2]}-{body[2:4]}-{body[4:6]}"),
                     "cp": body[6], "strike": int(body[7:]) / 1000.0,
                     "bid": o["bid"], "ask": o["ask"], "bid_sz": o["bid_size"],
                     "ask_sz": o["ask_size"], "oi": o["open_interest"], "vol": o["volume"],
                     "gamma": o["gamma"], "delta": o["delta"], "iv": o["iv"]})
    df = pd.DataFrame(rows)
    spot = float(d.get("current_price") or d.get("close") or np.nan)
    return df, spot, j.get("timestamp", "")


def pin_strike(df: pd.DataFrame, spot: float, expiry: pd.Timestamp) -> dict:
    """K* per the brief: max gamma-weighted OI within +-1.5% of spot, on the given expiration."""
    e = df[(df.expiry == expiry) & (df.strike.between(spot * (1 - BAND), spot * (1 + BAND)))].copy()
    if e.empty:
        return {}
    # Dollar gamma per 1% move, which is the units dealers actually hedge in.
    e["gxoi"] = e.gamma.abs() * e.oi * 100 * spot * spot * 0.01
    by = e.groupby("strike").agg(gxoi=("gxoi", "sum"), oi=("oi", "sum"), vol=("vol", "sum"))
    if by.empty or by.gxoi.sum() == 0:
        return {}
    kstar = float(by.gxoi.idxmax())
    grid = sorted(df[df.expiry == expiry].strike.unique())
    nearest = min(grid, key=lambda k: abs(k - spot))
    return {"kstar": kstar, "nearest": nearest, "on_5_grid": abs(kstar % GRID) < 1e-9,
            "dist_bp": 1e4 * abs(spot - kstar) / spot,
            "share_of_band_gamma": float(by.gxoi.max() / by.gxoi.sum()),
            "table": by.sort_values("gxoi", ascending=False)}


def atm_cost(df: pd.DataFrame, spot: float, expiry: pd.Timestamp, k: float) -> dict:
    """The straddle at K*: what does crossing the spread actually cost, right now, on expiration day?

    This is question 3 stated in the only units that matter. The straddle sold at K* collects its MID
    value; a retail account that must cross pays the ask to buy and receives the bid to sell. The
    round-trip cost as a share of the mid premium is the fraction of the entire theoretical decay that
    the market maker keeps before the trade has any view at all.
    """
    e = df[(df.expiry == expiry) & (df.strike == k)]
    c = e[e.cp == "C"]
    p = e[e.cp == "P"]
    if c.empty or p.empty:
        return {}
    cb, ca = float(c.bid.iloc[0]), float(c.ask.iloc[0])
    pb, pa = float(p.bid.iloc[0]), float(p.ask.iloc[0])
    mid = (cb + ca) / 2 + (pb + pa) / 2
    half = ((ca - cb) + (pa - pb)) / 2          # one-way cost of crossing both legs
    return {"strike": k, "straddle_mid": mid, "one_way_cross": half,
            "round_trip": 2 * half,
            "cross_pct_of_mid": 100 * half / mid if mid > 0 else np.nan,
            "rt_pct_of_mid": 100 * 2 * half / mid if mid > 0 else np.nan,
            "mid_bp_of_spot": 1e4 * mid / spot,
            "call": (cb, ca, float(c.bid_sz.iloc[0]), float(c.ask_sz.iloc[0])),
            "put": (pb, pa, float(p.bid_sz.iloc[0]), float(p.ask_sz.iloc[0]))}
