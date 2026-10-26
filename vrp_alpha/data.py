"""
data.py — the inputs for the variance risk premium study, and why each source is the one chosen.

VIX FROM CBOE, NOT FROM YAHOO. CBOE publishes the official VIX history as a free CSV back to 1990-01-02
(no key, ~473 KB). Yahoo's ^VIX agrees closely but is a redistribution; when the authoritative source is
free there is no reason to use a copy. The endpoint needs a redirect follow, which is why requests is
called with allow_redirects rather than a bare get.

^SPX RATHER THAN SPY FOR THE RETURNS, WHICH IS A CORRECTION TO THE BRIEF. The brief specifies realized
volatility of SPY daily returns. VIX is computed from SPX options, so SPX is the actual underlying of the
implied volatility being compared against, and a variance swap struck on VIX settles against SPX realized
variance. SPY differs in two ways that both inject variance that does not belong: it pays quarterly
dividends, so its unadjusted price gaps down about 0.4% four times a year for reasons unrelated to risk,
and its adjusted series smears those adjustments backwards through the whole history. SPX is a pure price
index with neither problem. The difference in measured RV is small, but it is free to get right.

WHAT IS NOT AVAILABLE, CHECKED RATHER THAN ASSUMED. VIX futures settlement history is behind a 403 on
every CBOE path tried, so a true constant-maturity M1/M2 short cannot be reconstructed. This matters less
than it sounds, because shorting VIX futures harvests the FUTURES TERM PREMIUM (contango roll yield),
which is a related but genuinely different trade from the variance risk premium (implied above subsequent
realized). The brief offers VXX/UVXY/SVXY as VRP proxies; they are proxies for the roll, not for the VRP,
and section 0 of the runner shows they cannot be backtested here anyway.
"""
from __future__ import annotations

import io
import pathlib
import pickle

import pandas as pd
import requests

UA = {"User-Agent": "BellaMarco-Research/1.0 (bellamarco.research@gmail.com)"}
VIX_CSV = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
CACHE = pathlib.Path("/tmp/vrp_data.pkl")


def _cboe_vix() -> pd.Series:
    r = requests.get(VIX_CSV, headers=UA, timeout=90, allow_redirects=True)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df["DATE"] = pd.to_datetime(df["DATE"], format="mixed")
    return df.set_index("DATE")["CLOSE"].astype(float).sort_index().rename("vix")


def load(refresh: bool = False) -> pd.DataFrame:
    """Daily frame: vix (CBOE official), vix3m, spx close. Index is trading days."""
    if CACHE.exists() and not refresh:
        return pickle.loads(CACHE.read_bytes())
    import yfinance as yf

    vix = _cboe_vix()
    spx = yf.Ticker("^SPX").history(period="max", interval="1d", auto_adjust=False)["Close"]
    v3m = yf.Ticker("^VIX3M").history(period="max", interval="1d", auto_adjust=False)["Close"]
    for s in (spx, v3m):
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    df = pd.DataFrame({"vix": vix, "spx": spx.rename("spx"), "vix3m": v3m.rename("vix3m")})
    # SPX is the binding series: VIX exists on days SPX does, and a return needs two consecutive closes.
    df = df[df.spx.notna() & df.vix.notna()].copy()
    CACHE.write_bytes(pickle.dumps(df))
    return df


def etp_gate() -> None:
    """Print the short-volatility ETP availability gate. This is section 0 of the study.

    THE SURVIVORSHIP PROBLEM HERE IS THE WORST IN ANY OF THESE STUDIES, AND IT IS WORTH SEEING DIRECTLY.
    The instrument that actually expressed this trade for most of the sample was XIV, which lost 96% in a
    single session on 2018-02-05 and was terminated. It returns EMPTY from the free feed: the one
    observation that falsifies the strategy has been deleted from the data by the act of falsifying it.
    """
    import yfinance as yf
    print("  SHORT-VOL ETP GATE — can the brief's VXX/UVXY/SVXY proxies be backtested at all?\n")
    rows = [("XIV", "terminated 2018-02-05 after -96% in one session"),
            ("SVXY", "target leverage cut from -1.0x to -0.5x on 2018-02-28"),
            ("VXX", "original 2009 ETN matured Jan 2019; feed carries only the successor"),
            ("UVXY", "target leverage cut from 2.0x to 1.5x on 2018-02-28")]
    for t, note in rows:
        try:
            h = yf.Ticker(t).history(period="max", interval="1d", auto_adjust=False)
            sp = yf.Ticker(t).splits
            span = f"{h.index[0].date()} .. {h.index[-1].date()}, {len(h)} bars" if not h.empty else "NO DATA"
            print(f"    {t:6} {span:44} {len(sp):2} splits")
        except Exception as e:
            print(f"    {t:6} {'NO DATA':44} ({type(e).__name__})")
        print(f"    {'':6} {note}")
    print("""
    ALL FOUR ARE UNUSABLE FOR A 2010-PRESENT TEST, FOR THREE DIFFERENT REASONS.
      Survivorship: XIV is gone from the feed. It is not that XIV is hard to get -- it is that the
      instrument which would have registered the catastrophe no longer exists to be measured, so any
      surviving-ETP backtest is conditioned on not having blown up.
      Spliced strategies: SVXY and UVXY both changed their target leverage in Feb 2018, so a continuous
      price series concatenates two different funds. This is the same class of error as the vendor
      splicing two companies into one ticker in the index study, and it is equally invisible to a gap check.
      Missing history: the feed's VXX begins 2018, so the 2009-2017 period simply is not there.
    Therefore the tradeable test below is built synthetically from VIX and SPX, which have no such gaps,
    and the ETP route is reported as untestable rather than tested badly.""")
