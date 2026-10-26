"""
reaction.py — the question the whole 8-K leg reduces to: is the move already over when you can first see it?

THE TWO MEASURED FACTS THIS COMBINES, AND WHY NEITHER ALONE DECIDES ANYTHING.
  edgar_timing.py  : only ~1.7% of 8-Ks are hard-material AND land inside 09:30-16:00 ET (~1/day market-wide)
  edgar_latency.py : acceptance -> first visible on the free feed is ~30-60s, median ~39s, and that delay
                     sits inside the SEC's pipeline (the feed snapshot itself is 1s old), so no amount of
                     polling speed removes it
A rare event is still tradeable if the payoff is large. A 40-second delay is harmless if the price takes
minutes to adjust. The leg lives or dies on the INTERACTION: how much of the total move happens in the
window you are structurally locked out of.

HOW THE LOCKOUT IS ISOLATED WITH ONLY 1-MINUTE BARS. One-minute resolution cannot see a 39-second lag
directly, but it does not need to. Bar t0 is the minute containing the acceptance timestamp, and the
median filing is invisible to you for most of that bar. So:
    LOST     = open(t0) -> close(t0)          the minute you cannot act in
    CAPTURED = open(t0+1) -> close(t0+k)      what is still on the table once you can act
That split is conservative in the strategy's favour: it credits you with the entire t0+1 bar even though
in practice you are placing an order partway into it, and it charges you only one bar for the lockout even
when the lag runs past 57 seconds. If CAPTURED still looks thin under assumptions that generous, it is
thinner in reality.

WHY THE PRESS RELEASE MATTERS MORE THAN EDGAR, WHICH IS THE THING THE ORIGINAL PLAN DID NOT PRICE. An
issuer announcing earnings does not publish to EDGAR and wait. It pushes the release to Business Wire or
GlobeNewswire at the same instant, and the wire hits Bloomberg, Reuters and every retail broker's news
panel with no 40-second queue. EDGAR is frequently the SLOWEST of the simultaneous channels, not the
fastest. So the measured LOST share is not "the SEC's fault" — it is other people reading a faster copy of
the same document. That is the opposite of the premise: rather than reaching a primary source before the
crowd, the plan reaches a secondary mirror of it after the crowd.

DATA CONSTRAINT, STATED BECAUSE IT BOUNDS THE SAMPLE. Free 1-minute equity bars go back about 7 days, so
this study can only cover filings from the last week. That caps n in the low tens. The numbers below are
therefore directional, and the honest use of a small sample is to look for an effect large enough to
survive it rather than to estimate one precisely.
"""
from __future__ import annotations

import collections
import statistics
import time
from datetime import timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from event_alpha.edgar_timing import HARD_PHRASES, EdgarTiming, classify

ET = ZoneInfo("America/New_York")
TICKER_MAP = "https://www.sec.gov/files/company_tickers.json"
UA = "BellaMarco-Research/1.0 (bellamarco.research@gmail.com)"


def cik_to_ticker() -> dict[str, str]:
    """CIK (zero-stripped string) -> primary ticker, from the SEC's own mapping file."""
    r = requests.get(TICKER_MAP, headers={"User-Agent": UA}, timeout=40)
    r.raise_for_status()
    out = {}
    for row in r.json().values():
        cik = str(row["cik_str"])
        # First listing wins: the file is ordered by size, so for dual-class names this picks the
        # more liquid line, which is also the only one with usable 1-minute bars.
        out.setdefault(cik, row["ticker"])
    return out


def gather_hard(days: int = 7) -> list:
    """Every hard-material 8-K in the last `days` business days. No sampling — the population is small."""
    e = EdgarTiming()
    d = pd.Timestamp.now(tz=ET).date()
    out, checked, walked = [], 0, 0
    while checked < days and walked < 25:
        walked += 1
        fl = e.daily_index(d) if d.weekday() < 5 else []
        if fl:
            checked += 1
            hard_n = 0
            for f in fl:
                e.header(f)
                if f.tags:
                    out.append(f)
                    hard_n += 1
            print(f"  {d}: {len(fl)} 8-Ks -> {hard_n} hard-material")
        d -= timedelta(days=1)
        if checked >= days:
            break
    return out


def bars(ticker: str) -> pd.DataFrame | None:
    import yfinance as yf
    try:
        df = yf.Ticker(ticker).history(period="7d", interval="1m", auto_adjust=False)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    df.index = df.index.tz_convert(ET)
    return df


def study(days: int = 7, horizons: tuple[int, ...] = (1, 2, 5, 15, 30)) -> pd.DataFrame:
    hard = gather_hard(days)
    intraday = [f for f in hard if f.session == "INTRADAY"]
    print(f"\n  hard-material 8-Ks found: {len(hard)}   of which accepted intraday: {len(intraday)}")
    if not intraday:
        print("  Nothing to measure: no hard-material filing landed in a continuous session this week.")
        return pd.DataFrame()

    cmap = cik_to_ticker()
    rows = []
    for f in intraday:
        tkr = cmap.get(str(int(f.cik)))
        if not tkr:
            continue                        # private filer or no listed common stock
        b = bars(tkr)
        time.sleep(0.4)
        if b is None or len(b) < 60:
            continue
        t0 = pd.Timestamp(f.accepted, tz=ET).floor("min")
        if t0 not in b.index:
            continue                        # halted, or the bar is outside the retained window
        i = b.index.get_loc(t0)
        o0, c0 = b.Open.iloc[i], b.Close.iloc[i]
        row = {"ticker": tkr, "when": t0.strftime("%m-%d %H:%M"),
               "items": ",".join(f.tags),
               "lost_bp": round(1e4 * (c0 / o0 - 1), 1)}
        for k in horizons:
            if i + k < len(b):
                o1 = b.Open.iloc[i + 1]
                row[f"cap_{k}m_bp"] = round(1e4 * (b.Close.iloc[i + k] / o1 - 1), 1)
        row["vol_x"] = round(b.Volume.iloc[i] / max(b.Volume.iloc[max(0, i-30):i].mean(), 1), 1)
        # CAPACITY AND SPREAD, BECAUSE A "MOVE" YOU CANNOT TRANSACT IN IS NOT A MOVE. Two numbers decide
        # whether the returns above are real or arithmetic. First, dollar volume: if the whole name turns
        # over $200k in a day, a $5,000 order is a meaningful share of it and you are the price move
        # rather than a passenger on it. Second, the high-low range of a single minute as a spread proxy:
        # in a wide-spread microcap, consecutive prints alternate between bid and ask, so a 1-minute
        # "return" of 60bp can be pure bounce with nothing having happened. Where that range is the same
        # order of magnitude as the measured move, the move is an artifact of which side printed last.
        row["min_$vol"] = round(b.Close.iloc[i] * b.Volume.iloc[i])
        row["day_$vol"] = round((b.Close * b.Volume).sum() / 7)
        row["spread_bp"] = round(1e4 * (b.High.iloc[i] - b.Low.iloc[i]) / max(c0, 1e-9), 1)
        rows.append(row)

    if not rows:
        print("  No filing could be matched to usable 1-minute bars.")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    print(f"\n{df.to_string(index=False)}")
    report(df, horizons)
    return df


def report(df: pd.DataFrame, horizons: tuple[int, ...]) -> None:
    print(f"\n  n = {len(df)} filings with usable bars")
    # ABSOLUTE moves, not signed. The strategy is directional and would trade either way, so what matters
    # is the SIZE of the move in each window; averaging signed returns across long and short setups
    # cancels real moves against each other and manufactures a zero.
    lost = df.lost_bp.abs()
    print(f"\n  |move| in the acceptance minute — the window you are locked out of:")
    print(f"    median {lost.median():6.1f} bp   mean {lost.mean():6.1f} bp   max {lost.max():6.1f} bp")
    for k in horizons:
        col = f"cap_{k}m_bp"
        if col not in df:
            continue
        cap = df[col].abs()
        share = lost.median() / max(lost.median() + cap.median(), 1e-9)
        print(f"  |move| available from t0+1 through t0+{k:<2}m: median {cap.median():6.1f} bp"
              f"   -> lockout captured {100*share:4.0f}% of the {k}-minute total")
    if "vol_x" in df:
        print(f"\n  volume in the acceptance minute vs prior 30-minute average: "
              f"median {df.vol_x.median():.1f}x")
        print("  A multiple near or below 1.0 means the market did not notice the filing at all, which is")
        print("  its own verdict: there is no reaction to be early to.")

    if "spread_bp" in df:
        # THE COMPARISON THAT DECIDES IT. Put the 1-minute high-low range next to the 1-minute "captured"
        # move. If the range is as large as the move, the move is bid-ask bounce and the sign of it is
        # decided by which side happened to print, not by the filing.
        sp, cap1 = df.spread_bp.median(), df.cap_1m_bp.abs().median()
        print(f"\n  1-minute high-low range (spread proxy): median {sp:.1f} bp")
        print(f"  1-minute 'captured' move:                median {cap1:.1f} bp  "
              f"-> move / spread = {cap1/max(sp,1e-9):.2f}x")
        if cap1 <= 2 * sp:
            print("  The move is inside twice the spread. That is bounce, not a reaction: you would pay")
            print("  the spread to enter and the spread to exit, and the entire measured edge is smaller")
            print("  than the round trip.")
        print(f"\n  liquidity of the names that actually file hard 8-Ks intraday:")
        print(f"    acceptance-minute dollar volume:  median ${df['min_$vol'].median():,.0f}")
        print(f"    average daily dollar volume:      median ${df['day_$vol'].median():,.0f}")
        frac = 100 * 5000 / max(df["min_$vol"].median(), 1e-9)
        print(f"    a $5,000 order is {frac:,.0f}% of the dollar volume in the minute you would trade")
        print("  This is the odd-lot problem in a new costume: the filings that arrive while the market is")
        print("  open belong to names too small for anyone larger to bother with — and too small for the")
        print("  price you see to survive your own order.")


if __name__ == "__main__":
    print("=" * 110)
    print("8-K REACTION STUDY — is the move already gone by the time the free feed shows you the filing?")
    print("=" * 110)
    study()
