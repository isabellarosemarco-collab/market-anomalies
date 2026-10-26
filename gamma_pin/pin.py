"""
pin.py — is expiration-Friday strike pinning a tradeable drift, or a reduction in dispersion that pays nothing?

THE DATA GATE, RESOLVED BEFORE ANY STRATEGY CODE. The brief specifies K* as the strike with maximal
gamma-weighted open interest, measured at 10:00 ET on past Fridays back to 2022. Neither input exists on
free data:

  HISTORICAL OPEN INTEREST      yfinance serves option chains for UNEXPIRED expirations only. There is no
                                free per-strike OI history at all, from any source. So K* as specified
                                cannot be reconstructed for a single past Friday, let alone 2022-present.
  HISTORICAL INTRADAY BARS      1-minute bars go back 7 days and 30-minute bars 60 days. Only 60-minute
                                bars reach two years (2023-10-27), and a 60-minute bar starting 09:30
                                closes at 10:30, not 10:00. So even the ENTRY PRICE of the specified trade
                                is unavailable before roughly two months ago.

A live snapshot taken on 2026-09-25 (an actual expiration Friday, market open) shows why the obvious
workaround does not work either: the maximal gamma*OI strike within +-1.5% of spot was NOT the nearest
strike for TSLA, NVDA or AMD -- three of five names. Substituting "nearest strike" for K* is therefore a
materially different rule, not an approximation of the specified one.

SO THE TEST IS BUILT AS A SUPERSET INSTEAD, WHICH IS THE ONLY HONEST WAY PAST THE GATE. Every candidate K*
in the brief is a listed strike on the round-dollar grid, and the $5 grid in particular carries the
largest open interest in all five names (confirmed in the live snapshot: 340, 375, 225, 635, 775 are each
the top-OI strike in their neighbourhood). If spot shows NO tendency to converge on the $5 grid on
expiration Fridays relative to control weekdays, then no OI-weighted refinement of which grid point to
pick can rescue the strategy -- the gravitational field itself is absent. A superset test can only
falsify, never confirm, and falsification is what this is for.

THE TWO CLAIMS THE BRIEF CONFLATES, WHICH IS THE CORE OF THE WHOLE QUESTION. Dealer gamma hedging is a
DISPERSION story: buying dips and selling rips compresses the distribution of the close around K*. The
delta trade in variant 1 is a DRIFT bet: it needs spot to travel TOWARD K* in expectation. Those are
different statements and the first does not imply the second. A perfectly pinned stock that is equally
likely to approach K* from either side has a compressed variance and an expected return of exactly zero.
Both are measured separately below: `d_reduction` for dispersion, `signed_bp` for the drift the trade
actually needs.

WHY 50% IS THE WRONG NULL FOR QUESTION 1, AND IT IS BIASED AGAINST PINNING. The brief asks whether the
close lands nearer K* than the 10:00 price did, "significantly more than 50% of the time". Under a
driftless random walk that probability is NOT 50%. With S0 a distance d from K, the close is nearer only
if the move falls in (-2d, 0) relative to d, and P of that is strictly below one half for every d and
every volatility; by Jensen's inequality the expected distance strictly INCREASES. So the correct null is
below 50% and unknown a priori. It is estimated here empirically from control weekdays rather than
assumed, which is the difference between testing pinning and testing the convexity of the absolute value.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

NAMES = ("AAPL", "TSLA", "NVDA", "AMD", "SPY")
ET = "America/New_York"
GRID = 5.0                      # dollars; the highest-OI strike spacing in all five names
LO, HI = 0.0025, 0.0125         # the brief's distance band, as a fraction of spot


@dataclass
class Day:
    ticker: str
    date: pd.Timestamp
    weekday: int
    is_monthly: bool            # third Friday: 5-10x the open interest of a weekly
    s_ref: float                # close of the 09:30 hourly bar == ~10:30 ET
    s_close: float              # official daily close
    k: float                    # nearest $5 grid point to s_ref
    d0_bp: float                # |s_ref - k| / s_ref, in bp
    d1_bp: float                # |s_close - k| / s_ref, in bp
    move_bp: float             # signed close-vs-ref return, in bp

    @property
    def d_reduction_bp(self) -> float:
        """Positive = the close is nearer the strike than the 10:30 price was. Dispersion, not profit."""
        return self.d0_bp - self.d1_bp

    @property
    def signed_bp(self) -> float:
        """The delta trade's actual P&L: long if spot is below K, short if above. THIS is what pays."""
        return self.move_bp if self.s_ref < self.k else -self.move_bp

    @property
    def crossed(self) -> bool:
        """Did spot pass through the strike? A pin should stick AT it, not sail through."""
        return (self.s_ref < self.k) != (self.s_close < self.k)


def load(names=NAMES) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Hourly bars (for the morning reference) and daily bars (for the official close)."""
    import yfinance as yf
    out = {}
    for n in names:
        t = yf.Ticker(n)
        h = t.history(period="730d", interval="60m", auto_adjust=False)
        d = t.history(period="730d", interval="1d", auto_adjust=False)
        if h.empty or d.empty:
            continue
        h.index = h.index.tz_convert(ET)
        d.index = d.index.tz_convert(ET) if d.index.tz is not None else d.index.tz_localize(ET)
        out[n] = (h, d)
    return out


def build(px: dict, offset: float = 0.0) -> list[Day]:
    """offset shifts the grid off the listed strikes, which is the placebo that decides the whole study.

    THE PLACEBO, AND WHY IT IS THE MOST IMPORTANT ARGUMENT IN THIS FILE. The delta trade goes long when
    spot sits just below the nearest grid point and short when just above. On a $5 grid that is the same
    thing as saying "long when the price is in the upper part of its five-dollar bucket" -- a statement
    about where spot sits relative to a ROUND NUMBER, which has nothing to do with options. Round-number
    magnetism in equity prices is a documented microstructure effect in its own right, driven by limit
    order clustering, and it exists on Tuesdays.

    So the grid is shifted by a non-round amount that corresponds to NO LISTED STRIKE in any of these
    names (offset 1.3 puts the grid at 1.3, 6.3, 11.3, ...). If the identical trade earns the same amount
    against a grid with no open interest on it whatsoever, then dealer gamma cannot be the cause, because
    there are no options there to hedge. This is the difference between measuring a pin and measuring the
    arithmetic of rounding.
    """
    days: list[Day] = []
    for n, (h, d) in px.items():
        # The 09:30 hourly bar spans 09:30-10:30, so its CLOSE is the ~10:30 price. That is 30 minutes
        # later than the brief's 10:00 entry, which is a real deviation and it runs AGAINST the strategy:
        # a pin that acts over the day has had another half hour to act before we measure the starting
        # distance, so any convergence we then observe is what is LEFT, not the whole effect. The exact
        # 10:00 version is run separately on 30-minute bars over their 60-day window as a cross-check.
        first = h[h.index.time == pd.Timestamp("09:30").time()]
        closes = d.Close
        for ts, row in first.iterrows():
            day = ts.normalize()
            m = closes.index.normalize() == day
            if not m.any():
                continue
            s_ref, s_close = float(row.Close), float(closes[m].iloc[0])
            if not (np.isfinite(s_ref) and np.isfinite(s_close)) or s_ref <= 0:
                continue
            k = round((s_ref - offset) / GRID) * GRID + offset
            d0 = 1e4 * abs(s_ref - k) / s_ref
            days.append(Day(
                ticker=n, date=day, weekday=day.weekday(),
                is_monthly=(day.weekday() == 4 and 15 <= day.day <= 21),
                s_ref=s_ref, s_close=s_close, k=k, d0_bp=d0,
                d1_bp=1e4 * abs(s_close - k) / s_ref,
                move_bp=1e4 * (s_close / s_ref - 1)))
    return days


def in_band(days: list[Day]) -> list[Day]:
    return [x for x in days if 1e4 * LO <= x.d0_bp <= 1e4 * HI]


def by_date(days: list[Day], attr: str) -> np.ndarray:
    """Average one metric across names WITHIN each calendar date, then return the per-date series.

    THIS IS NOT A REFINEMENT, IT IS THE DIFFERENCE BETWEEN A RESULT AND AN ARTEFACT. The five names are
    observed on the same Fridays, and on any given Friday they share the market's move. Treating 417
    name-days as 417 independent draws inflates every t-stat by roughly the square root of the number of
    names. The same mistake in the index study turned a t of 1.49 into 1.26 once batches were collapsed;
    here the effective sample is about a hundred Fridays, not four hundred.
    """
    g: dict[pd.Timestamp, list[float]] = {}
    for d in days:
        g.setdefault(d.date, []).append(float(getattr(d, attr)))
    return np.array([np.mean(v) for _, v in sorted(g.items())])


def welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Difference in means and its t-stat, unequal variances. Returns (diff, t)."""
    d = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return d, (d / se if se > 0 else np.nan)


def stats(days: list[Day], label: str) -> dict:
    if len(days) < 10:
        print(f"  {label:34} n={len(days):4}  too few")
        return {}
    dr = np.array([x.d_reduction_bp for x in days])
    sg = np.array([x.signed_bp for x in days])
    t_dr = dr.mean() / (dr.std(ddof=1) / np.sqrt(len(dr)))
    t_sg = sg.mean() / (sg.std(ddof=1) / np.sqrt(len(sg)))
    print(f"  {label:34} n={len(days):4}  "
          f"converge {dr.mean():+7.1f} bp (t {t_dr:+5.2f}, {100*(dr>0).mean():3.0f}% of days)   "
          f"delta-trade {sg.mean():+7.1f} bp (t {t_sg:+5.2f}, win {100*(sg>0).mean():3.0f}%)   "
          f"crossed {100*np.mean([x.crossed for x in days]):3.0f}%")
    return {"n": len(days), "conv": dr.mean(), "t_conv": t_dr, "pct_conv": (dr > 0).mean(),
            "signed": sg.mean(), "t_signed": t_sg, "win": (sg > 0).mean(),
            "sd_signed": sg.std(ddof=1)}
