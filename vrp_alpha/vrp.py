"""
vrp.py — the signal, and the three statistical traps that decide whether the answer means anything.

TRAP 1: THE BRIEF'S VRP IS NOT THE VARIANCE RISK PREMIUM, AND IT IS NOT TRADEABLE. The brief defines
VRP_t = VIX_t - RV_21,t with RV backward-looking. That subtracts what ALREADY happened from what is
expected NEXT. It is reliably positive for a reason unrelated to any risk premium: volatility is strongly
mean-reverting and clusters, so after a calm month VIX sits above trailing RV simply because VIX prices
the possibility of the calm ending. Nobody can trade it, because there is no instrument that pays you
trailing realized variance. The premium that actually pays is

    VRP_fwd(t) = VIX_t - RV(t -> t+21)

i.e. implied against the realized volatility that FOLLOWS. Both are computed below, and the relationship
between them is the most important measurement in this file -- but it is not the one you would guess. Their
MEANS are nearly identical, +3.73 against +3.74 vol points since 2010, because realized volatility is
stationary over a long sample so its trailing and forward averages coincide. The brief's definition
therefore gets the average size of the premium right, by accident.

What it does not get is any information about WHEN the premium is earned. The correlation between the two
series is +0.003. They are orthogonal. So the backward-looking spread is not a noisy version of the
tradeable one; it is an unrelated quantity that happens to share a mean, and conditioning on it -- which is
exactly what the brief's "trade when VRP > 0" filter does -- conditions on a variable independent of the
payoff. That is why the filter cannot work, and it is a stronger objection than a difference in magnitude
would have been.

TRAP 2: OVERLAPPING WINDOWS INFLATE EVERY T-STAT BY ABOUT sqrt(21). VRP_fwd computed daily uses 21-day
forward windows, so consecutive observations share 20 of their 21 days. The 4,000 daily observations
contain roughly 190 independent months. A naive t-stat on the daily series overstates significance by
about sqrt(21) ~ 4.6x. This is the same clustering pathology as the S&P announcement batches, in new
clothing. Two corrections are applied and both are reported: Newey-West standard errors with 21+ lags on
the overlapping series, and a clean non-overlapping monthly series which needs no correction at all.

TRAP 3: THE P&L IS QUADRATIC IN REALIZED VOL, SO THE AVERAGE SPREAD OVERSTATES THE AVERAGE PROFIT. VIX is
a variance swap rate, not a volatility forecast: VIX^2/10^4 is the fair strike for annualized variance. A
short variance swap pays K^2 - RV^2, not K - RV. Because that is convex in RV, a vol spike costs
disproportionately more than a calm month pays. Quoting the premium in volatility points and then assuming
you earn it linearly is exactly how short-vol strategies are made to look safe on paper. Both are computed:
the linear spread (what a delta-hedged straddle approximately earns) and the variance-swap P&L in
vega-normalized vol points, (K^2 - RV^2) / 2K, which is what the instrument matching VIX actually pays.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WIN = 21          # trading days ~ the 30 calendar days VIX spans
ANN = 252


def realized_vol(spx: pd.Series, win: int = WIN, forward: bool = False) -> pd.Series:
    """RV = sqrt(ANN/N * sum r^2) on log returns, as the brief specifies (no demeaning).

    forward=False gives the trailing window ending at t (known at t).
    forward=True gives the window t+1..t+win, which is NOT known at t and is the strategy's payoff.
    """
    r = np.log(spx / spx.shift(1))
    sq = r.pow(2)
    if forward:
        # shift(-1) first so the window starts the day AFTER t: no same-day leakage into the signal.
        s = sq.shift(-1).rolling(win).sum().shift(-(win - 1))
    else:
        s = sq.rolling(win).sum()
    return (np.sqrt(ANN / win * s) * 100).rename("rv")


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Attach trailing RV, forward RV, both VRP definitions, and the variance-swap P&L."""
    out = df.copy()
    out["rv_bwd"] = realized_vol(out.spx, forward=False)
    out["rv_fwd"] = realized_vol(out.spx, forward=True)
    out["vrp_bwd"] = out.vix - out.rv_bwd          # the brief's spelling: not tradeable
    out["vrp_fwd"] = out.vix - out.rv_fwd          # the premium that actually pays
    # Variance-swap P&L per unit VEGA notional, in vol points. K and RV in percent, so the /100s cancel.
    # This is the correct payoff of the instrument VIX represents, and it is convex in realized vol.
    out["vswap_bp"] = (out.vix ** 2 - out.rv_fwd ** 2) / (2 * out.vix)
    return out


def newey_west_t(x: np.ndarray, lags: int = WIN) -> tuple[float, float]:
    """t-stat of the mean under Newey-West/Bartlett HAC errors. Returns (mean, t).

    The naive standard error assumes independent observations. With 21-day overlapping windows the series
    is autocorrelated out to lag 20 by construction, so the Bartlett kernel is summed to `lags` to absorb
    it. This is the honest t-stat for the daily series; the non-overlapping monthly series is the check
    that does not rely on getting the kernel right.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 30:
        return (np.nan, np.nan)
    e = x - x.mean()
    g0 = e @ e / n
    var = g0
    for L in range(1, min(lags, n - 1) + 1):
        w = 1.0 - L / (lags + 1.0)                 # Bartlett weight, guarantees a positive variance
        var += 2.0 * w * (e[L:] @ e[:-L]) / n
    se = np.sqrt(max(var, 1e-18) / n)
    return (x.mean(), x.mean() / se)


def monthly(s: pd.Series, offset: int = 0) -> pd.Series:
    """Non-overlapping 21-trading-day samples of a daily series. No windows share a day.

    offset shifts which day of the month the sample starts on. Reporting the spread across all 21 offsets
    is the cheapest guard against a result that exists only because of where the grid happened to land.
    """
    v = s.dropna()
    return v.iloc[offset::WIN]


def describe(x: np.ndarray, label: str, lags: int = WIN, unit: str = "vol pts") -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 30:
        print(f"  {label:38} n={len(x):5}  too few")
        return {}
    m, t = newey_west_t(x, lags)
    print(f"  {label:38} n={len(x):5}  mean {m:+7.2f} {unit}  HAC t {t:+6.2f}  "
          f"median {np.median(x):+7.2f}  >0 {100*(x>0).mean():4.0f}%  sd {x.std(ddof=1):6.2f}  "
          f"min {x.min():+8.2f}")
    return {"n": len(x), "mean": m, "t": t, "median": float(np.median(x)),
            "pos": float((x > 0).mean()), "sd": float(x.std(ddof=1)), "min": float(x.min())}
