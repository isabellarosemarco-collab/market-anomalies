"""
strategy.py — the filtered short-variance harvest, with the tail modelled rather than assumed away.

WHY THE POSITION IS MARKED DAILY INSTEAD OF ONLY AT MATURITY, WHICH IS THE DIFFERENCE BETWEEN A REAL
DRAWDOWN AND A FICTIONAL ONE. A short variance swap settled only at expiry produces a monthly P&L series
whose worst observation is one bad month. That is not what happens to the account. When VIX goes from 15
to 82 in three weeks, the position is marked to a catastrophic loss immediately, margin is called, and the
trade is very often closed at the worst possible moment rather than held to maturity. So the swap is
marked every day using the standard decomposition of a variance swap's mid-life value:

    MTM_short(t) = K0^2 - (e/N) * RV_elapsed^2 - ((N-e)/N) * K_t^2

with e days elapsed of N, RV_elapsed the variance realized so far, and K_t the current fair strike for the
remaining tenor (VIX_t is used for it). The first term is what was sold, the second is variance already
delivered, the third is the cost of buying back the remaining exposure at today's price. This is what
makes the March 2020 and February 2018 marks appear at full size on the day they happened.

THE FILTER USES ONLY INFORMATION AVAILABLE WHEN THE DECISION IS MADE, WHICH HAS TO BE CHECKED AND NOT
ASSERTED. The brief's gate is VRP > 0 and VIX < 30, cutting to cash on inversion or on VIX above 30. Both
inputs -- VIX_t and trailing RV_t -- are known at the close of t, and entry is at the close of t. That is
implementable. Note what it is NOT: it is not a filter on forward RV, which is the quantity that decides
the P&L and is unknowable at t. A filter on forward RV would be a perfect short-vol strategy and a pure
look-ahead artifact, and it is the single easiest way to fake this result.

THE TAIL GATE CANNOT PROTECT THE POSITION YOU ALREADY HAVE, AND THIS IS THE CENTRAL POINT ABOUT VRP. The
gate stops you OPENING new swaps once VIX exceeds 30. It does nothing about the swap opened three days
earlier at VIX 14, which is the one that does the damage. Volatility gaps; it does not ramp politely
through the trigger. Both versions are therefore computed: the gate as specified, and a version that also
liquidates open positions when the gate trips, so the difference measures how much the gate can actually
save.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from vrp_alpha.vrp import ANN, WIN


def ladder(d: pd.DataFrame, vix_cap: float = 30.0, require_vrp: bool = True,
           cost_volpts: float = 0.5, liquidate_on_gate: bool = False) -> pd.DataFrame:
    """Open a 1/WIN-size short variance swap each day the gate passes; mark every open swap daily.

    Returns a daily frame with the portfolio's mark-to-market P&L in vega-normalized vol points.
    cost_volpts is charged once per swap opened, against its vega notional: it covers the variance-swap
    bid-ask (or the straddle spread plus delta-hedge slippage, which is the same order of magnitude).
    """
    r = np.log(d.spx / d.spx.shift(1))
    sq = r.pow(2).values
    vix = d.vix.values
    gate = (vix < vix_cap)
    if require_vrp:
        gate &= (d.vrp_bwd.values > 0)
    gate &= np.isfinite(d.vrp_bwd.values)

    n = len(d)
    pnl = np.zeros(n)
    opened = np.zeros(n, dtype=int)
    live: list[tuple[int, float, float]] = []   # (open_index, K0, cumulative realized variance so far)
    for i in range(1, n):
        # 1) Mark every live swap, then retire the matured ones.
        keep: list[tuple[int, float, float]] = []
        for (i0, k0, acc) in live:
            acc += sq[i]                                   # today's squared return joins realized variance
            e = i - i0
            rv_el2 = ANN / e * acc * 1e4                    # annualized variance so far, in (vol pts)^2
            if e >= WIN:
                mtm = k0 ** 2 - rv_el2                      # settled: pay realized, receive strike
            else:
                kt = vix[i]
                mtm = k0 ** 2 - (e / WIN) * rv_el2 - ((WIN - e) / WIN) * kt ** 2
            prev = 0.0
            if e > 1:
                e_p = e - 1
                rv_p2 = ANN / e_p * (acc - sq[i]) * 1e4
                kt_p = vix[i - 1]
                prev = k0 ** 2 - (e_p / WIN) * rv_p2 - ((WIN - e_p) / WIN) * kt_p ** 2
            # Daily change in the mark, converted to vega-normalized vol points and scaled to 1/WIN size.
            pnl[i] += (mtm - prev) / (2 * k0) / WIN
            if e < WIN and not (liquidate_on_gate and not gate[i]):
                keep.append((i0, k0, acc))
        live = keep
        # 2) Open a new swap if the gate passes. Cost is charged on the day of opening.
        if gate[i] and np.isfinite(vix[i]):
            live.append((i, vix[i], 0.0))
            opened[i] = 1
            pnl[i] -= cost_volpts / WIN
    out = pd.DataFrame({"pnl": pnl, "opened": opened, "gate": gate.astype(int)}, index=d.index)
    out["equity"] = out.pnl.cumsum()
    return out


def perf(x: pd.Series, per_year: float = ANN, label: str = "") -> dict:
    """Sharpe, max drawdown and Calmar on an additive P&L series.

    ALL THREE ARE LEVERAGE-INVARIANT, WHICH IS WHY NO MARGIN ASSUMPTION IS MADE ANYWHERE IN THIS FILE.
    A short variance swap has no natural capital base, so any "return on capital" is a leverage choice
    dressed up as a result -- and it is the knob that makes short-vol backtests print whatever Sharpe the
    author wants. Sharpe scales as mean/sd and Calmar as mean/maxDD; both are unchanged if the whole
    position is doubled. The comparison against SPY is therefore apples to apples without either series
    needing a notional.
    """
    x = x.dropna()
    if len(x) < 30:
        return {}
    mu, sd = x.mean() * per_year, x.std(ddof=1) * np.sqrt(per_year)
    eq = x.cumsum()
    dd = (eq - eq.cummax())
    mdd = float(dd.min())
    r = {"ann_mean": mu, "ann_sd": sd, "sharpe": mu / sd if sd else np.nan,
         "maxdd": mdd, "calmar": mu / abs(mdd) if mdd else np.nan,
         "worst_day": float(x.min()), "best_day": float(x.max()),
         "skew": float(pd.Series(x).skew()), "kurt": float(pd.Series(x).kurt())}
    if label:
        print(f"  {label:34} ann {r['ann_mean']:+8.2f}  sd {r['ann_sd']:7.2f}  Sharpe {r['sharpe']:+5.2f}  "
              f"maxDD {r['maxdd']:+9.2f}  Calmar {r['calmar']:+5.2f}  worst day {r['worst_day']:+8.2f}  "
              f"skew {r['skew']:+5.1f}  kurt {r['kurt']:7.1f}")
    return r


def worst_episodes(pnl: pd.Series, k: int = 6) -> pd.DataFrame:
    """The k worst 21-day stretches, which is where the entire question lives."""
    roll = pnl.rolling(WIN).sum().dropna()
    idx = roll.nsmallest(k).index
    return pd.DataFrame({"end": [i.date() for i in idx],
                         "21d_pnl": [round(float(roll[i]), 2) for i in idx]})
