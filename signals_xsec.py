"""
signals_xsec.py — cross-sectional 12-1 momentum with inverse-volatility sizing.

Lookahead discipline. Formation happens on the last trading day of each month using data
through that day's close. The resulting weights are applied to returns starting the NEXT
trading day, which `backtester_xsec` enforces by shifting the weight matrix forward one bar.
Two subtler points:

  * The momentum score is (P[t-skip] / P[t-lookback]) - 1, so the most recent month is
    skipped deliberately to avoid short-term reversal contaminating the signal.
  * Eligibility is evaluated per formation date: a sector must have a non-NaN score, which
    requires `lookback` days of real history. XLC lists in 2018-06, so pre-2018 rebalances
    rank 8 sectors, not 9. Back-filling it would grant the strategy knowledge of a fund that
    did not exist.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PARAMS = {
    "lookback": 252,      # total momentum window, trading days
    "skip": 21,           # most recent month skipped
    "n_sel": 3,           # names per side
    "vol_window": 60,     # realized-vol window for risk parity
    "sma_regime": 200,    # SPY trend filter
    "use_regime": True,   # flat-to-cash when SPY is below its SMA
    "short_book": 0.5,    # 0.0 = long only; 0.5 = dollar neutral (gross 1.0)
    "long_book": 0.5,
    "require_positive": True,   # only hold longs with positive momentum
    "max_weight": 1.0,    # per-asset cap; 1.0 = uncapped
}


def build_panel(data: dict[str, pd.DataFrame], field: str = "close",
                bench: str = "SPY") -> pd.DataFrame:
    """Symbols to columns on a shared calendar. No cross-sectional fill: a symbol that had
    not listed yet stays NaN and is therefore ineligible.

    The calendar is pinned to the benchmark's sessions. Taking the union of 40+ symbols' indices
    instead would admit any date on which a single thinly-traded ETF printed a bar, inserting a
    row where every other name is NaN — and the next real bar would then register as a two-day
    move in a one-day slot, inflating realized vol.
    """
    s = {sym: df[field] for sym, df in data.items()}
    px = pd.DataFrame(s).sort_index()
    if bench in px:
        px = px.loc[px[bench].notna()]
    return px


def momentum_score(px: pd.DataFrame, lookback: int, skip: int) -> pd.DataFrame:
    return px.shift(skip) / px.shift(lookback) - 1.0


def realized_vol(px: pd.DataFrame, window: int) -> pd.DataFrame:
    """Annualised daily-return volatility."""
    return px.pct_change().rolling(window, min_periods=window).std(ddof=0) * np.sqrt(252)


def formation_dates(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each month present in the index."""
    s = pd.Series(idx, index=idx)
    return pd.DatetimeIndex(s.groupby([idx.year, idx.month]).last().values)


def regime_ok(spy: pd.Series, window: int) -> pd.Series:
    sma = spy.rolling(window, min_periods=window).mean()
    return spy > sma


def capped_inverse_vol(vols: pd.Series, budget: float, cap: float) -> pd.Series:
    """Inverse-volatility weights scaled to `budget`, with no single name above `cap`.

    Water-filling: allocate proportionally, pin any name that breaches the cap at the cap, then
    redistribute the remainder across the names still free. Naive clipping would silently shrink
    total exposure; this preserves it. If cap * n_names < budget the shortfall is left
    unallocated on purpose — the caller parks it in cash rather than forcing concentration.
    """
    vols = vols.replace([np.inf, -np.inf], np.nan).dropna()
    vols = vols[vols > 0]
    w = pd.Series(0.0, index=vols.index, dtype=float)
    if vols.empty or budget <= 0:
        return w

    free = list(vols.index)
    remaining = budget
    while free and remaining > 1e-12:
        inv = 1.0 / vols[free]
        prop = inv / inv.sum() * remaining
        over = prop[prop > cap + 1e-12]
        if over.empty:
            w[prop.index] = w[prop.index].values + prop.values
            break
        w[over.index] = cap
        free = [n for n in free if n not in set(over.index)]
        remaining = budget - float(w.sum())
    return w


def target_weights(
    px: pd.DataFrame,
    sectors: list[str],
    spy: pd.Series,
    params: dict | None = None,
) -> pd.DataFrame:
    """Weight matrix indexed by formation date, columns = sectors + CASH."""
    p = {**PARAMS, **(params or {})}
    mom = momentum_score(px[sectors], p["lookback"], p["skip"])
    vol = realized_vol(px[sectors], p["vol_window"])
    reg = regime_ok(spy, p["sma_regime"])
    fdates = formation_dates(px.index)

    cols = sectors + ["CASH"]
    W = pd.DataFrame(0.0, index=fdates, columns=cols)

    for d in fdates:
        if d not in mom.index:
            continue
        if p["use_regime"] and not bool(reg.get(d, False)):
            W.loc[d, "CASH"] = 1.0
            continue

        m = mom.loc[d].dropna()
        v = vol.loc[d].reindex(m.index).dropna()
        m = m.reindex(v.index).dropna()
        # A long/short book needs both tails; long-only only needs the top.
        min_names = p["n_sel"] * (2 if p["short_book"] > 0 else 1)
        if len(m) < min_names:
            W.loc[d, "CASH"] = 1.0
            continue

        ranked = m.sort_values(ascending=False)
        longs = ranked.head(p["n_sel"])
        if p["require_positive"]:
            longs = longs[longs > 0]
        shorts = ranked.tail(p["n_sel"]) if p["short_book"] > 0 else ranked.iloc[0:0]

        def rp(names: pd.Index, budget: float, sign: float) -> None:
            """Capped inverse-volatility weights scaled to a signed dollar budget."""
            if len(names) == 0 or budget <= 0:
                return
            w = capped_inverse_vol(v.reindex(names), budget, p["max_weight"]) * sign
            if w.empty:
                return
            W.loc[d, w.index] = W.loc[d, w.index].astype(float) + w.values

        rp(longs.index, p["long_book"], +1.0)
        rp(shorts.index, p["short_book"], -1.0)

        # Unallocated long budget (e.g. momentum was negative everywhere) parks in cash.
        used_long = W.loc[d, sectors].clip(lower=0).sum()
        W.loc[d, "CASH"] = max(p["long_book"] - used_long, 0.0)

    return W


def signal_summary(W: pd.DataFrame, sectors: list[str]) -> dict:
    inv = W[sectors]
    return {
        "rebalances": len(W),
        "months_in_cash": int((W["CASH"] >= 0.999).sum()),
        "avg_gross_exposure": round(float(inv.abs().sum(axis=1).mean()), 3),
        "avg_net_exposure": round(float(inv.sum(axis=1).mean()), 3),
        "avg_n_long": round(float((inv > 0).sum(axis=1).mean()), 2),
        "avg_n_short": round(float((inv < 0).sum(axis=1).mean()), 2),
    }
