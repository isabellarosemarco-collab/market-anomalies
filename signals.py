"""
signals.py — VWAP dislocation Z-score, time-of-day RVOL, ATR, and entry triggers.

Lookahead discipline. Every indicator is built from data up to and including bar t, then
the whole indicator block is shifted forward one bar by `build_signals`. The backtester
reads row t and fills at bar t's OPEN, so the newest information any decision can contain
is bar t-1's close. Two places where lookahead normally sneaks into this kind of strategy
are handled explicitly:

  * The time-of-day volume baseline uses an EXPANDING mean/std over prior same-slot bars
    only (`.shift(1)` inside the groupby). A full-sample groupby mean would leak the
    future into every RVOL reading and is the single most common bug in RVOL strategies.
  * Rolling std of the VWAP spread is computed with min_periods equal to the full window,
    so early bars emit NaN rather than a std estimated from a handful of points.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Defaults; walk_forward.py overrides these per fold.
PARAMS = {
    "vwap_days": 20,      # VWAP / spread-std lookback, in trading days
    "z_entry": 2.0,       # |Z| threshold
    "rvol_thr": 2.5,      # RVOL sigma above time-of-day mean
    "atr_bars": 14,       # ATR lookback, in bars
    "atr_mult": 1.5,      # trailing stop multiple
    "extreme_bars": 5,    # "5-period" low/high rejection window
    "time_stop_days": 3,  # flat after N trading days
}


def bars_per_day(df: pd.DataFrame) -> int:
    n = df.index.normalize().nunique()
    return max(int(round(len(df) / max(n, 1))), 1)


def rolling_vwap(df: pd.DataFrame, window: int) -> pd.Series:
    """Volume-weighted average of the typical price over a rolling bar window."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = (tp * df["volume"]).rolling(window, min_periods=window).sum()
    v = df["volume"].rolling(window, min_periods=window).sum()
    return pv / v.replace(0, np.nan)


def vwap_zscore(df: pd.DataFrame, window: int) -> tuple[pd.Series, pd.Series]:
    """Z of (close - VWAP) against its own rolling dispersion."""
    vwap = rolling_vwap(df, window)
    spread = df["close"] - vwap
    sd = spread.rolling(window, min_periods=window).std(ddof=0)
    return spread / sd.replace(0, np.nan), vwap


def tod_rvol(df: pd.DataFrame, min_obs: int = 20) -> pd.Series:
    """Volume in sigma units above the same-time-of-day historical mean.

    Causal by construction: for each time-of-day slot the mean and std are expanding
    over that slot's PRIOR observations only.
    """
    slot = df.index.time
    vol = df["volume"].astype(float)
    g = vol.groupby(slot)
    mean = g.transform(lambda s: s.expanding(min_obs).mean().shift(1))
    std = g.transform(lambda s: s.expanding(min_obs).std(ddof=0).shift(1))
    return (vol - mean) / std.replace(0, np.nan)


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    """Wilder ATR."""
    pc = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - pc).abs(),
        (df["low"] - pc).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def build_indicators(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """OHLCV plus one-bar-lagged indicators, with no threshold applied.

    Split out from `build_signals` so the walk-forward grid can compute the expensive
    indicators once per symbol and then vary only the cheap threshold comparisons. The
    indicators here depend solely on vwap_days / atr_bars / extreme_bars, never on the
    entry thresholds being searched.
    """
    p = {**PARAMS, **(params or {})}
    bpd = bars_per_day(df)
    w = max(int(p["vwap_days"] * bpd), 5)

    ind = pd.DataFrame(index=df.index)
    ind["z"], ind["vwap"] = vwap_zscore(df, w)
    ind["rvol"] = tod_rvol(df)
    ind["atr"] = atr(df, int(p["atr_bars"]))
    ind["low_n"] = df["low"].rolling(int(p["extreme_bars"]), min_periods=int(p["extreme_bars"])).min()
    ind["high_n"] = df["high"].rolling(int(p["extreme_bars"]), min_periods=int(p["extreme_bars"])).max()
    ind["close_ref"] = df["close"]

    lagged = ind.shift(1)  # <- the single lookahead barrier

    out = df[["open", "high", "low", "close", "volume"]].copy()
    out = out.join(lagged)
    out["bars_per_day"] = bpd
    out["time_stop_bars"] = int(p["time_stop_days"] * bpd)
    return out.dropna(subset=["z", "rvol", "atr"])


def apply_entries(sig: pd.DataFrame, z_entry: float, rvol_thr: float) -> pd.DataFrame:
    """Add the entry booleans for one threshold pair. Cheap; safe to call in a grid loop."""
    vol_ok = sig["rvol"] > rvol_thr
    sig["long_entry"] = (sig["z"] < -z_entry) & vol_ok & (sig["close_ref"] > sig["low_n"])
    sig["short_entry"] = (sig["z"] > z_entry) & vol_ok & (sig["close_ref"] < sig["high_n"])
    return sig


def build_signals(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Indicators plus entry triggers for a single symbol."""
    p = {**PARAMS, **(params or {})}
    return apply_entries(build_indicators(df, p), p["z_entry"], p["rvol_thr"])


def signal_diagnostics(sig: pd.DataFrame) -> dict:
    return {
        "bars": len(sig),
        "long_signals": int(sig["long_entry"].sum()),
        "short_signals": int(sig["short_entry"].sum()),
        "pct_bars_signalling": round(
            100 * float((sig["long_entry"] | sig["short_entry"]).mean()), 3
        ),
        "z_abs_p99": round(float(sig["z"].abs().quantile(0.99)), 2),
        "rvol_p99": round(float(sig["rvol"].quantile(0.99)), 2),
    }


if __name__ == "__main__":
    from data_loader import load_universe

    data = load_universe(interval="60m", period="730d")
    rows = []
    for sym, df in data.items():
        d = signal_diagnostics(build_signals(df))
        d["symbol"] = sym
        rows.append(d)
    print(pd.DataFrame(rows).set_index("symbol").to_string())
