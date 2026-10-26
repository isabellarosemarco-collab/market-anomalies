"""
diagnostics.py — structural checks that explain WHY the backtest lands where it does.

Two questions, both in-sample and labelled as such:

  1. Is the 1.5x ATR trailing stop specified for the strategy compatible with a
     mean-reversion entry? Sweeping the multiple isolates the exit design from the signal.
  2. Does the causal (expanding, shifted) time-of-day RVOL baseline actually matter, or
     would the conventional full-sample groupby mean have given the same answer? The gap
     between the two is the size of the lookahead bias that implementation avoids.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtester import run_backtest
from data_loader import load_universe
from signals import PARAMS, build_indicators


def exit_sensitivity(prep: dict) -> pd.DataFrame:
    rows = []
    for m in [1.5, 2.0, 3.0, 6.0, 12.0, 100.0]:
        res = run_backtest(prepared=prep, params={"atr_mult": m})
        met = res["metrics"]
        mix = res["trades"]["reason"].value_counts() if not res["trades"].empty else pd.Series(dtype=int)
        rows.append({
            "atr_mult": m, "trades": met["n_trades"], "sharpe": met["sharpe"],
            "profit_factor": met["profit_factor"], "ret_pct": met["total_return_pct"],
            "maxdd_pct": met["max_drawdown_pct"],
            "pct_trail_stop": round(100 * mix.get("trail_stop", 0) / max(met["n_trades"], 1), 1),
            "pct_z_target": round(100 * mix.get("z_target", 0) / max(met["n_trades"], 1), 1),
            "pct_time_stop": round(100 * mix.get("time_stop", 0) / max(met["n_trades"], 1), 1),
        })
    return pd.DataFrame(rows).set_index("atr_mult")


def _leaky_rvol(df: pd.DataFrame) -> pd.Series:
    """The buggy version: time-of-day mean/std over the WHOLE sample, including the future."""
    slot = df.index.time
    vol = df["volume"].astype(float)
    g = vol.groupby(slot)
    return (vol - g.transform("mean")) / g.transform("std").replace(0, np.nan)


def lookahead_audit(data: dict) -> pd.DataFrame:
    """Compare causal RVOL against the leaky full-sample variant."""
    rows = []
    causal = {s: build_indicators(d) for s, d in data.items()}

    leaky = {}
    for s, d in data.items():
        f = build_indicators(d).copy()
        lk = _leaky_rvol(d).shift(1)          # same one-bar lag, but a leaky baseline
        f["rvol"] = lk.reindex(f.index)
        leaky[s] = f.dropna(subset=["rvol"])

    for label, prep in (("causal (as built)", causal), ("leaky full-sample baseline", leaky)):
        m = run_backtest(prepared=prep, params={"atr_mult": 6.0})["metrics"]
        rows.append({"rvol_baseline": label, "trades": m["n_trades"],
                     "sharpe": m["sharpe"], "profit_factor": m["profit_factor"],
                     "ret_pct": m["total_return_pct"]})
    return pd.DataFrame(rows).set_index("rvol_baseline")


if __name__ == "__main__":
    data = load_universe(interval="60m", period="730d")
    prep = {s: build_indicators(d) for s, d in data.items()}

    print("\n=== 1. EXIT SENSITIVITY: ATR trailing-stop multiple (IN-SAMPLE) ===")
    print(f"    baseline entry params: z={PARAMS['z_entry']}, rvol={PARAMS['rvol_thr']}")
    print("    NOTE: atr_mult=100 is NOT a clean 'stop off' test. Stop distance drives")
    print("    position sizing, so a 100x ATR stop sizes almost every trade below one")
    print("    share and the entry is skipped. Read the 6-12x rows instead.\n")
    print(exit_sensitivity(prep).to_string(float_format=lambda x: f"{x:,.3f}"))

    print("\n=== 2. LOOKAHEAD AUDIT: RVOL time-of-day baseline (IN-SAMPLE) ===")
    print("    Any gap here is bias the causal implementation refuses to book.\n")
    print(lookahead_audit(data).to_string(float_format=lambda x: f"{x:,.3f}"))
