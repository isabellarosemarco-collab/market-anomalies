"""
backtester.py — portfolio backtest with explicit execution frictions.

Execution model. Signals are read from row t (already lagged one bar inside signals.py)
and filled at bar t's OPEN. Within a bar the order of operations is deliberate:

  1. Exits knowable at the open (Z reverted to 0, time stop) fill at the open.
  2. Otherwise the ATR trailing stop is tested against the bar's range and, if breached,
     fills at the stop price. Testing the stop before the open-based exits would let a
     position escape at a price it could not have reached yet.
  3. New entries fill at the open, and only if the symbol is flat.

Frictions: 0.02% slippage applied adversely to every fill, plus $0.005/share each side.
Sizing risks a fixed fraction of live equity against the initial ATR stop distance, hard
capped at 2% risk, 25% of equity notional per position, and 100% gross exposure — without
those caps a tight ATR produces share counts that silently imply heavy leverage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from signals import PARAMS, apply_entries, build_indicators, build_signals

COSTS = {
    "slippage_pct": 0.0002,     # 0.02% per fill
    "fee_per_share": 0.005,     # $/share each side
    "risk_frac": 0.01,          # fraction of equity risked per position
    "max_risk_frac": 0.02,      # hard ceiling
    "max_pos_notional": 0.25,   # per-position notional cap, fraction of equity
    "max_gross": 1.00,          # portfolio gross exposure cap
    "start_equity": 10_000.0,
}

TRADING_DAYS = 252


def _fill(price: float, side: int, slip: float) -> float:
    """Adverse slippage: buys fill higher, sells fill lower."""
    return price * (1.0 + slip) if side > 0 else price * (1.0 - slip)


def run_backtest(
    data: dict[str, pd.DataFrame] | None = None,
    params: dict | None = None,
    costs: dict | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    prepared: dict[str, pd.DataFrame] | None = None,
) -> dict:
    """Backtest the universe. Pass `prepared` (output of signals.build_indicators) to reuse
    precomputed indicators across a parameter grid; thresholds are then applied per call."""
    c = {**COSTS, **(costs or {})}
    slip, fee = c["slippage_pct"], c["fee_per_share"]
    cfg = {**PARAMS, **(params or {})}

    frames = []
    source = prepared if prepared is not None else data
    for sym, df in source.items():
        if prepared is not None:
            sig = apply_entries(df.copy(), cfg["z_entry"], cfg["rvol_thr"])
        else:
            sig = build_signals(df, cfg)
        if start is not None:
            sig = sig[sig.index >= start]
        if end is not None:
            sig = sig[sig.index < end]
        if len(sig) < 50:
            continue
        sig = sig.assign(symbol=sym)
        frames.append(sig)
    if not frames:
        return _empty_result(c["start_equity"])

    allbars = pd.concat(frames).sort_index(kind="mergesort")
    equity = c["start_equity"]
    cash = equity
    pos: dict[str, dict] = {}
    trades: list[dict] = []
    eq_t: list[pd.Timestamp] = []
    eq_v: list[float] = []
    last_px: dict[str, float] = {}

    for ts, grp in allbars.groupby(level=0, sort=False):
        for r in grp.itertuples():
            sym = r.symbol
            last_px[sym] = r.close

            # ---------- manage an open position ----------
            if sym in pos:
                p = pos[sym]
                side = p["side"]
                p["bars"] += 1
                exit_px = None
                reason = None

                reverted = (side > 0 and r.z >= 0.0) or (side < 0 and r.z <= 0.0)
                if reverted:
                    exit_px, reason = _fill(r.open, -side, slip), "z_target"
                elif p["bars"] >= p["time_stop"]:
                    exit_px, reason = _fill(r.open, -side, slip), "time_stop"
                else:
                    if side > 0:
                        p["best"] = max(p["best"], r.high)
                        stop = p["best"] - p["stop_dist"]
                        if r.low <= stop:
                            exit_px, reason = _fill(min(stop, r.open), -side, slip), "trail_stop"
                    else:
                        p["best"] = min(p["best"], r.low)
                        stop = p["best"] + p["stop_dist"]
                        if r.high >= stop:
                            exit_px, reason = _fill(max(stop, r.open), -side, slip), "trail_stop"

                if exit_px is not None:
                    sh = p["shares"]
                    gross = side * (exit_px - p["entry_px"]) * sh
                    exit_fee = fee * sh
                    # Basis is held as escrow at entry for both sides, so releasing it and
                    # banking gross P&L nets to exactly (gross - fees) in cash either way.
                    cash += p["cost_basis"] + gross - exit_fee
                    trades.append({
                        "symbol": sym, "side": "long" if side > 0 else "short",
                        "entry_time": p["entry_time"], "exit_time": ts,
                        "entry_px": p["entry_px"], "exit_px": exit_px,
                        "shares": sh, "bars_held": p["bars"],
                        "gross_pnl": gross,
                        "fees": p["entry_fee"] + exit_fee,
                        "pnl": gross - p["entry_fee"] - exit_fee,
                        "reason": reason,
                        "z_at_entry": p["z_entry"],
                        "rvol_at_entry": p["rvol_entry"],
                    })
                    del pos[sym]

            # ---------- consider a new entry ----------
            if sym not in pos and (r.long_entry or r.short_entry):
                side = 1 if r.long_entry else -1
                stop_dist = float(r.atr) * float(cfg["atr_mult"])
                if not np.isfinite(stop_dist) or stop_dist <= 0:
                    continue
                entry_px = _fill(r.open, side, slip)

                risk_cap = min(c["risk_frac"], c["max_risk_frac"]) * equity
                sh = risk_cap / stop_dist
                sh = min(sh, c["max_pos_notional"] * equity / entry_px)

                gross_open = sum(
                    abs(q["shares"]) * last_px.get(s, q["entry_px"]) for s, q in pos.items()
                )
                room = max(c["max_gross"] * equity - gross_open, 0.0)
                sh = min(sh, room / entry_px)
                sh = float(np.floor(sh))
                if sh < 1:
                    continue

                entry_fee = fee * sh
                basis = sh * entry_px
                cash -= basis + entry_fee
                pos[sym] = {
                    "side": side, "shares": sh, "entry_px": entry_px,
                    "entry_time": ts, "bars": 0, "best": r.high if side > 0 else r.low,
                    "stop_dist": stop_dist, "entry_fee": entry_fee,
                    "cost_basis": basis, "time_stop": int(r.time_stop_bars),
                    "z_entry": float(r.z), "rvol_entry": float(r.rvol),
                }

        # ---------- mark to market ----------
        mtm = 0.0
        for s, q in pos.items():
            px = last_px.get(s, q["entry_px"])
            mtm += q["cost_basis"] + q["side"] * (px - q["entry_px"]) * q["shares"]
        equity = cash + mtm
        eq_t.append(ts)
        eq_v.append(equity)

    eq = pd.Series(eq_v, index=pd.DatetimeIndex(eq_t), name="equity")
    tr = pd.DataFrame(trades)
    bpd = int(allbars["bars_per_day"].iloc[0])
    return {"equity": eq, "trades": tr, "metrics": compute_metrics(eq, tr, bpd),
            "bars_per_day": bpd}


def _empty_result(start_equity: float) -> dict:
    return {"equity": pd.Series(dtype=float), "trades": pd.DataFrame(),
            "metrics": compute_metrics(pd.Series(dtype=float), pd.DataFrame(), 7),
            "bars_per_day": 7}


def compute_metrics(eq: pd.Series, trades: pd.DataFrame, bars_per_day: int) -> dict:
    """Annualised risk/return stats. Sharpe and Sortino are computed from per-bar equity
    returns and scaled by sqrt(bars per year); Sortino's denominator is downside deviation."""
    n_per_year = bars_per_day * TRADING_DAYS
    out = {
        "n_trades": 0, "total_return_pct": 0.0, "sharpe": 0.0, "sortino": 0.0,
        "max_drawdown_pct": 0.0, "profit_factor": 0.0, "win_rate_pct": 0.0,
        "gross_profit": 0.0, "gross_loss": 0.0, "total_fees": 0.0, "final_equity": 0.0,
    }
    if eq.empty or len(eq) < 3:
        return out

    ret = eq.pct_change().dropna()
    sd = ret.std(ddof=0)
    out["sharpe"] = float(ret.mean() / sd * np.sqrt(n_per_year)) if sd > 0 else 0.0
    dn = ret[ret < 0].std(ddof=0)
    out["sortino"] = float(ret.mean() / dn * np.sqrt(n_per_year)) if dn and dn > 0 else 0.0
    dd = eq / eq.cummax() - 1.0
    out["max_drawdown_pct"] = float(dd.min() * 100)
    out["total_return_pct"] = float((eq.iloc[-1] / eq.iloc[0] - 1) * 100)
    out["final_equity"] = float(eq.iloc[-1])

    if not trades.empty:
        wins = trades.loc[trades["pnl"] > 0, "pnl"]
        losses = trades.loc[trades["pnl"] < 0, "pnl"]
        gp, gl = float(wins.sum()), float(abs(losses.sum()))
        out["n_trades"] = int(len(trades))
        out["gross_profit"], out["gross_loss"] = gp, gl
        out["profit_factor"] = float(gp / gl) if gl > 0 else (np.inf if gp > 0 else 0.0)
        out["win_rate_pct"] = float(100 * len(wins) / len(trades))
        out["total_fees"] = float(trades["fees"].sum())
    return out


if __name__ == "__main__":
    from data_loader import load_universe

    data = load_universe(interval="60m", period="730d")
    res = run_backtest(data)
    m = res["metrics"]
    print("\n=== FULL-SAMPLE BACKTEST (in-sample; not evidence of edge) ===")
    for k, v in m.items():
        print(f"  {k:>22}: {v:,.3f}" if isinstance(v, float) else f"  {k:>22}: {v}")
    if not res["trades"].empty:
        print("\nExit reason mix:")
        print(res["trades"]["reason"].value_counts().to_string())
