"""
backtester_xsec.py — monthly-rebalanced portfolio simulation on daily bars.

Execution model. Weights formed at month-end close are held from the NEXT trading day until
the following formation date. The weight matrix is forward-filled across the month and then
shifted one day, so no day's return is ever earned by a position established with that same
day's information.

Frictions:
  * Slippage 0.05% charged on turnover, i.e. on sum|w_new - w_old| at each rebalance.
  * Short borrow accrued daily at an annual rate on the absolute short notional.
  * Cash sits in BIL and earns BIL's actual total return, not an assumed zero.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtester import compute_metrics
from signals_xsec import PARAMS, build_panel, target_weights

COSTS = {
    "slippage_pct": 0.0005,      # 0.05% per unit of turnover
    "borrow_annual": 0.010,      # 1.0%/yr on short notional
    "start_equity": 10_000.0,
}


def run_xsec_backtest(
    data: dict[str, pd.DataFrame],
    sectors: list[str],
    params: dict | None = None,
    costs: dict | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> dict:
    p = {**PARAMS, **(params or {})}
    c = {**COSTS, **(costs or {})}

    px = build_panel(data)
    spy, bil = px["SPY"], px.get("BIL")
    W = target_weights(px, sectors, spy, p)

    # Daily returns for every tradable column; CASH earns BIL.
    rets = px[sectors].pct_change()
    rets["CASH"] = bil.pct_change() if bil is not None else 0.0
    cols = sectors + ["CASH"]

    # Hold weights from the day AFTER formation until the next formation.
    held = W.reindex(px.index).ffill().shift(1).fillna(0.0)[cols]

    # Turnover on formation days only.
    turn = pd.Series(0.0, index=px.index)
    prev = pd.Series(0.0, index=cols)
    for d in W.index:
        cur = W.loc[d, cols]
        turn.loc[d] = float((cur - prev).abs().sum())
        prev = cur

    if start is not None:
        mask = px.index >= start
        held, rets, turn = held[mask], rets[mask], turn[mask]
    if end is not None:
        mask = held.index < end
        held, rets, turn = held[mask], rets[mask], turn[mask]
    if len(held) < 60:
        return {"equity": pd.Series(dtype=float), "trades": pd.DataFrame(),
                "metrics": compute_metrics(pd.Series(dtype=float), pd.DataFrame(), 1),
                "weights": W, "net_ret": pd.Series(dtype=float),
                "gross_ret": pd.Series(dtype=float), "cost_drag": 0.0}

    gross_ret = (held * rets.reindex(held.index)[cols]).sum(axis=1)
    borrow = held[sectors].clip(upper=0).abs().sum(axis=1) * (c["borrow_annual"] / 252.0)
    slip = turn.reindex(held.index).fillna(0.0) * c["slippage_pct"]
    net_ret = gross_ret - borrow - slip

    eq = c["start_equity"] * (1.0 + net_ret).cumprod()
    eq.name = "equity"

    trades = _position_trades(W, px, sectors, eq, c["start_equity"])
    return {"equity": eq, "trades": trades,
            "metrics": compute_metrics(eq, trades, bars_per_day=1),
            "weights": W,
            # net_ret is exported so the walk-forward can stitch folds from returns rather than
            # from equity. Chaining eq.pct_change() would discard each segment's first-day return.
            "net_ret": net_ret, "gross_ret": gross_ret,
            "cost_drag": float((borrow + slip).sum())}


def _position_trades(
    W: pd.DataFrame, px: pd.DataFrame, sectors: list[str],
    eq: pd.Series, start_equity: float,
) -> pd.DataFrame:
    """One row per held sector position per month, for win-rate and profit-factor stats.

    P&L is weight x holding-period return x equity at formation, which attributes the month's
    result to each leg. It excludes the cash leg, whose return is not a bet.
    """
    rows = []
    fdates = list(W.index)
    for i, d in enumerate(fdates[:-1]):
        nxt = fdates[i + 1]
        cap = float(eq.reindex(eq.index[eq.index <= d]).iloc[-1]) if (eq.index <= d).any() else start_equity
        for s in sectors:
            w = float(W.loc[d, s])
            if abs(w) < 1e-9:
                continue
            try:
                p0 = float(px[s].loc[d])
                p1 = float(px[s].reindex(px.index[px.index <= nxt]).iloc[-1])
            except Exception:
                continue
            if not (np.isfinite(p0) and np.isfinite(p1)) or p0 <= 0:
                continue
            r = p1 / p0 - 1.0
            rows.append({
                "formation": d, "exit": nxt, "symbol": s,
                "side": "long" if w > 0 else "short",
                "weight": w, "asset_ret_pct": r * 100,
                "pnl": w * r * cap, "fees": 0.0,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from data_loader import BENCH, CASH, SECTORS, load_daily
    from signals_xsec import signal_summary

    data = load_daily(SECTORS + [BENCH, CASH])
    sectors = [s for s in SECTORS if s in data]
    res = run_xsec_backtest(data, sectors, start=pd.Timestamp("2015-01-01"))

    print("\n=== PORTFOLIO CONSTRUCTION (full sample, baseline params) ===")
    for k, v in signal_summary(res["weights"], sectors).items():
        print(f"  {k:>22}: {v}")

    print("\n=== FULL-SAMPLE RESULT 2015-present (IN-SAMPLE) ===")
    for k, v in res["metrics"].items():
        print(f"  {k:>22}: {v:,.3f}" if isinstance(v, float) else f"  {k:>22}: {v}")
    print(f"  {'cumulative cost drag':>22}: {res['cost_drag']*100:,.2f}% of equity-return")

    # Benchmark: buy and hold SPY over the identical window.
    spy = build_panel(data)["SPY"]
    spy = spy[spy.index >= pd.Timestamp("2015-01-01")]
    beq = 10_000 * (1 + spy.pct_change().fillna(0)).cumprod()
    bm = compute_metrics(beq, pd.DataFrame(), 1)
    print("\n=== BENCHMARK: SPY buy & hold, same window ===")
    print(f"  {'total_return_pct':>22}: {bm['total_return_pct']:,.3f}")
    print(f"  {'sharpe':>22}: {bm['sharpe']:,.3f}")
    print(f"  {'max_drawdown_pct':>22}: {bm['max_drawdown_pct']:,.3f}")
