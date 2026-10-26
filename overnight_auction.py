"""
overnight_auction.py — can the overnight premium be harvested at auction-quality execution?

TEST 1: IWM overnight, 2018+ and 2020+, priced at realistic auction cost.

LOC/LOO orders clear at the single auction price, so no bid-ask spread is crossed. But "zero
spread" is not zero cost, and three components remain regardless of broker:

  * SEC Section 31 fee, sells only, $27.80 per $1M notional as of FY2025 = 0.278bp per round trip.
    Unavoidable. It is larger than the modelled slippage and no commission structure removes it.
  * FINRA TAF, sells only, $0.000166/share.
  * Residual slippage. Auctions have no spread but do have price impact and, more importantly,
    fill uncertainty; 0.5bp/side is the assumption under test, not a measurement.

ADVERSE SELECTION ON LIMIT ORDERS (not in the original spec, but decisive)

A Limit-On-Close order pegged to the cross only fills if the cross prints at or better than the
limit. That conditions your fills on price: you reliably get filled on nights the market is drifting
toward you, and miss fills precisely when it gaps away. A backtest that assumes 100% fills therefore
books the best nights for free. The `drop_best_pct` sensitivity removes the top k% of overnight
returns to bound this. It is a bound, not a model — but a strategy whose edge vanishes when the best
5% of nights are removed is not robust to real order handling.

TEST 2: execution overlay for a passive holder.

Separate question, no daily round trips: a 60/40 IWM/cash investor rebalancing monthly routes all
BUYS to the closing auction and all SELLS to the opening auction, versus executing both at an
arbitrary intraday time. Because IWM's intraday session has negative drift and its overnight session
has positive drift, buying as late as possible and selling as early as possible harvests part of the
same structural tilt at zero incremental transaction cost — you were going to trade anyway.

"Random intraday time" is modelled as open * (1 + u * r_day), u ~ U(0,1): a linear interpolation
along the day's realized move. This is a proxy, not a path simulation — it has the right endpoints
and the right mean, but understates intraday variance. Reported with Monte Carlo dispersion so the
timing luck is visible rather than hidden in a point estimate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from overnight_drift import ANN, decompose, hac_t, load_ohlc_div, max_dd, signals, stats

# Regulatory costs, both sell-side only.
SEC_FEE_FRAC = 27.80 / 1e6          # 0.278bp of sell notional (FY2025 rate)
FINRA_TAF_PER_SHARE = 0.000166

BROKERS = {
    "IBKR fixed $0.0035/sh": 0.0035,
    "zero-commission retail": 0.0,
}


def round_trip_cost(price: pd.Series, slip_bp: float, comm_per_share: float) -> pd.Series:
    """Total round-trip cost as a fraction of notional: buy at the close, sell at the open."""
    slip = 2.0 * (slip_bp / 1e4)
    comm = 2.0 * comm_per_share / price
    reg = SEC_FEE_FRAC + FINRA_TAF_PER_SHARE / price      # sells only
    return slip + comm + reg


# ---------------------------------------------------------------------------- test 1

def auction_table(r: pd.DataFrame, start: str, slip_bps=(0.0, 0.25, 0.5, 1.0)) -> pd.DataFrame:
    w = r[r.index >= start]
    on = pd.Series(1.0, index=w.index)
    rows = [{"config": "GROSS (no costs)", **_pick(stats(w["r_night"], "g"))}]
    for bname, comm in BROKERS.items():
        for bp in slip_bps:
            net = w["r_night"] - round_trip_cost(w["close"], bp, comm)
            rows.append({"config": f"{bname} @ {bp}bp/side", **_pick(stats(net, "n"))})
    return pd.DataFrame(rows).set_index("config")


def _pick(d: dict) -> dict:
    return {k: d[k] for k in ("days", "ann_ret_%", "ann_vol_%", "sharpe", "sortino",
                              "maxdd_%", "t_stat")}


def adverse_selection(r: pd.DataFrame, start: str, slip_bp: float, comm: float,
                      drops=(0.0, 1.0, 2.5, 5.0, 10.0)) -> pd.DataFrame:
    """Remove the best k% of overnight returns to bound limit-order non-fill selection bias."""
    w = r[r.index >= start]
    net = w["r_night"] - round_trip_cost(w["close"], slip_bp, comm)
    rows = []
    for k in drops:
        if k == 0:
            kept = net
        else:
            thr = net.quantile(1 - k / 100)
            kept = net[net < thr]          # missed fills on the most favourable nights
        st = stats(kept, "x")
        rows.append({"best_nights_missed_%": k, "nights": st["days"],
                     "ann_ret_%": st["ann_ret_%"], "sharpe": st["sharpe"],
                     "t_stat": st["t_stat"], "maxdd_%": st["maxdd_%"]})
    return pd.DataFrame(rows).set_index("best_nights_missed_%")


def conditional_auction(r: pd.DataFrame, start: str, slip_bp: float, comm: float) -> pd.DataFrame:
    """Does conditioning still help once costs are auction-realistic and only paid when trading?"""
    w = r[r.index >= start]
    rows = []
    for q in (0.0, 0.2, 0.3):
        for name in ("cond_t1", "cond_t2"):
            act = signals(r, 20, q)[name].astype(float).reindex(w.index).fillna(0.0)
            gross = w["r_night"] * act
            net = gross - act * round_trip_cost(w["close"], slip_bp, comm)
            st = stats(net, "x", act)
            g = stats(gross, "g", act)
            per_night = (1e4 * gross.mean() / act.mean()) if act.mean() > 0 else np.nan
            rows.append({"config": f"{name} q={q}", "exposed_%": st["exposed_%"],
                         "gross_bp/night": per_night, "net_ann_%": st["ann_ret_%"],
                         "net_sharpe": st["sharpe"], "net_t": st["t_stat"],
                         "maxdd_%": st["maxdd_%"], "gross_sharpe": g["sharpe"]})
    rows.append({"config": "always (uncond)", "exposed_%": 100.0,
                 "gross_bp/night": 1e4 * w["r_night"].mean(),
                 **{k: v for k, v in zip(
                     ("net_ann_%", "net_sharpe", "net_t", "maxdd_%"),
                     (lambda s: (s["ann_ret_%"], s["sharpe"], s["t_stat"], s["maxdd_%"]))(
                         stats(w["r_night"] - round_trip_cost(w["close"], slip_bp, comm), "x")))},
                 "gross_sharpe": stats(w["r_night"], "g")["sharpe"]})
    return pd.DataFrame(rows).set_index("config")


# ---------------------------------------------------------------------------- test 2

def rebalance_sim(px: pd.DataFrame, cash_ret: pd.Series, mode: str, rng: np.random.Generator,
                  equity_target: float = 0.60, years: int = 5,
                  diag: dict | None = None, end: pd.Timestamp | None = None) -> float:
    """Monthly-rebalanced 60/40 IWM/cash portfolio. Returns terminal equity.

    mode='overlay'  -> buys fill at the closing auction, sells at the opening auction
    mode='reverse'  -> the exact opposite routing: buys at the open, sells at the close
    mode='midpoint' -> both fill at the open/close midpoint (neutral reference)
    mode='intraday' -> both fill at a uniformly random time, open*(1 + u*r_day)

    A NOTE ON WHY 'intraday' IS NOT THE PRIMARY BENCHMARK. Randomising the fill price injects
    variance that the deterministic modes do not have, and buys convert price into shares via
    delta/price. Because E[1/P] > 1/E[P] (Jensen), the randomised baseline receives a small free
    share bonus that has nothing to do with execution timing — it is an artifact of the model. So
    'midpoint' is the neutral reference and 'reverse' is the two-sided bound; 'intraday' is reported
    only to show the dispersion from timing luck.

    Rebalance need is computed at the close of t-1 so the execution price is never used to decide
    the trade. Both modes pay identical transaction costs; the only difference is WHEN in the day
    the same trade prints, which isolates the timing tilt from any cost effect.
    """
    idx = px.index
    end = idx[-1] if end is None else end
    start = end - pd.DateOffset(years=years)
    idx = idx[(idx >= start) & (idx <= end)]
    o, c = px["open"], px["close"]

    # Last trading day of each month = decision dates; execution is the following session.
    s = pd.Series(idx, index=idx)
    dec_dates = set(pd.DatetimeIndex(s.groupby([idx.year, idx.month]).last().values))

    # The initial entry is NOT a rebalance, and it must be established at an identical price in
    # every mode. Routing it through the overlay logic makes it by far the largest trade in the run
    # (~60% of equity vs ~1-2% for a monthly rebalance), so a single day's open-to-close move on
    # that one fill swamps the sixty trades the test is actually about.
    cash = 10_000.0
    shares = (equity_target * cash) / float(c[idx[0]])
    cash -= shares * float(c[idx[0]])
    pending = None

    for i, t in enumerate(idx):
        if i > 0:
            cash *= 1.0 + float(cash_ret.get(t, 0.0))

        if pending is not None:
            delta_val, kind = pending
            op, cl = float(o[t]), float(c[t])
            if mode == "overlay":
                price = op if kind == "sell" else cl
            elif mode == "reverse":
                price = cl if kind == "sell" else op
            elif mode == "midpoint":
                price = 0.5 * (op + cl)
            else:
                price = op * (1.0 + rng.random() * (cl / op - 1.0))
            sh = delta_val / price
            if diag is not None:
                diag.setdefault("trades", []).append(
                    {"date": t, "kind": kind, "r_day": cl / op - 1.0,
                     "notional": abs(delta_val)})
            shares += sh
            cash -= sh * price
            # Identical costs in both modes: commission both sides, reg fees on sells only.
            notional = abs(sh) * price
            cost = notional * (0.5 / 1e4) + abs(sh) * 0.0035
            if sh < 0:
                cost += notional * SEC_FEE_FRAC + abs(sh) * FINRA_TAF_PER_SHARE
            cash -= cost
            pending = None

        if t in dec_dates:
            total = shares * float(c[t]) + cash
            target = equity_target * total
            delta = target - shares * float(c[t])
            if abs(delta) > 1.0:
                pending = (delta, "buy" if delta > 0 else "sell")

    return shares * float(c[idx[-1]]) + cash


def overlay_test(px: pd.DataFrame, cash_ret: pd.Series, years: int = 5,
                 n_paths: int = 400, seed: int = 7) -> tuple[pd.DataFrame, dict]:
    """Deterministic comparisons first, randomised path dispersion second.

    'midpoint' is the honest benchmark for "an arbitrary time of day": same trades, same costs,
    same dates, no injected price variance. 'reverse' is the worst-case routing and brackets the
    total size of the timing effect (overlay - reverse = twice the one-sided tilt).
    """
    ann = lambda v: (v / 10_000.0) ** (1 / years) - 1
    rows, diags = [], []

    # Several non-overlapping-ish 5y windows rather than one. Sixty rebalances is a thin sample and
    # a single window's answer is dominated by whatever the intraday moves happened to be on those
    # sixty specific dates.
    ends = [px.index[-1] - pd.DateOffset(years=k) for k in range(0, 4)]
    for wend in ends:
        rng = np.random.default_rng(seed)
        diag: dict = {}
        det = {m: rebalance_sim(px, cash_ret, m, rng, years=years, end=wend,
                                diag=diag if m == "overlay" else None)
               for m in ("overlay", "midpoint", "reverse")}
        mc = np.array([rebalance_sim(px, cash_ret, "intraday", rng, years=years, end=wend)
                       for _ in range(n_paths)])
        a_ov = ann(det["overlay"])
        tr = pd.DataFrame(diag.get("trades", []))
        rows.append({
            "window_end": wend.date(),
            "overlay_ann_%": 100 * a_ov,
            "vs_midpoint_bp/yr": 1e4 * (a_ov - ann(det["midpoint"])),
            "vs_reverse_bp/yr": 1e4 * (a_ov - ann(det["reverse"])),
            "vs_random_bp/yr": 1e4 * float((a_ov - ann(mc)).mean()),
            "pct_random_beaten": float(100 * (a_ov > ann(mc)).mean()),
        })
        diags.append(tr.assign(window_end=wend.date()))

    tbl = pd.DataFrame(rows).set_index("window_end")
    tr = pd.concat(diags, ignore_index=True)

    # The ceiling on this idea. The overlay only touches the traded delta, so its benefit is
    # bounded by turnover: gain/yr <= (rebalance notional / equity per year) * (intraday move
    # captured). A 60/40 book drifts little in a month, so annual turnover is a couple of percent —
    # which caps the whole effect at a fraction of a basis point no matter how large the drift is.
    turn = float(tr.groupby("window_end")["notional"].sum().mean()) / 10_000.0 / years
    w = px[px.index >= px.index[-1] - pd.DateOffset(years=years)]
    half_drift = 0.5 * abs(float((w["close"] / w["open"] - 1.0).mean()))
    bound = {
        "rebalances_per_window": float(tr.groupby("window_end").size().mean()),
        "pct_sells": float(100 * (tr["kind"] == "sell").mean()),
        "mean_trade_notional_$": float(tr["notional"].mean()),
        "annual_turnover_%_of_equity": 100 * turn,
        "mean_r_day_on_exec_days_bp": 1e4 * float(tr["r_day"].mean()),
        "half_intraday_drift_bp/trade": 1e4 * half_drift,
        "theoretical_max_gain_bp/yr": 1e4 * turn * half_drift,
    }
    return tbl, bound


def concentration_control(r: pd.DataFrame, start: str) -> pd.DataFrame:
    """Is the overnight book unusually dependent on a few nights, or is that just equity fat tails?

    Dropping the best k% of observations craters ANY long equity series, so the adverse-selection
    table above is only damning relative to a control. Buy-and-hold is subjected to the identical
    truncation on the identical dates-count basis.
    """
    w = r[r.index >= start]
    rows = []
    for k in (0.0, 1.0, 2.5, 5.0):
        out = {"best_dropped_%": k}
        for lab, s in (("overnight", w["r_night"]), ("intraday", w["r_day"]),
                       ("buy&hold", w["r_bh"])):
            kept = s if k == 0 else s[s < s.quantile(1 - k / 100)]
            st = stats(kept, "x")
            out[f"{lab}_ann_%"] = st["ann_ret_%"]
            out[f"{lab}_sharpe"] = st["sharpe"]
        rows.append(out)
    return pd.DataFrame(rows).set_index("best_dropped_%")


# -------------------------------------------------------------------------------- main

def fmt(df: pd.DataFrame) -> str:
    return df.to_string(float_format=lambda x: f"{x:,.3f}")


def main() -> int:
    data = load_ohlc_div(["IWM", "SPY"])
    dec = {s: decompose(d) for s, d in data.items()}
    r = dec["IWM"]

    for start, label in (("2018-01-01", "2018-present"), ("2020-01-01", "2020-present")):
        print("=" * 104)
        print(f"TEST 1 — IWM OVERNIGHT AT AUCTION EXECUTION, {label}")
        print("=" * 104)
        print(fmt(auction_table(r, start)))
        print(f"\n  Unavoidable regulatory floor: SEC Sec.31 {1e4*SEC_FEE_FRAC:.3f}bp/round trip "
              f"(sells) + FINRA TAF — present in every row above, including zero-commission.")

        print(f"\n  --- conditional variants, IBKR $0.0035 @ 0.5bp/side, {label} ---")
        print(fmt(conditional_auction(r, start, 0.5, 0.0035)))

        print(f"\n  --- adverse selection: LOC/LOO non-fills, $0.0035 @ 0.5bp/side, {label} ---")
        print(fmt(adverse_selection(r, start, 0.5, 0.0035)))

        print(f"\n  --- control: same truncation applied to every session type, {label} ---")
        print(fmt(concentration_control(r, start)))
        print()

    print("=" * 104)
    print("TEST 2 — EXECUTION OVERLAY FOR A PASSIVE 60/40 IWM HOLDER (5-year hold)")
    print("=" * 104)
    bil = load_ohlc_div(["BIL"])["BIL"]
    cash_ret = (bil["close"] + bil["dividends"].fillna(0.0)) / bil["close"].shift(1) - 1.0
    px = data["IWM"]
    tbl, res = overlay_test(px, cash_ret.fillna(0.0), years=5)
    print("  Rolling 5-year windows. Initial entry priced identically in all modes; only the")
    print("  ~60 monthly rebalance fills are routed differently.\n")
    print(fmt(tbl))
    print("\n  WHY THE EFFECT IS BOUNDED (averaged across windows)")
    for k, v in res.items():
        print(f"  {k:>32}: {v:,.4f}" if isinstance(v, float) else f"  {k:>32}: {v}")

    print("\n  Same trades, same costs, same dates — only the time of day differs.")
    print("  Intraday drift on IWM over this window:")
    w = r[r.index >= px.index[-1] - pd.DateOffset(years=5)]
    print(f"    intraday  {1e4*w['r_day'].mean():+.2f}bp/day  ({100*((1+w['r_day']).prod()**(252/len(w))-1):+.2f}%/yr)")
    print(f"    overnight {1e4*w['r_night'].mean():+.2f}bp/day  ({100*((1+w['r_night']).prod()**(252/len(w))-1):+.2f}%/yr)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
