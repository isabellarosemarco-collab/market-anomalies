"""
futures_overnight.py — harvesting the overnight premium through CME micro futures instead of ETFs.

THE CENTRAL ACCOUNTING POINT: FUTURES EXPOSURE IS NOT FREE.

A futures price embeds the cost of carry, F = S * exp((r - q) * tau). Holding a long future through
an interval dt therefore earns the index price move MINUS (r - q) * dt: you implicitly borrow at the
financing rate and implicitly receive the dividend yield. The $10,000 of collateral sitting in
T-bills is not incremental income — it is the other half of that same trade. Crediting the futures
book with a T-bill yield while ignoring the financing embedded in the basis would manufacture the
entire result out of double-counting.

So both structures are modelled on one symmetric convention:

  futures:  per night   L * (r_index_price - (r_f - q) * dt_night) + r_f * dt_night
            per day     r_f * dt_day                      (flat; collateral earns around the clock)

  cash ETF: per night   r_etf_total          (dividends included: the overnight holder is the
                                              holder of record on the ex-date, established in
                                              overnight_drift.py)
            per day     r_f * dt_day         (flat in T-bills, same as the futures book)

The earlier ETF tests did NOT credit the daytime T-bill yield, which understated them. Correcting
that here is what makes the comparison fair, and it removes most of the apparent futures advantage.

dt is CALENDAR time, not trading time. Financing accrues over weekends. The overnight leg spans
17.5h on a weekday and 65.5h over a weekend, so a 252-night year holds the position roughly 297
calendar days -- 81% of the year, not the ~50% a naive 17.5/24 calculation suggests.

WHY THE PER-CONTRACT FEE SCHEDULE FAVOURS THE WRONG CONTRACT

Futures fees are fixed per contract while ETF fees scale with notional, so the comparison turns
entirely on contract size:

    MES = $5 x SPX  ~ $34,000 notional  ->  $1.74 round trip = ~0.51bp
    M2K = $5 x RUT  ~ $12,000 notional  ->  $1.74 round trip = ~1.45bp

The overnight premium is concentrated in small caps (IWM/M2K), which is precisely the contract where
the fixed fee is most expensive in basis points. That tension is the result, not a side note.

LEVERAGE. "One contract on $10,000" is ~1.2x for M2K but ~3.4x for MES, so it is not the
cash-collateralised structure it appears to be. Sharpe is reported on excess-of-T-bill returns,
which is leverage-invariant, so the two contracts remain comparable; realised leverage and the
resulting drawdown are reported separately, where leverage actually shows up.

DATA. yfinance daily bars for MES=F/M2K=F are Globex sessions (18:00-17:00), so their open/close do
not correspond to 15:59 and 09:31 and cannot be used to reconstruct a cash-session overnight return.
The price path is therefore the cash index (^GSPC, ^RUT: price-only, exactly what a future tracks),
with the futures cost, carry and tax structure layered on top. `validate_basis` checks that
approximation against real M2K/MES intraday bars over the window where they are available, so the
error is measured rather than assumed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from overnight_drift import ANN, hac_t, max_dd

# ---------------------------------------------------------------------- contract specs

CONTRACTS = {
    "M2K": {"index": "^RUT", "etf": "IWM", "mult": 5.0, "tick_idx": 0.10,
            "launch": "2019-05-03", "label": "Micro Russell 2000"},
    "MES": {"index": "^GSPC", "etf": "SPY", "mult": 5.0, "tick_idx": 0.25,
            "launch": "2019-05-03", "label": "Micro S&P 500"},
}

# Per side, per contract. $0.62 IBKR unbundled + $0.25 exchange/NFA clearing = $0.87; $1.74 round trip.
COMM_PER_SIDE = 0.62 + 0.25
ROLLS_PER_YEAR = 4          # quarterly Mar/Jun/Sep/Dec

CAPITAL = 10_000.0
FLAT_RF = 0.045             # the spec's assumption, reported alongside realised rates

# Section 1256: 60% long-term / 40% short-term regardless of holding period.
TAX = {
    "spec (37% ord / 20% LTCG)": {"ord": 0.37, "ltcg": 0.20},
    "stated bracket (35% ord / 15% LTCG)": {"ord": 0.35, "ltcg": 0.15},
}


def blended_1256(ord_rate: float, ltcg_rate: float) -> float:
    return 0.60 * ltcg_rate + 0.40 * ord_rate


# ------------------------------------------------------------------------------ data

def load_index(sym: str, start: str = "2018-06-01") -> pd.DataFrame:
    """Price-only index OHLC. No dividend column needed: an index is already price-only, which is
    what the future tracks, so the ex-date attribution trap that plagued the ETF test cannot arise.
    """
    import yfinance as yf

    df = yf.Ticker(sym).history(start=start, interval="1d", auto_adjust=False)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
    df = df[(df > 0).all(axis=1)]
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def load_rf(start: str = "2018-06-01") -> pd.Series:
    """Realised 13-week T-bill yield (^IRX, quoted in percent) as a daily decimal rate.

    A flat 4.5% across 2019-2026 would be badly wrong in both directions: the policy rate was
    ~0.05% through 2021 and ~5.4% in 2023. Since the same rate drives BOTH the collateral yield and
    the financing embedded in the futures basis, getting it wrong distorts the levered case most.
    """
    import yfinance as yf

    s = yf.Ticker("^IRX").history(start=start, interval="1d")["Close"] / 100.0
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    return s.clip(lower=0.0).ffill()


def div_yield(etf: str, start: str = "2018-06-01") -> pd.Series:
    """Trailing 12-month dividend yield of the tracking ETF, as the index dividend yield q.

    Empirical rather than assumed, and it matters: q offsets the financing rate, so a 1.3% small-cap
    yield removes about a third of the carry drag at a 4.5% funding rate.
    """
    import yfinance as yf

    t = yf.Ticker(etf)
    px = t.history(start=start, interval="1d", auto_adjust=False)
    px = px.rename(columns=str.lower)
    if px.index.tz is not None:
        px.index = px.index.tz_localize(None)
    d = px.get("dividends", pd.Series(0.0, index=px.index)).fillna(0.0)
    return (d.rolling(252, min_periods=60).sum() / px["close"]).bfill().fillna(0.0)


# ------------------------------------------------------------------- session geometry

def sessions(idx_df: pd.DataFrame, px_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Overnight price move plus the calendar-time length of each leg.

    THE PRICE PATH COMES FROM THE ETF, NOT THE INDEX, AND THAT IS NOT A SHORTCUT.

    A cash index "open" is a composite of constituent opening prints. Small-cap constituents open in
    a staggered fashion, so ^RUT's reported open sits partway between the prior close and the true
    09:30 level. The effect is attenuation, not an exact repeat of the prior close, so it hides:
    measured against IWM over 2019-05..present, ^RUT's overnight move averages +3.24bp with sd 54.9bp
    against IWM's +6.42bp / 102.3bp, correlation only 0.837, with the missing variance reappearing in
    the intraday session (^RUT 139bp sd vs IWM 121bp). Roughly half the overnight move is simply not
    in the ^RUT open. ^GSPC is far better behaved (corr 0.979 with SPY) because large caps open
    promptly, but it is attenuated in the same direction.

    The ETF's opening print is a real auction fill at a transactable price, which is what the futures
    exit is actually benchmarked against, so it is the more accurate measurement of the same
    quantity. Dividends are stripped (r_night_px, not r_night) because a futures holder receives the
    dividend yield through the basis via -(r_f - q), not as a distribution -- crediting both would
    double-count it. The index is still used for contract notional, which is all it is needed for.
    """
    src = idx_df if px_df is None else px_df
    o, c = src["open"], src["close"]
    r = pd.DataFrame(index=src.index)
    r["r_night_px"] = o / c.shift(1) - 1.0
    r["r_day_px"] = c / o - 1.0
    r["idx_open"] = idx_df["open"].reindex(src.index).ffill()
    r["idx_prev_close"] = idx_df["close"].shift(1).reindex(src.index).ffill()

    # Calendar days between the prior close and this open, minus the 6.5h cash session.
    gap_days = pd.Series(src.index, index=src.index).diff().dt.days.astype(float)
    r["dt_night"] = (gap_days - 6.5 / 24.0) / 365.0
    r["dt_day"] = (6.5 / 24.0) / 365.0
    return r.dropna()


# ------------------------------------------------------------------------- the model

def futures_book(sess: pd.DataFrame, spec: dict, rf: pd.Series, q: pd.Series,
                 contracts: int = 1, capital: float = CAPITAL,
                 cross_prob: float = 0.5, seed: int = 11) -> pd.DataFrame:
    """Per-night P&L of holding `contracts` micro futures overnight, flat intraday.

    Spread convention: a passive limit inside the 1-tick spread fills at no cost; crossing pays the
    full tick. With 50/50 fill probability the expected cost is half a tick per side, one tick per
    round trip. Modelled as a realised draw, not the expectation, so the spread contributes variance
    as well as drag.
    """
    rng = np.random.default_rng(seed)
    w = sess.copy()
    rf = rf.reindex(w.index).ffill().fillna(0.0)
    q = q.reindex(w.index).ffill().fillna(0.0)

    notional = spec["mult"] * w["idx_prev_close"] * contracts
    tick_val = spec["mult"] * spec["tick_idx"]

    # Index P&L on the contract, in dollars.
    pnl_index = notional * w["r_night_px"]

    # Carry embedded in the basis: long futures pays r_f, receives q, over calendar time held.
    pnl_carry = -notional * (rf - q) * w["dt_night"]

    # Collateral earns around the clock (night leg here, day leg added below).
    pnl_collateral_night = capital * rf * w["dt_night"]

    n = len(w)
    crossings = rng.random((n, 2)) < cross_prob        # independent entry and exit draws
    pnl_spread = -tick_val * contracts * crossings.sum(axis=1)
    pnl_comm = -COMM_PER_SIDE * contracts * 2.0

    # Quarterly roll amortised over the nights in that quarter: both legs of a calendar spread.
    roll_per_night = -(COMM_PER_SIDE * 2 + tick_val) * contracts * ROLLS_PER_YEAR / ANN

    # Component attribution, each annualised as %/yr of capital, so the structure is legible.
    for nm, v in (("c_index", pnl_index), ("c_carry", pnl_carry),
                  ("c_collateral", pnl_collateral_night + capital * rf * w["dt_day"]),
                  ("c_spread", pd.Series(pnl_spread, index=w.index)),
                  ("c_comm", pd.Series(pnl_comm, index=w.index)),
                  ("c_roll", pd.Series(roll_per_night, index=w.index))):
        w[nm] = 100.0 * v / capital

    w["pnl_night"] = (pnl_index + pnl_carry + pnl_collateral_night
                      + pnl_spread + pnl_comm + roll_per_night)
    w["pnl_day"] = capital * rf * w["dt_day"]
    w["ret"] = (w["pnl_night"] + w["pnl_day"]) / capital
    w["rf_ret"] = rf * (w["dt_night"] + w["dt_day"])
    w["notional"] = notional
    w["cost_$"] = -(pnl_spread + pnl_comm + roll_per_night)
    w["cost_bp"] = 1e4 * w["cost_$"] / notional
    w["leverage"] = notional / capital
    w["carry_bp"] = 1e4 * pnl_carry / notional
    return w


def etf_book(etf_night_total: pd.Series, sess: pd.DataFrame, rf: pd.Series,
             slip_bp: float = 0.5, comm_per_share: float = 0.0035,
             price: pd.Series | None = None, capital: float = CAPITAL) -> pd.DataFrame:
    """The cash-ETF equivalent on the identical carry convention, including the daytime T-bill
    yield that the earlier ETF tests omitted."""
    from overnight_auction import round_trip_cost

    w = pd.DataFrame(index=sess.index)
    rf = rf.reindex(w.index).ffill().fillna(0.0)
    px = price.reindex(w.index).ffill()
    cost = round_trip_cost(px, slip_bp, comm_per_share)

    w["ret"] = (etf_night_total.reindex(w.index) - cost) + rf * sess["dt_day"]
    w["rf_ret"] = rf * (sess["dt_night"] + sess["dt_day"])
    w["cost_bp"] = 1e4 * cost
    w["leverage"] = 1.0
    return w.dropna()


# ------------------------------------------------------------------------------ stats

def perf(w: pd.DataFrame, label: str, tax_rate: float | None = None) -> dict:
    """Excess-of-T-bill Sharpe is the primary figure because it is leverage-invariant; total-return
    Sharpe is reported too since that is what the account statement shows.

    Post-tax scales gains by (1 - rate) annually. Section 1256 marks to market at year end, and the
    ETF book realises short-term gains continuously, so both are effectively annual -- but note this
    ignores loss-carryback asymmetries, which favour futures slightly further.
    """
    r = w["ret"].dropna()
    ex = (w["ret"] - w["rf_ret"]).dropna()
    yrs = len(r) / ANN
    eq = (1 + r).cumprod()
    cagr = eq.iloc[-1] ** (1 / yrs) - 1

    ann_gain = r.groupby(r.index.year).apply(lambda s: (1 + s).prod() - 1)
    if tax_rate is not None:
        taxed = ann_gain * (1 - tax_rate)
        post_cagr = float(np.prod(1 + taxed) ** (1 / len(taxed)) - 1)
        r_post = r * (1 - tax_rate)
        ex_post = r_post - w["rf_ret"].reindex(r_post.index)
    else:
        post_cagr, r_post, ex_post = cagr, r, ex

    dn = ex[ex < 0]
    return {
        "book": label, "nights": len(r),
        "cagr_%": 100 * cagr, "post_tax_cagr_%": 100 * post_cagr,
        "sharpe_excess": float(ex.mean() / ex.std(ddof=1) * np.sqrt(ANN)),
        "sharpe_total": float(r.mean() / r.std(ddof=1) * np.sqrt(ANN)),
        "sharpe_post_tax": float(ex_post.mean() / ex_post.std(ddof=1) * np.sqrt(ANN)),
        "sortino": float(ex.mean() / dn.std(ddof=1) * np.sqrt(ANN)) if len(dn) > 2 else np.nan,
        "maxdd_%": max_dd(r), "t_stat": hac_t(ex),
        "cost_bp_rt": float(w["cost_bp"].mean()),
        "leverage": float(w["leverage"].mean()),
    }


# -------------------------------------------------------------------- basis validation

def validate_basis(sym: str, index_sess: pd.DataFrame) -> dict:
    """Measure, don't assume: does the real future's 16:00 -> 09:30 move match the cash index's
    overnight move? Uses 1h bars (2y ceiling) with ~30min boundary error, and 15m bars (60d ceiling)
    with exact boundaries.
    """
    import yfinance as yf

    out = {"symbol": sym}
    for interval, period, close_h, open_h in (("1h", "730d", 15, 9), ("15m", "60d", 15, 9)):
        try:
            b = yf.Ticker(sym).history(period=period, interval=interval)
        except Exception as e:
            out[interval] = f"unavailable: {e}"
            continue
        if b is None or b.empty:
            out[interval] = "unavailable: empty"
            continue
        b = b.rename(columns=str.lower)
        b.index = b.index.tz_convert("America/New_York")
        if interval == "15m":
            last = b[(b.index.hour == close_h) & (b.index.minute == 45)]["close"]
            first = b[(b.index.hour == open_h) & (b.index.minute == 30)]["open"]
        else:
            last = b[b.index.hour == close_h]["close"]
            first = b[b.index.hour == open_h]["open"]
        last.index, first.index = last.index.normalize().tz_localize(None), \
            first.index.normalize().tz_localize(None)
        fut = (first / last.shift(1) - 1.0).dropna()
        cash = index_sess["r_night_px"].reindex(fut.index).dropna()
        j = pd.concat([fut.rename("fut"), cash.rename("cash")], axis=1).dropna()
        if len(j) < 30:
            out[interval] = f"only {len(j)} overlapping nights"
            continue
        d = j["fut"] - j["cash"]
        out[interval] = {"nights": len(j), "corr": float(j.corr().iloc[0, 1]),
                         "mean_diff_bp": 1e4 * float(d.mean()),
                         "sd_diff_bp": 1e4 * float(d.std(ddof=1))}
    return out


# ------------------------------------------------------------------------------- main

def fmt(df: pd.DataFrame) -> str:
    return df.to_string(float_format=lambda x: f"{x:,.3f}")


def prepare(spec: dict, start: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Sessions on the ETF price path, index levels for notional, ETF decomposition for the
    cash-ETF counterpart. Cached per contract so the repeated subsample runs do not refetch."""
    from overnight_drift import decompose, load_ohlc_div

    key = (spec["index"], spec["etf"])
    if key not in prepare._cache:
        idx = load_index(spec["index"])
        etf_raw = load_ohlc_div([spec["etf"]])[spec["etf"]]
        prepare._cache[key] = (idx, sessions(idx, etf_raw), decompose(etf_raw),
                               div_yield(spec["etf"]))
    idx, sess, etf_r, q = prepare._cache[key]
    lo = start or spec["launch"]
    return sess[sess.index >= lo], etf_r, q


prepare._cache = {}


def main() -> int:
    rf_real = load_rf()
    results, cost_rows = [], []

    for name, spec in CONTRACTS.items():
        sess, etf_r, q = prepare(spec)

        tick_val = spec["mult"] * spec["tick_idx"]
        notional_now = spec["mult"] * float(sess["idx_open"].iloc[-1])
        notional_avg = spec["mult"] * float(sess["idx_prev_close"].mean())
        rt = 2 * COMM_PER_SIDE + tick_val
        # Reported at BOTH index levels on purpose. A fixed per-contract fee is a shrinking
        # percentage cost as the index rises, so quoting only today's level backdates an advantage
        # the strategy did not have: the Russell traded near 1,500 in 2019 and the performance
        # figures below are driven by that lower notional, not by today's.
        cost_rows.append({
            "contract": f"{name} ({spec['label']})",
            "notional_now_$": notional_now,
            "notional_avg_$": notional_avg,
            "tick_$": tick_val,
            "comm_rt_$": 2 * COMM_PER_SIDE,
            "spread_rt_$ (exp.)": tick_val,
            "total_rt_$": rt,
            "rt_bp_at_now": 1e4 * rt / notional_now,
            "rt_bp_at_avg": 1e4 * rt / notional_avg,
            # Notional at which the fixed fee equals the ETF's proportional ~1.65bp round trip.
            "breakeven_notional_$": rt / 1.649e-4,
            "leverage_on_$10k": notional_now / CAPITAL,
        })

        for rf_label, rf in (("realised ^IRX", rf_real),
                             ("flat 4.5%", pd.Series(FLAT_RF, index=rf_real.index))):
            w = futures_book(sess, spec, rf, q)
            for tax_label, t in TAX.items():
                rate = blended_1256(t["ord"], t["ltcg"])
                p = perf(w, f"{name} futures", rate)
                p.update({"rates": rf_label, "tax": f"1256 @ {100*rate:.1f}%",
                          "tax_scenario": tax_label})
                results.append(p)

        # ETF counterpart on the same convention, same dates.
        for rf_label, rf in (("realised ^IRX", rf_real),
                             ("flat 4.5%", pd.Series(FLAT_RF, index=rf_real.index))):
            we = etf_book(etf_r["r_night"], sess, rf, price=etf_r["close"])
            for tax_label, t in TAX.items():
                p = perf(we, f"{spec['etf']} cash ETF", t["ord"])
                p.update({"rates": rf_label, "tax": f"ordinary @ {100*t['ord']:.0f}%",
                          "tax_scenario": tax_label})
                results.append(p)

    print("=" * 118)
    print("ROUND-TRIP COST STRUCTURE (fixed per-contract fees vs contract notional)")
    print("=" * 118)
    print(fmt(pd.DataFrame(cost_rows).set_index("contract")))
    print("\n  Cash ETF reference at 0.5bp/side + $0.0035/sh + SEC/FINRA: ~1.6bp round trip.")

    print("\n" + "=" * 118)
    print("P&L ATTRIBUTION, %/yr of $10,000 capital (realised rates) — where the structure wins "
          "and loses")
    print("=" * 118)
    attr = []
    for name, spec in CONTRACTS.items():
        sess, _, q = prepare(spec)
        w = futures_book(sess, spec, rf_real, q)
        yrs = len(w) / ANN
        row = {"contract": name, "leverage": float(w["leverage"].mean())}
        row.update({k.replace("c_", ""): float(w[k].sum()) / yrs
                    for k in ("c_index", "c_carry", "c_collateral", "c_spread", "c_comm", "c_roll")})
        row["sum"] = sum(v for k, v in row.items() if k not in ("contract", "leverage"))
        attr.append(row)
    print(fmt(pd.DataFrame(attr).set_index("contract")))
    print("  index    = leveraged overnight price move on the contract")
    print("  carry    = -(r_f - q) * dt on full notional: the financing embedded in the basis")
    print("  collateral = T-bill on $10,000, around the clock")
    print("  Note carry scales with NOTIONAL while collateral scales with CAPITAL, so leverage")
    print("  turns the futures carry structure from a small credit into a net cost.")

    res = pd.DataFrame(results)
    cols = ["book", "rates", "tax", "nights", "leverage", "cost_bp_rt", "cagr_%",
            "post_tax_cagr_%", "sharpe_total", "sharpe_excess", "sharpe_post_tax",
            "sortino", "maxdd_%", "t_stat"]

    for scen in TAX:
        print("\n" + "=" * 118)
        print(f"PERFORMANCE — tax scenario: {scen}")
        print("=" * 118)
        sub = res[res["tax_scenario"] == scen][cols]
        print(fmt(sub.set_index("book")))

    print("\n" + "=" * 118)
    print("POST-2020 SUBSAMPLE (realised rates, spec tax scenario) — does anything clear Sharpe 1.0?")
    print("=" * 118)
    rows = []
    for name, spec in CONTRACTS.items():
        sess, etf_r, q = prepare(spec, start="2020-01-01")
        w = futures_book(sess, spec, rf_real, q)
        rate = blended_1256(TAX["spec (37% ord / 20% LTCG)"]["ord"],
                            TAX["spec (37% ord / 20% LTCG)"]["ltcg"])
        rows.append(perf(w, f"{name} futures"))
        rows.append(perf(w, f"{name} futures (post-tax)", rate))

        we = etf_book(etf_r["r_night"], sess, rf_real, price=etf_r["close"])
        rows.append(perf(we, f"{spec['etf']} ETF"))
        rows.append(perf(we, f"{spec['etf']} ETF (post-tax)", TAX["spec (37% ord / 20% LTCG)"]["ord"]))
    print(fmt(pd.DataFrame(rows).set_index("book")[
        ["nights", "leverage", "cost_bp_rt", "cagr_%", "post_tax_cagr_%", "sharpe_total",
         "sharpe_excess", "sharpe_post_tax", "sortino", "maxdd_%", "t_stat"]]))
    print("\n  sharpe_total is comparable to the earlier ETF tests (Sharpe 0.794 post-2020);")
    print("  sharpe_excess nets out the T-bill and is the leverage-invariant figure.")

    print("\n" + "=" * 118)
    print("BASIS VALIDATION — real futures 16:00->09:30 move vs cash index overnight move")
    print("=" * 118)
    print("  Reference path = ETF opening auction print (price-only), per sessions() docstring.")
    for name, spec in CONTRACTS.items():
        sess, _, _ = prepare(spec)
        v = validate_basis(f"{name}=F", sess)
        print(f"  {name}=F")
        for k in ("1h", "15m"):
            print(f"    {k:>4}: {v.get(k)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
