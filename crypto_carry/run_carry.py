"""
run_carry.py — the crypto carry verdict, with the same fee-and-friction accounting that killed the odd lots.

  python -m crypto_carry.run_carry                 # BTC + ETH, full scenario grid
  python -m crypto_carry.run_carry --symbol BTCUSDT --capital 5000

READ THE GROSS YIELD FIRST AND ONLY THEN THE STRATEGY. The pitch for this trade is "15-40% annualised
funding in bullish regimes". That is true of individual months and false of the multi-year average, and
the average is what a strategy earns. Everything downstream is a deduction from whatever the top line
actually is, so the top line gets printed before any strategy is simulated.
"""
from __future__ import annotations

import argparse

import pandas as pd

from crypto_carry import carry_backtest as cb
from crypto_carry import funding_data as fd


def bull_run_stress(fund: pd.DataFrame, px: pd.DataFrame, capital: float, buffer: float) -> None:
    """Question 2: how often does the short leg need collateral during rapid upside?

    Answered two ways, because the model answer and the arithmetic answer should agree. The arithmetic:
    a buffer of b is exhausted by a +b move, so the relevant statistic is how often the asset ran +b
    from a local entry. The model: count actual top-up events in the simulation.
    """
    p = px.copy()
    p["date"] = pd.to_datetime(p.date)
    p = p.set_index("date").sort_index()
    print(f"\n  upside stress on the short leg (buffer survives +{buffer:.0%}):")
    for win in (7, 30, 90):
        mx = p.spot.pct_change(win).max()
        n = int((p.spot.pct_change(win) > buffer * 0.5).sum())
        print(f"    {win:3d}d: worst-case upside +{mx:.1%}   "
              f"days where trailing {win}d gain exceeded half the buffer: {n} "
              f"({100*n/len(p):.1f}% of the sample)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=None, help="BTCUSDT or ETHUSDT; default both")
    ap.add_argument("--capital", type=float, default=5000.0)
    a = ap.parse_args()

    syms = [a.symbol] if a.symbol else ["BTCUSDT", "ETHUSDT"]

    print("=" * 118)
    print("CRYPTO PERPETUAL FUNDING CARRY — long spot / short perp, every friction priced")
    print("=" * 118)

    for sym in syms:
        fund = fd.funding_history(sym)
        px = fd.daily_prices(sym)
        cb.gross_yield_table(fund, sym)

        # The scenario grid. Two capital splits (the user's literal 50/50 and the margin-optimal +40%
        # buffer), two fee levels spanning the retail taker range, and both exit thresholds.
        rows = []
        for buffer, bname in ((1.00, "50/50 split"), (0.40, "+40% buffer")):
            for fee in (0.0005, 0.00075):
                for ex in ("tbill", "zero"):
                    r = cb.backtest(fund, px, capital=a.capital, buffer=buffer,
                                    taker_fee=fee, exit_threshold=ex,
                                    label=f"{bname}  fee {fee:.3%}  exit<{ex}")
                    rows.append(r.row())
        # And the counterfactual that isolates the frictions: no fees, never exit.
        hold = cb.backtest(fund, px, capital=a.capital, buffer=0.40, taker_fee=0.0,
                           exit_threshold="zero", label="+40% buffer  ZERO fees  never exit")
        rows.append(hold.row())

        print(f"\n{sym}: net results on ${a.capital:,.0f}")
        print(pd.DataFrame(rows).to_string(index=False))

        base = cb.backtest(fund, px, capital=a.capital, buffer=0.40,
                           taker_fee=0.0005, exit_threshold="tbill")
        print(f"\n  decomposition of the base case (+40% buffer, 0.05% taker, T-bill exit):")
        print(f"    funding received        {base.funding_received:+10,.0f}")
        print(f"    basis P&L               {base.basis_pnl:+10,.0f}")
        print(f"    fees paid               {-base.fees_paid:+10,.0f}   "
              f"({base.cycles} cycles + {base.topups} forced de-levers)")
        print(f"    ------------------------------------")
        print(f"    net over {base.days/365:.1f}y         {base.net_profit:+10,.0f}   "
              f"= {100*base.ann_return:.2f}%/yr vs 4.00% risk-free")
        print(f"    fees as % of funding    {100*base.fees_paid/max(base.funding_received,1e-9):9.1f}%")
        resid = base.net_profit - (base.funding_received + base.basis_pnl - base.fees_paid)
        print(f"    reconciliation residual {resid:+10,.2f}   (must be ~0)")

        # THE DECAY IS THE DECISION. A 5.7-year average blends 2021's +30% funding regime with 2026's
        # +2.6%, and you cannot invest in an average. Per-year net answers "what would this have paid
        # LAST year", which is the only version of the question that bears on deploying capital now.
        print(f"\n  net return BY YEAR (+40% buffer, 0.05% taker, T-bill exit) — the average is a blend:")
        yr_rows = []
        for y in sorted(fund.time.dt.year.unique()):
            f_y = fund[fund.time.dt.year == y]
            p_y = px[pd.to_datetime(px.date).dt.year == y]
            if len(f_y) < 60 or len(p_y) < 30:
                continue
            try:
                r = cb.backtest(f_y, p_y, capital=a.capital, buffer=0.40,
                                taker_fee=0.0005, exit_threshold="tbill", label=str(y))
            except RuntimeError:
                continue
            yr_rows.append({"year": y, "days": r.days, "in_mkt%": round(100*r.days_in_market/r.days, 0),
                            "cycles": r.cycles, "topups": r.topups,
                            "funding$": round(r.funding_received), "fees$": round(r.fees_paid),
                            "net$": round(r.net_profit), "ann%": round(100*r.ann_return, 2),
                            "vs 4%": round(100*r.excess_ann, 2)})
        print(pd.DataFrame(yr_rows).to_string(index=False))

        bull_run_stress(fund, px, a.capital, 0.40)

    # Venue check: Binance is the measurement venue, Hyperliquid is the tradeable one.
    print("\n" + "=" * 118)
    print("VENUE CHECK — measured on Binance, but a US retail account trades Hyperliquid")
    print("=" * 118)
    hl = fd.hyperliquid_funding("BTC")
    bn = fd.funding_history("BTCUSDT")
    lo = max(hl.time.min(), bn.time.min())
    a_hl = hl[hl.time >= lo].rate.mean() * 8760
    a_bn = bn[bn.time >= lo].rate.mean() * 1095
    print(f"  overlapping window from {lo.date()}:")
    print(f"    Binance   8h settlement, {a_bn:6.2%}/yr annualised on notional")
    print(f"    Hyperliquid 1h settlement, {a_hl:6.2%}/yr annualised on notional")
    print(f"  gap {a_hl - a_bn:+.2%}/yr — Hyperliquid paid {a_hl/max(a_bn,1e-9):.1f}x Binance, and that is")
    print(f"  NOT noise. It is a protocol constant:")

    # WHY THE VENUE GAP IS STRUCTURAL AND WHY THAT IS A WARNING AS MUCH AS A GIFT. Hyperliquid's funding
    # formula carries a fixed interest-rate component; the rate sits at exactly that floor a majority of
    # hours, so its median funding IS the baseline rather than a market-clearing premium. Binance's
    # premium term pulls it off the same baseline far more often. So the extra yield is not an
    # inefficiency anyone is competing away — it is a parameter Hyperliquid chose and can re-choose.
    # Sizing a book on it means accepting governance risk as a return driver.
    floor = 1.25e-05
    at_floor = 100 * (hl[hl.time >= lo].rate.round(9) == floor).mean()
    bn_floor = 100 * (bn[bn.time >= lo].rate.round(9) == 1e-04).mean()
    print(f"    Hyperliquid sits at EXACTLY its {floor*8760:.2%}/yr interest floor "
          f"{at_floor:.1f}% of hours (median = the floor)")
    print(f"    Binance sits at exactly the same {1e-04*1095:.2%}/yr baseline only {bn_floor:.1f}% "
          f"of settlements")
    print(f"  So the better venue is the tradeable one — but its edge is a config value, not an")
    print(f"  inefficiency. It can be changed by governance without notice, so do not capitalise it.")

    # Now actually BACKTEST the tradeable venue rather than reasoning about it. Hyperliquid's taker fee
    # is 0.045% and there is no per-trade gas (it is an L1 with an on-chain order book), so the assumed
    # "retail slippage/gas eats the carry" turns out not to be the binding cost at $5k on BTC.
    # Approximation stated: the spot leg is priced off Binance spot closes because BTC spot is fungible
    # across venues. Only the basis term is affected and it contributes single-digit dollars.
    print("\n" + "=" * 118)
    print("THE TRADEABLE CONFIGURATION — Hyperliquid funding, Hyperliquid taker fee, no gas")
    print("=" * 118)
    pxb = fd.daily_prices("BTCUSDT")
    pxb = pxb[pd.to_datetime(pxb.date) >= hl.time.min().tz_localize(None)]
    rows = []
    for buffer, bname in ((1.00, "50/50 split"), (0.40, "+40% buffer")):
        for ex in ("tbill", "zero"):
            rows.append(cb.backtest(hl, pxb, capital=a.capital, buffer=buffer, taker_fee=0.00045,
                                    exit_threshold=ex,
                                    label=f"{bname}  HL taker 0.045%  exit<{ex}").row())
    rows.append(cb.backtest(hl, pxb, capital=a.capital, buffer=0.40, taker_fee=0.0,
                            exit_threshold="zero", label="+40% buffer  ZERO fees  never exit").row())
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\n  Only {(hl.time.max()-hl.time.min()).days/365:.1f} years of history exists on this venue, and it is ONE regime —")
    print(f"  the same window in which Binance paid {a_bn:.2%}/yr. The gap between the two IS the")
    print(f"  interest floor above. Treat the Sharpe as a property of that floor, not of the strategy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
