"""
carry_backtest.py — delta-neutral cash-and-carry (long spot / short perp) with every friction priced.

THE STRUCTURAL HAIRCUT THAT THE HEADLINE YIELD HIDES, AND IT IS NOT THE FEES. Quoted funding yields are
returns on NOTIONAL. This trade earns them on a notional strictly smaller than your capital, because the
short leg needs its own margin and that margin earns no funding.

    delta-neutral  ->  spot notional S long, perp notional S short
    survive a +b adverse move without liquidation  ->  margin M >= b * S
    capital deployed  C = S + M = S(1 + b)   ->   S = C / (1 + b)

    RETURN ON CAPITAL  =  funding_rate  x  1/(1 + b)

So the capital-efficiency factor is 1/(1+b), and it is brutal:

    b = 40%  (survives +40%)   ->  0.714 x the headline yield
    b = 100% (the literal 50/50 split, survives +100%)  ->  0.500 x

A "20% annualised funding rate" is 10% on capital at a 50/50 split before a single fee is paid. This is
the same shape as the odd-lot fee drag: a fixed structural cost measured against a capped deployable
base. Both books die in the denominator.

THE SECOND DRAG IS THE ONE NOBODY MODELS: BULL RUNS FORCE YOU TO SHRINK. When price rises, the short leg
bleeds margin. To avoid liquidation you must move cash in, and the only cash available is the spot leg —
so you sell spot AND buy back perp in equal measure to stay neutral. The position gets smaller exactly
when funding is richest, and you pay two taker fees for the privilege. A model that assumes constant
notional through a bull market is measuring a position it could not have held.

THE THIRD IS CHURN AGAINST THE EXIT RULE. A full cycle is four taker legs (buy spot, sell perp, buy back
perp, sell spot). At 0.05% per leg that is 0.20% of notional. Binance settles funding 3x/day, so at a
0.01% rate you earn 0.03%/day of notional: a round trip costs SIX AND A HALF DAYS of carry. Any exit
rule that flips position more than about twice a month hands the entire edge to the venue.

WHAT IS NOT MODELLED, STATED PLAINLY, BECAUSE IT ALL CUTS THE SAME WAY:
  * Exchange insolvency / withdrawal freeze. Unquantifiable and total. FTX paid excellent funding.
  * Slippage beyond the taker fee. At $5k notional on BTC/ETH this is genuinely negligible; it would
    not be on an altcoin.
  * Spot borrow or USDC yield on idle margin (a credit, so excluding it is conservative).
  * Gas / DEX settlement costs on Hyperliquid or dYdX.
So every number this file prints is an UPPER bound on what was achievable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TBILL = 0.04                 # risk-free alternative; carry must beat this to be worth any risk
MAINT_MARGIN = 0.005         # 0.5% maintenance margin on notional (BTC/ETH tier)


@dataclass
class Result:
    label: str
    days: int
    days_in_market: int
    cycles: int
    topups: int
    funding_received: float
    fees_paid: float
    basis_pnl: float
    net_profit: float
    capital: float
    ann_return: float
    ann_vol: float
    sharpe: float
    max_dd: float
    excess_ann: float
    equity: pd.Series = field(repr=False, default=None)

    def row(self) -> dict:
        return {
            "scenario": self.label,
            "yrs": round(self.days / 365, 2),
            "in_mkt%": round(100 * self.days_in_market / max(self.days, 1), 1),
            "cycles": self.cycles,
            "topups": self.topups,
            "funding$": round(self.funding_received),
            "fees$": round(self.fees_paid),
            "basis$": round(self.basis_pnl),
            "net$": round(self.net_profit),
            "ann%": round(100 * self.ann_return, 2),
            "vol%": round(100 * self.ann_vol, 2),
            "Sharpe": round(self.sharpe, 2),
            "maxDD%": round(100 * self.max_dd, 2),
            "vs 4% T-bill": round(100 * self.excess_ann, 2),
        }


def _daily_funding(fund: pd.DataFrame) -> pd.DataFrame:
    """Collapse funding settlements to a daily total rate and a trailing 72h mean settlement rate."""
    f = fund.copy()
    f["date"] = f.time.dt.date
    per_day = f.groupby("date").agg(day_rate=("rate", "sum"),
                                    n_settle=("rate", "size"),
                                    interval_h=("interval_h", "first"))
    # 72-hour trailing mean of the SETTLEMENT rate, which is what the exit rule is specified on.
    # THE WINDOW MUST BE DERIVED FROM THE SETTLEMENT INTERVAL, NOT HARDCODED. 72 hours is 9 settlements
    # on Binance's 8h schedule but 72 on Hyperliquid's hourly one. A fixed 9 silently became a NINE-HOUR
    # average on Hyperliquid, which is a different rule: it tracks noise instead of regime and tripled
    # the round trips (77 cycles in 1.9 years instead of 25), handing the churn straight to the venue.
    # Same unit trap as annualisation, one level down.
    iv = float(f.interval_h.mode()[0])
    win = max(2, int(round(72.0 / iv)))
    f["ma72"] = f.rate.rolling(win, min_periods=max(2, win // 3)).mean()
    per_day["ma72"] = f.groupby("date")["ma72"].last()
    return per_day.reset_index()


def backtest(fund: pd.DataFrame, px: pd.DataFrame, *, capital: float = 5000.0,
             buffer: float = 0.40, taker_fee: float = 0.0005,
             exit_threshold: str = "tbill", topup_at: float = 0.5,
             label: str = "") -> Result:
    """Simulate the delta-neutral carry day by day.

    `buffer`         margin as a fraction of notional; the adverse move survivable without liquidation.
    `taker_fee`      per leg, per side. A full cycle pays this four times.
    `exit_threshold` 'tbill' = unwind when trailing 72h funding < the T-bill equivalent per settlement.
                     'zero'  = unwind only when funding actually goes negative.
    `topup_at`       when margin equity falls to this fraction of its required level, de-lever to restore
                     it: sell spot and buy back perp in equal measure, two taker fees, smaller position.
    """
    d = _daily_funding(fund)
    p = px.copy()
    p["date"] = pd.to_datetime(p.date).dt.date
    m = d.merge(p[["date", "perp", "spot"]], on="date", how="inner").sort_values("date")
    m = m.dropna(subset=["perp", "spot"]).reset_index(drop=True)
    if m.empty:
        raise RuntimeError("no overlapping funding/price dates")

    settles_per_yr = 8760.0 / float(m.interval_h.iloc[0])
    thresh = (TBILL / settles_per_yr) if exit_threshold == "tbill" else 0.0

    cash = capital          # uninvested capital (also the margin pool when in position)
    qty = 0.0               # units of the asset held long spot (= units short perp)
    entry_perp = 0.0
    entry_spot = 0.0        # needed for basis attribution; see the exit block
    margin = 0.0
    in_mkt = False
    funding_tot = fees_tot = basis_tot = 0.0
    cycles = topups = days_in = 0
    equity_path, dates = [], []

    for _, r in m.iterrows():
        spot, perp, day_rate, ma = r.spot, r.perp, r.day_rate, r.ma72
        want_in = bool(pd.notna(ma) and ma >= thresh)

        # ---- ENTER ------------------------------------------------------------------
        if want_in and not in_mkt:
            # Solve S from C = S + b*S, then buy spot and short perp at that notional.
            notional = cash / (1.0 + buffer)
            qty = notional / spot
            margin = cash - notional
            fee = 2.0 * taker_fee * notional          # buy spot + sell perp
            fees_tot += fee
            margin -= fee
            entry_perp, entry_spot = perp, spot
            cash = 0.0
            in_mkt = True
            cycles += 1

        # ---- HOLD: funding, then margin maintenance --------------------------------
        if in_mkt:
            days_in += 1
            notional = qty * perp
            # Short the perp: POSITIVE funding is received. day_rate already sums the day's settlements.
            f_pnl = day_rate * notional
            funding_tot += f_pnl
            margin += f_pnl

            # Unrealised loss on the short as price moves up; the spot leg gains it back but that gain
            # is not in the margin account, which is the account that gets liquidated.
            short_upnl = qty * (entry_perp - perp)
            equity = margin + short_upnl
            required = buffer * notional
            if equity < topup_at * required and qty > 0:
                # De-lever to restore the buffer: sell spot dq and buy back perp dq, staying neutral.
                short = topup_at * required - equity
                dq = min(qty, (short / max(spot, 1e-9)) * (1.0 + buffer))
                fee = 2.0 * taker_fee * dq * spot
                fees_tot += fee
                # Realise the short leg P&L on the closed slice and bank the spot proceeds as margin.
                # Basis attribution needs BOTH entry legs: total non-funding P&L on a closed quantity is
                # dq * (basis_entry - basis_exit) with basis = perp - spot, which is identically the spot
                # gain plus the short gain. An earlier version approximated basis_entry using only
                # entry_perp scaled by spot/perp, because entry_spot was not stored — an expression that
                # happens to be near-zero here (mean basis is ~2bp) and so looked plausible while being
                # unjustifiable. Store the price and do the arithmetic.
                basis_tot += dq * ((entry_perp - entry_spot) - (perp - spot))
                margin += dq * spot + dq * (entry_perp - perp) - fee
                qty -= dq
                topups += 1

            # ---- EXIT -------------------------------------------------------------
            if not want_in:
                notional = qty * perp
                fee = 2.0 * taker_fee * notional      # buy back perp + sell spot
                fees_tot += fee
                short_pnl = qty * (entry_perp - perp)
                cash = margin + qty * spot + short_pnl - fee
                basis_tot += qty * ((entry_perp - entry_spot) - (perp - spot))
                qty, margin, in_mkt = 0.0, 0.0, False

        # ---- mark equity -----------------------------------------------------------
        if in_mkt:
            eq = margin + qty * spot + qty * (entry_perp - perp)
        else:
            eq = cash
        equity_path.append(eq)
        dates.append(r.date)

    # Mark any still-open position's basis to market so the decomposition CLOSES: terminal equity minus
    # capital must equal funding + basis - fees exactly. Without this the open position's unrealised
    # basis sits in equity but not in basis_tot, and the parts silently fail to sum to the whole by
    # whatever the open trade happens to be carrying.
    if in_mkt:
        last = m.iloc[-1]
        basis_tot += qty * ((entry_perp - entry_spot) - (last.perp - last.spot))

    eq = pd.Series(equity_path, index=pd.to_datetime(dates))
    ret = eq.pct_change().dropna()
    days = len(m)
    yrs = days / 365.0
    net = float(eq.iloc[-1]) - capital
    ann = (float(eq.iloc[-1]) / capital) ** (1 / yrs) - 1 if yrs > 0 and eq.iloc[-1] > 0 else -1.0
    vol = float(ret.std() * np.sqrt(365)) if len(ret) > 2 else 0.0
    sharpe = ((ann - TBILL) / vol) if vol > 1e-9 else 0.0
    dd = float((eq / eq.cummax() - 1).min())

    return Result(label=label or f"b={buffer:.0%} fee={taker_fee:.2%} exit={exit_threshold}",
                  days=days, days_in_market=days_in, cycles=cycles, topups=topups,
                  funding_received=funding_tot, fees_paid=fees_tot, basis_pnl=basis_tot,
                  net_profit=net, capital=capital, ann_return=ann, ann_vol=vol,
                  sharpe=sharpe, max_dd=dd, excess_ann=ann - TBILL, equity=eq)


def gross_yield_table(fund: pd.DataFrame, label: str) -> None:
    """What the funding stream looks like BEFORE any of this trade's structure is applied."""
    iv = float(fund.interval_h.mode()[0])
    spy = 8760.0 / iv
    r = fund.rate
    by_yr = fund.assign(y=fund.time.dt.year).groupby("y").rate.agg(["mean", "std", "size"])
    print(f"\n{label}: raw funding, {len(r):,} settlements at {iv:.0f}h "
          f"({fund.time.min().date()}..{fund.time.max().date()})")
    print(f"  mean {r.mean():.6%}/settle = {r.mean()*spy:.2%}/yr gross on NOTIONAL   "
          f"positive {100*(r>0).mean():.1f}% of settlements")
    print("  by year:  " + "   ".join(
        f"{int(y)}: {v['mean']*spy:+.1%}" for y, v in by_yr.iterrows()))
