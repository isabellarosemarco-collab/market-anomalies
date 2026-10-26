"""
run_vrp.py — the variance risk premium study end to end.

    python -m vrp_alpha.run_vrp

SECTION 0  the short-vol ETP gate: can the brief's VXX/SVXY/UVXY route be backtested at all?
SECTION 1  the pure premium, both definitions, with HAC and non-overlapping statistics, by regime.
SECTION 2  the tradeable harvest: filtered short variance, marked daily, against SPY buy-and-hold.
SECTION 3  the tail: what the gate can and cannot stop, and what one bad month costs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from vrp_alpha import data, strategy, vrp

COST = 0.5          # vol points per swap, round trip
REGIMES = [("2010-2019 (the brief's calm sample)", "2010-01-01", "2019-12-31"),
           ("2020-2026 (covid, 2022, Aug 2024)", "2020-01-01", "2026-12-31"),
           ("1990-2009 (excluded by the brief)", "1990-01-01", "2009-12-31")]


def section0() -> None:
    print("=" * 112)
    print("0. DATA GATE")
    print("=" * 112)
    data.etp_gate()


def section1(d: pd.DataFrame) -> None:
    print("\n" + "=" * 112)
    print("1. THE PURE PREMIUM — is VIX above realized vol, and by how much, and does it mean anything?")
    print("=" * 112)
    full = d.loc["2010":]
    print("  THE BRIEF'S DEFINITION AND THE TRADEABLE ONE, SIDE BY SIDE (2010-present, daily observations).")
    print("  The first is VIX minus TRAILING realized vol. The second is VIX minus the realized vol that")
    print("  FOLLOWS. Only the second is a premium anyone can collect.\n")
    vrp.describe(full.vrp_bwd.values, "VIX - trailing RV21 (brief's spelling)")
    vrp.describe(full.vrp_fwd.values, "VIX - FORWARD RV21 (tradeable)")
    vrp.describe(full.vswap_bp.values, "variance-swap P&L, (K^2-RV^2)/2K")
    print("\n  The third row is the same forward premium priced as the instrument that actually matches VIX.")
    print("  It is smaller than the linear spread and far more negatively skewed, because a short variance")
    print("  swap's loss grows with the SQUARE of realized vol. That gap is the whole risk of the trade.")

    print("\n  NON-OVERLAPPING MONTHLY SAMPLES — the check that does not depend on the HAC kernel.")
    print("  Daily forward-looking windows share 20 of 21 days, so the ~4,000 daily points are about 190")
    print("  independent months. Below, each observation is a distinct month; all 21 possible grid offsets")
    print("  are reported because a single offset is a choice nobody should have to trust.")
    for nm, col in (("VIX - forward RV21", "vrp_fwd"), ("variance-swap P&L", "vswap_bp")):
        ms = [vrp.monthly(full[col], o) for o in range(vrp.WIN)]
        means = np.array([m.mean() for m in ms])
        ts = np.array([m.mean() / (m.std(ddof=1) / np.sqrt(len(m))) for m in ms])
        print(f"    {nm:26} n~{int(np.mean([len(m) for m in ms])):4} months   "
              f"mean {means.mean():+6.2f} (offsets {means.min():+.2f} to {means.max():+.2f})   "
              f"t {ts.mean():+5.2f} (offsets {ts.min():+.2f} to {ts.max():+.2f})")

    print("\n  BY REGIME. The brief's window starts in 2010, which is a choice with consequences.")
    for nm, a, b in REGIMES:
        sub = d.loc[a:b]
        if len(sub) < 200:
            continue
        print(f"\n    {nm}  ({sub.index[0].date()} .. {sub.index[-1].date()})")
        vrp.describe(sub.vrp_bwd.values, "      VIX - trailing RV")
        vrp.describe(sub.vrp_fwd.values, "      VIX - forward RV")
        vrp.describe(sub.vswap_bp.values, "      variance-swap P&L")
    print("\n  THE 2008 OMISSION MATTERS, BUT NOT IN THE DIRECTION EVERY SHORT-VOL WARNING ASSUMES, AND THIS")
    print("  IS THE MOST COUNTERINTUITIVE NUMBER IN THE STUDY. The obvious objection to a 2010 start date is")
    print("  that it excludes the financial crisis, so the sample flatters the trade. Measured, the opposite")
    print("  is true: 1990-2009 INCLUDING 2008 is the better half, and section 2's full-history panel gives")
    print("  it a Sharpe. The reason is visible in the regime rows above -- in 2008 volatility rose ahead of")
    print("  itself, so swaps kept being struck at high VIX into realized vol that, however awful, was")
    print("  already partly priced. The catastrophe for this trade is not high volatility; it is volatility")
    print("  exploding from a CALM base, because that is when the strike you sold is low. On 2020-02-14 VIX")
    print("  closed at 13.7 against a forward realized vol of 84.2. All twelve of the worst single days in")
    print("  thirty-six years are in Feb-Mar 2020, and none are in 2008.")


def section2(d: pd.DataFrame) -> None:
    print("\n" + "=" * 112)
    print("2. THE TRADEABLE HARVEST — filtered short variance, marked daily, vs SPY buy-and-hold")
    print("=" * 112)
    sub = d.loc["2010":].copy()
    spy_r = np.log(sub.spx / sub.spx.shift(1)) * 100      # in percent, comparable units to vol points
    print("  P&L is in vega-normalized volatility points. Sharpe, maxDD and Calmar are all leverage-")
    print("  invariant, so no margin assumption is made anywhere — see strategy.perf for why that matters.\n")
    variants = [
        ("unfiltered short variance", dict(vix_cap=1e9, require_vrp=False)),
        ("gate: VRP>0 and VIX<30", dict(vix_cap=30.0, require_vrp=True)),
        ("gate + liquidate on trip", dict(vix_cap=30.0, require_vrp=True, liquidate_on_gate=True)),
        ("gate, VIX<20 (tighter)", dict(vix_cap=20.0, require_vrp=True)),
    ]
    res = {}
    for nm, kw in variants:
        L = strategy.ladder(sub, cost_volpts=COST, **kw)
        res[nm] = (L, strategy.perf(L.pnl, label=nm))
    strategy.perf(spy_r, label="SPY buy-and-hold (log %/day)")
    print("\n  DAYS IN THE MARKET, which is how the filter buys its Sharpe:")
    for nm, (L, _) in res.items():
        print(f"    {nm:34} gate open {100*L.gate.mean():4.0f}% of days, "
              f"{int(L.opened.sum()):5} swaps opened")

    print("\n  COST SENSITIVITY on the gated variant. One vol point of round-trip cost is a realistic upper")
    print("  bound for a 1-month SPX variance swap or an ATM straddle plus daily delta-hedge slippage.")
    for c in (0.0, 0.25, 0.5, 1.0, 2.0):
        L = strategy.ladder(sub, vix_cap=30.0, require_vrp=True, cost_volpts=c)
        p = strategy.perf(L.pnl)
        print(f"    cost {c:4.2f} vol pts   ann {p['ann_mean']:+7.2f}  Sharpe {p['sharpe']:+5.2f}  "
              f"maxDD {p['maxdd']:+8.2f}  Calmar {p['calmar']:+5.2f}")

    print("\n  OUT-OF-SAMPLE SPLIT. The filter's thresholds (VRP>0, VIX<30) were not fitted here, but they")
    print("  are conventional numbers chosen with full knowledge of 2018 and 2020, so the split still matters.")
    for nm, a, b in (("2010-2017 (in-sample-ish)", "2010-01-01", "2017-12-31"),
                     ("2018-2026 (after volmageddon)", "2018-01-01", "2026-12-31")):
        s2 = d.loc[a:b]
        L = strategy.ladder(s2, vix_cap=30.0, require_vrp=True, cost_volpts=COST)
        strategy.perf(L.pnl, label=nm)

    print("\n  THE FULL HISTORY, WHICH REVERSES THE USUAL WARNING. Same gated strategy, same cost, run over")
    print("  the two halves of the available data. The half containing the financial crisis is the good half.")
    for nm, a, b in (("1990-2009 (incl. 2008)", "1990-01-01", "2009-12-31"),
                     ("2010-2026 (the brief's window)", "2010-01-01", "2026-12-31"),
                     ("1990-2026 (everything)", "1990-01-01", "2026-12-31")):
        s2 = d.loc[a:b]
        L = strategy.ladder(s2, vix_cap=30.0, require_vrp=True, cost_volpts=COST)
        strategy.perf(L.pnl, label=nm)
    ep = strategy.ladder(d.loc["2008-08-01":"2008-12-31"], vix_cap=30.0, require_vrp=True, cost_volpts=COST)
    v8 = d.loc["2008-08-01":"2008-12-31"].vix
    print(f"    for scale: 2008-08..2008-12 cost {ep.pnl.sum():+.1f} vol pts (worst day {ep.pnl.min():+.1f}) "
          f"while VIX went {v8.iloc[0]:.0f} -> {v8.max():.0f}")
    print("    March 2020 cost nearly four times as much (section 3) and the two episodes peaked at almost")
    print("    exactly the same VIX, 81 against 83. What differed was the floor they started from: swaps were")
    print("    already being struck in the 20s in 2008, and at 14 in February 2020.")


def section2b(d: pd.DataFrame) -> None:
    print("\n" + "=" * 112)
    print("2b. DOES THE BRIEF'S FILTER CARRY ANY INFORMATION? — the decisive test, and it is one line of math")
    print("=" * 112)
    sub = d.loc["2010":].dropna(subset=["vswap_bp", "vrp_bwd"])
    y = sub.vswap_bp
    print("  Every variant above differs only in WHEN it is willing to sell variance. So before comparing")
    print("  Sharpes, ask whether the gate's inputs predict the payoff at all. Splitting the same payoff by")
    print("  each gate condition:\n")
    for lbl, mask in (("VRP_bwd > 0  (filter says GO)", sub.vrp_bwd > 0),
                      ("VRP_bwd <= 0 (filter says STOP)", sub.vrp_bwd <= 0),
                      ("VIX < 30     (filter says GO)", sub.vix < 30),
                      ("VIX >= 30    (filter says STOP)", sub.vix >= 30),
                      ("full gate PASSES", (sub.vrp_bwd > 0) & (sub.vix < 30)),
                      ("full gate FAILS", ~((sub.vrp_bwd > 0) & (sub.vix < 30)))):
        vrp.describe(y[mask].values, "    " + lbl)
    print("\n  CORRELATIONS, which say the same thing without the subsetting:")
    print(f"    corr(VRP_bwd, variance-swap payoff) = {y.corr(sub.vrp_bwd):+.3f}")
    print(f"    corr(VIX,     variance-swap payoff) = {y.corr(sub.vix):+.3f}")
    print(f"    corr(VRP_bwd, VRP_fwd)              = {sub.vrp_bwd.corr(sub.vrp_fwd):+.3f}")
    print("""
  THE GATE IS NOT MERELY USELESS, IT SELECTS THE WORSE HALF, AND THE REASON IS MECHANICAL. The days the
  gate rejects pay MORE than the days it accepts, and they pay it with a SHALLOWER tail. That is not noise
  in one subsample; it follows from what VIX is. VIX is the strike. Selling variance at VIX 35 is selling
  insurance dear, and volatility mean-reverts downward from panic, so those are the best sales anyone ever
  makes. Selling at VIX 12 is selling it cheap with the whole distance to the upside still available. The
  brief's rule -- sell only when things are calm -- systematically sells at the lowest strikes and stands
  aside at the highest, which is the wrong side of the only variable that matters.

  It gets worse on the risk side. The -252 worst observation sits INSIDE the gate-passes bucket and the
  gate-fails bucket's worst is -120. The tail-risk gate does not reduce tail risk; the tail lives almost
  entirely among the trades the gate approves, because a catastrophe requires a low strike and a low strike
  is exactly what the gate insists on. And the VRP>0 condition cannot help even in principle: it is
  measured above at a correlation of +0.003 with the forward premium it is standing in for. Conditioning on
  a variable independent of the payoff changes the number of trades and nothing else.""")


def section3(d: pd.DataFrame) -> None:
    print("\n" + "=" * 112)
    print("3. THE TAIL — what the gate cannot stop")
    print("=" * 112)
    sub = d.loc["2010":].copy()
    L = strategy.ladder(sub, vix_cap=30.0, require_vrp=True, cost_volpts=COST)
    tot = L.pnl.sum()
    print(f"  Total P&L 2010-present, gated: {tot:+.1f} vol points over {len(L)} sessions.\n")
    print("  THE WORST 21-DAY STRETCHES, gated strategy:")
    w = strategy.worst_episodes(L.pnl, 6)
    for _, r in w.iterrows():
        print(f"    to {r['end']}   {r['21d_pnl']:+9.2f} vol pts   "
              f"= {100*r['21d_pnl']/tot:+6.1f}% of everything the strategy ever made")
    print("\n  WHAT THE GATE ACTUALLY PREVENTED. The gate blocks OPENING when VIX>30. It cannot close the")
    print("  swap sold three days earlier at VIX 14, and volatility gaps rather than ramping through the")
    print("  trigger. Comparing the two gated variants isolates that:")
    for nm, kw in (("gate only (as specified)", dict()),
                   ("gate + liquidate positions", dict(liquidate_on_gate=True))):
        LL = strategy.ladder(sub, vix_cap=30.0, require_vrp=True, cost_volpts=COST, **kw)
        p = strategy.perf(LL.pnl)
        print(f"    {nm:30} ann {p['ann_mean']:+7.2f}  Sharpe {p['sharpe']:+5.2f}  "
              f"maxDD {p['maxdd']:+8.2f}  worst day {p['worst_day']:+7.2f}")

    print("\n  THE KEY EVENTS, day by day, gated strategy:")
    for nm, a, b in (("Feb 2018 volmageddon", "2018-01-29", "2018-02-12"),
                     ("Mar 2020 covid", "2020-02-19", "2020-03-25"),
                     ("Aug 2024 yen carry unwind", "2024-07-29", "2024-08-09")):
        ep = L.loc[a:b]
        if ep.empty:
            continue
        print(f"    {nm:28} {a} .. {b}   P&L {ep.pnl.sum():+8.2f} vol pts   "
              f"worst day {ep.pnl.min():+7.2f}   VIX {sub.loc[a:b].vix.min():.0f} -> "
              f"{sub.loc[a:b].vix.max():.0f}")

    print("\n  HOW MANY GOOD MONTHS ONE BAD MONTH COSTS:")
    m = vrp.monthly(L.pnl.rolling(vrp.WIN).sum().dropna())
    pos = m[m > 0]
    if len(pos) and m.min() < 0:
        print(f"    median positive month {pos.median():+.2f} vol pts; worst month {m.min():+.2f}; "
              f"ratio {abs(m.min()/pos.median()):.0f}x")
        print(f"    so one worst-case month erases about {abs(m.min()/pos.median()):.0f} median good months.")


def verdict() -> None:
    print("\n" + "=" * 112)
    print("VERDICT")
    print("=" * 112)
    print("""
  THE PREMIUM IS REAL. IT IS NOT AN EDGE. Those are different claims and the study separates them.

  1. THE PURE PREMIUM TEST — CONFIRMED, AND IT IS THE ONLY HYPOTHESIS THAT SURVIVES.
     VIX exceeds trailing RV21 on 87% of days since 2010, mean +3.73 vol points, HAC t +12.76. The brief
     asked for that number and it is as strong as advertised. But the tradeable version -- VIX against the
     realized vol that FOLLOWS -- is what a seller collects, and priced as the instrument VIX actually
     represents, (K^2-RV^2)/2K, the premium is +2.04 vol points at HAC t +2.67. Still positive, roughly
     half the headline, and the t-stat falls by a factor of five once overlapping windows are paid for.
     Non-overlapping monthly sampling across all 21 grid offsets agrees on the mean (+2.04) but its t-stat
     ranges from +1.10 to +4.04 depending purely on which day of the month you sample -- so "significant"
     is a function of an arbitrary choice. Stability across regimes fails the brief's own test: 2010-2019
     mean +2.45 at t +5.39, 2020-2026 mean +1.43 at t +0.80. The premium did not merely shrink after 2020,
     it stopped being measurable.

  2. THE TRADEABLE PROXY WITH REGIME FILTER — REJECTED, TWICE OVER, AND THE SECOND REASON IS FATAL.
     The ETP route cannot be tested at all: XIV is absent from the data because it died, SVXY and UVXY both
     re-levered in Feb 2018 so their price histories splice two different funds, and the feed's VXX starts
     in 2018. Any backtest on the survivors is conditioned on not having blown up.
     Built synthetically instead, the gated harvest earns ann +13.31 at Sharpe +0.57, maxDD -133.83,
     Calmar +0.10, with skew -9.1 and kurtosis 142.6. SPY buy-and-hold over the same window: Sharpe +0.67,
     Calmar +0.28, skew -0.6. Short variance loses to doing nothing on both risk-adjusted measures, by 3x
     on Calmar, while carrying a fifteenth the symmetry. There is no version of this that is worth the
     operational burden of rolling a monthly swap.
     And the filter itself is backwards. The days the gate rejects pay +2.92 against the +1.85 it accepts,
     the worst single observation in the sample sits inside the gate-approved bucket, and the correlation
     between the gate's signal and its own payoff is +0.002. The gate cannot select good trades because
     VIX is the strike: it is designed to sell cheap insurance and refuse the expensive kind.

  3. EXECUTION AND FRICTIONS — THE STRATEGY DIES WELL INSIDE REALISTIC COST.
     At zero cost the gated harvest prints Sharpe +0.78. At 0.5 vol points, +0.57. At 1.0, +0.36. At 2.0
     vol points -- a defensible round-trip for a retail-accessible monthly variance exposure once the
     straddle spread and daily delta-hedge slippage are counted -- it is -0.06, i.e. the entire premium is
     the transaction cost. A strategy whose survival depends on institutional execution is not available at
     the size this account trades.
     Out of sample it decays without needing any cost assumption: 2010-2017 Sharpe +1.07, Calmar +0.50;
     2018-2026 Sharpe +0.26, Calmar +0.05.

  ANSWERING THE QUESTION AS ASKED: does the VRP survive out-of-sample as an investable premium, or does
  tail risk eat the entire historical spread?
     Tail risk eats it, and the arithmetic is not close. The worst 21-day stretch, ending 2020-03-18, cost
     -123.13 vol points against +222.2 earned across all 4,207 sessions -- one month erased 55% of
     everything the strategy ever made, and the six worst stretches in sixteen years are all the same
     month. One worst month costs about 19 median good months. The gate cannot prevent this: it blocks
     opening at VIX>30, but the damage is done by the swap sold at VIX 14 three days earlier, and adding
     forced liquidation halves the drawdown (-62 vs -134) while leaving the worst DAY identical at -29.70.
     Volatility gaps through the trigger; there is no exit at the level you chose.

  THE FINDING WORTH KEEPING, WHICH IS NOT THE ONE THE BRIEF EXPECTED. The instinct that a 2010 start date
  flatters a short-vol study is wrong. 1990-2009, crisis included, runs at Sharpe +2.09 and Calmar +0.82
  against +0.57 and +0.10 for 2010-2026; the whole autumn of 2008 cost -31.6 vol points while VIX went 23
  to 81, and every one of the twelve worst days in thirty-six years falls in Feb-Mar 2020. High vol is
  not the risk to selling volatility -- high volatility means high strikes. The risk is a violent repricing
  from a calm base, which is unforecastable by construction, because if it were forecastable the strike
  would not have been calm. That is why the premium exists and why no regime filter built from VIX and
  trailing RV can capture it.

  WHERE THIS LEAVES THE THREE STUDIES. Index reconstitution: the flow is real, the alpha is arbitraged.
  Gamma pinning: the mechanism was misidentified as drift; a placebo grid on prices where no option is
  listed reproduced the whole result. VRP: the premium is measurable, positive, and genuinely economic --
  it is compensation for exactly the risk that showed up in March 2020 -- and it still loses to holding
  SPY. Three for three, and each failed for a different reason, which is more informative than three
  variations of the same one. The pattern is worth naming: every one of these had a correct mechanism
  attached to an untradeable implementation, and in all three cases the disqualifying fact was cheap to
  measure and available before the pipeline was built.""")


def main() -> int:
    print("=" * 112)
    print("VARIANCE RISK PREMIUM — is IV minus RV an investable premium, or rent paid for tail exposure?")
    print("=" * 112)
    d = vrp.build(data.load())
    print(f"  data: {d.index[0].date()} .. {d.index[-1].date()}, {len(d)} sessions; "
          f"VIX from CBOE's official history, returns from ^SPX")
    section0()
    section1(d)
    section2(d)
    section2b(d)
    section3(d)
    verdict()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
