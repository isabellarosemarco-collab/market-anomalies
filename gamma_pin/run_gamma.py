"""
run_gamma.py — the OpEx gamma-pinning study end to end.

    python -m gamma_pin.run_gamma

THREE SECTIONS, ANSWERING THE THREE QUESTIONS IN THE ORDER THEY CAN ACTUALLY BE ANSWERED.

  A. HISTORICAL CONVERGENCE, two years of hourly bars, five names, expiration Fridays against control
     weekdays. This is the superset test described in pin.py: it asks whether spot converges on the $5
     strike grid at all. Answers question 1, and question 2 by subtracting the fee.

  B. K* VALIDATION on a live chain with CBOE's own gamma. Checks whether the brief's gamma-weighted-OI
     strike is in fact the nearest strike, i.e. whether section A's grid proxy is the same rule.

  C. OPTION SPREAD COST, measured on real 0DTE quotes captured intraday today with the realised 16:00
     settlement known. Answers question 3 in the only honest way available: not with a slippage
     assumption but with the actual bid, the actual ask, and the actual outcome.

TODAY IS 2026-09-25, AN EXPIRATION FRIDAY, AND THAT IS WHY SECTION C EXISTS AT ALL. Two snapshots were
taken: the option chain at 12:10 ET with roughly four hours to expiry, and the CBOE chain after the close.
Options data is perishable in a way price data is not — an expired chain simply stops being published —
so the intraday capture is not reproducible tomorrow and the section is written to use what was saved.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from gamma_pin import cboe, pin

FEE_BP = 2.0                    # the brief's upper bound on an equity round trip
SNAP = "/tmp/gamma_live.pkl"


def section_a() -> None:
    print("=" * 108)
    print("A. DOES SPOT CONVERGE ON THE STRIKE GRID? (the superset test — see pin.py for why it is a superset)")
    print("=" * 108)
    px = pin.load()
    days = pin.build(px)
    if not days:
        print("  no bars")
        return
    lo, hi = min(d.date for d in days), max(d.date for d in days)
    print(f"  {len(days)} name-days, {lo.date()} .. {hi.date()}, 60-minute bars (the longest intraday")
    print(f"  history yfinance serves). Reference price is the 09:30 bar's close, i.e. ~10:30 ET.\n")

    band = pin.in_band(days)
    print(f"  after the brief's distance filter |S0-K|/S0 in [0.25%, 1.25%]: {len(band)} of {len(days)}\n")

    fri = [d for d in band if d.weekday == 4]
    ctl = [d for d in band if d.weekday != 4]
    mon = [d for d in fri if d.is_monthly]
    wk = [d for d in fri if not d.is_monthly]

    print("  PANEL 1 — expiration Fridays against the control group of non-Fridays.")
    print("  'converge' is the brief's distance reduction. 'delta-trade' is the P&L of actually trading it.")
    r_f = pin.stats(fri, "Fridays (all expirations)")
    r_c = pin.stats(ctl, "Mon-Thu (control, no expiry)")
    r_m = pin.stats(mon, "third Friday (monthly, big OI)")
    r_w = pin.stats(wk, "other Fridays (weekly)")

    if r_f and r_c:
        # THE ONLY COMPARISON THAT MEANS ANYTHING. Question 1 asks whether convergence beats 50%, but
        # under a driftless walk it cannot reach 50% (see pin.py), so the control group IS the null.
        d = r_f["conv"] - r_c["conv"]
        se = np.sqrt(np.var([x.d_reduction_bp for x in fri], ddof=1) / len(fri) +
                     np.var([x.d_reduction_bp for x in ctl], ddof=1) / len(ctl))
        print(f"\n  FRIDAY MINUS CONTROL, convergence: {d:+.1f} bp   t = {d/se:+.2f}")
        print(f"  the random-walk null measured from control days: {100*r_c['pct_conv']:.0f}% of days "
              f"converge, NOT 50%.")
        print(f"  Friday: {100*r_f['pct_conv']:.0f}%. The brief's '>50%' test would have been passed or "
              f"failed by the shape")
        print(f"  of the absolute-value function, not by dealer hedging.")

        ds = r_f["signed"] - r_c["signed"]
        ses = np.sqrt(np.var([x.signed_bp for x in fri], ddof=1) / len(fri) +
                      np.var([x.signed_bp for x in ctl], ddof=1) / len(ctl))
        print(f"\n  FRIDAY MINUS CONTROL, the tradeable delta P&L: {ds:+.1f} bp   t = {ds/ses:+.2f}")

    print("\n  PANEL 2 — per name, expiration Fridays only. A real dealer effect should be strongest in")
    print("  single stocks with concentrated OI, and weakest in SPY where the gamma is spread over hundreds")
    print("  of strikes and a dozen expirations.")
    for n in pin.NAMES:
        pin.stats([d for d in fri if d.ticker == n], n)

    print("\n  PANEL 3 — question 2: the delta trade net of a 2 bp round-trip equity fee.")
    for lbl, grp in (("Fridays", fri), ("third Fridays only", mon)):
        if len(grp) < 10:
            continue
        sg = np.array([x.signed_bp for x in grp]) - FEE_BP
        t = sg.mean() / (sg.std(ddof=1) / np.sqrt(len(sg)))
        # Per-day Sharpe annualised on ~50 Fridays a year, not sqrt(252): these are discrete day trades.
        shp = sg.mean() / sg.std(ddof=1) * np.sqrt(len(grp) / ((max(d.date for d in grp) -
                                                               min(d.date for d in grp)).days / 365.25))
        print(f"    {lbl:22} n={len(sg):4}  net {sg.mean():+7.1f} bp  t {t:+5.2f}  "
              f"win {100*(sg>0).mean():3.0f}%  ann.Sharpe {shp:5.2f}  "
              f"sd {sg.std(ddof=1):6.1f} bp")
        print(f"    {'':22} the 2 bp fee is {100*FEE_BP/max(abs(sg.mean()+FEE_BP),1e-9):.0f}% of the "
              f"gross edge and {100*FEE_BP/sg.std(ddof=1):.1f}% of one standard deviation.")
    print("    The fee is NOT the binding constraint on this trade, and that matters more than it looks:")
    print("    it means the trade fails on the absence of drift, so no cost improvement can rescue it.")

    print("\n  PANEL 4 — the same numbers with the FIVE NAMES COLLAPSED INTO ONE OBSERVATION PER DATE.")
    print("  The five tickers are sampled on the same Fridays and share that day's market move, so the")
    print("  name-day counts above overstate the independent sample by roughly sqrt(5).")
    for lbl, grp in (("Fridays", fri), ("Mon-Thu control", ctl), ("third Fridays", mon),
                     ("weekly Fridays", wk)):
        if len(grp) < 10:
            continue
        cv, sg = pin.by_date(grp, "d_reduction_bp"), pin.by_date(grp, "signed_bp")
        tc = cv.mean() / (cv.std(ddof=1) / np.sqrt(len(cv)))
        ts = sg.mean() / (sg.std(ddof=1) / np.sqrt(len(sg)))
        print(f"    {lbl:18} dates={len(cv):4}  converge {cv.mean():+7.1f} bp (t {tc:+5.2f})   "
              f"delta-trade {sg.mean():+7.1f} bp (t {ts:+5.2f}, {100*(sg>0).mean():3.0f}% of dates)")
    dif, tt = pin.welch(pin.by_date(fri, "signed_bp"), pin.by_date(ctl, "signed_bp"))
    print(f"    Friday minus control on the delta trade, per date: {dif:+.1f} bp   t = {tt:+.2f}")
    dif2, tt2 = pin.welch(pin.by_date(mon, "signed_bp"), pin.by_date(wk, "signed_bp"))
    print(f"    Monthly minus weekly, per date: {dif2:+.1f} bp   t = {tt2:+.2f}")
    print("    THE DOSE-RESPONSE TEST, WHICH IS THE ONE THAT DECIDES IT. Third Fridays carry five to ten")
    print("    times the open interest of a weekly, so a real dealer-hedging effect must be LARGER there.")
    print("    If the monthly-minus-weekly difference is negative, the apparent Friday edge is not gamma.")

    print("\n  PANEL 5 — VARIANT 2 without option prices: the realised settlement distance |S_close - K|.")
    print("  A short straddle at K pays premium minus |S_close - K|, so for a given implied vol the whole")
    print("  case for selling it on expiration Friday is that this distance is SMALLER than on other days.")
    print("  That is testable on price data alone, which sidesteps the missing historical option quotes.")
    for lbl, grp in (("Fridays", fri), ("Mon-Thu control", ctl), ("third Fridays", mon),
                     ("weekly Fridays", wk)):
        if len(grp) < 10:
            continue
        s = pin.by_date(grp, "d1_bp")
        print(f"    {lbl:18} dates={len(s):4}  mean |close-K| {s.mean():7.1f} bp   median "
              f"{np.median(s):7.1f} bp")
    ds, tds = pin.welch(pin.by_date(fri, "d1_bp"), pin.by_date(ctl, "d1_bp"))
    print(f"    Friday minus control: {ds:+.1f} bp   t = {tds:+.2f}   "
          f"(negative = tighter settlement = the straddle seller's edge)")

    print("\n  PANEL 6 — THE PLACEBO. The identical trade against grids shifted off the listed strikes, so")
    print("  that they hold no open interest at all. No options exist there, so no dealer can be hedging them.")
    print("  If the edge survives, it is round-number arithmetic and not gamma. See pin.build's docstring.")
    print(f"    {'grid':24} {'Fri dates':>10} {'Fri delta':>11} {'t':>6} {'ctl delta':>11} "
          f"{'Fri-ctl':>9} {'t':>6}")
    for off, lbl in ((0.0, "$5 strikes (real)"), (1.3, "$5 + $1.30 (no strikes)"),
                     (0.65, "$5 + $0.65 (no strikes)"), (2.5, "$5 + $2.50 (exact inverse)")):
        b = pin.in_band(pin.build(px, offset=off))
        f2 = [d for d in b if d.weekday == 4]
        c2 = [d for d in b if d.weekday != 4]
        if len(f2) < 10 or len(c2) < 10:
            continue
        fs, cs = pin.by_date(f2, "signed_bp"), pin.by_date(c2, "signed_bp")
        tf = fs.mean() / (fs.std(ddof=1) / np.sqrt(len(fs)))
        dd, td = pin.welch(fs, cs)
        print(f"    {lbl:24} {len(fs):10} {fs.mean():+10.1f} {tf:+6.2f} {cs.mean():+10.1f} "
              f"{dd:+9.1f} {td:+6.2f}")
    # The $2.50 grid sits exactly between the $5 strikes, so "spot below the nearest point" flips for most
    # prices: 339.60 is below 340 but above 337.50. The two signals are therefore near-mechanical mirrors,
    # and quantifying that is what turns the sign flip from a puzzle into an explanation.
    ref = {(d.ticker, d.date): d for d in pin.build(px, 0.0)}
    alt = {(d.ticker, d.date): d for d in pin.build(px, 2.5)}
    both = [(ref[k], alt[k]) for k in ref if k in alt]
    agree = np.mean([(a.s_ref < a.k) == (b.s_ref < b.k) for a, b in both])
    print(f"\n    READ THE $2.50 ROW AS A CONSISTENCY CHECK, NOT AS EVIDENCE. Its grid points ARE the")
    print(f"    midpoints between the $5 strikes, so 'spot below the nearest point' flips on every single")
    print(f"    day: the two rules agree {100*agree:.0f}% of the time. Its result is forced to carry the opposite")
    print("    sign and tells us nothing independent. The magnitudes differ only because the distance filter")
    print("    selects different days for each grid. The informative placebos are the non-strike offsets.")
    print("\n    THE $0.65 ROW IS THE RESULT THAT ENDS THE STUDY. That grid lands on no listed strike in any")
    print("    of these five names, so there is no open interest on it and no dealer has any position to")
    print("    hedge there. It earns the same thing as the real strike grid, on the same days, with the same")
    print("    t-stat. An effect that is indifferent to whether options exist at the price cannot be caused")
    print("    by option dealers. The spread across offsets ($1.30 giving half as much) is the noise level")
    print("    of the measurement, which is itself the point: the differences between grids are larger than")
    print("    the distance from any of them to zero.")

    print("\n  PANEL 6b — the same trade by day of week. If Friday is special, it should stand alone.")
    for wd, nm in enumerate(("Mon", "Tue", "Wed", "Thu", "Fri")):
        g = [d for d in band if d.weekday == wd]
        if len(g) < 10:
            continue
        s = pin.by_date(g, "signed_bp")
        t_ = s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))
        print(f"    {nm:5} dates={len(s):4}  delta-trade {s.mean():+7.1f} bp  t {t_:+5.2f}  "
              f"{100*(s>0).mean():3.0f}% of dates positive")
    print("    It does not. Tuesday is as significant as Friday and points the other way. Across five")
    print("    weekdays and four grid offsets that is twenty tests of a signal with no theory behind it on")
    print("    nineteen of them, and finding one |t| above 2 in each direction is the expected yield of that")
    print("    search. Friday won a small lottery; it did not pass a test.")

    print("\n  PANEL 7 — is the Friday result stable in time, or is it a handful of dates?")
    fs = pin.by_date(fri, "signed_bp")
    dts = sorted({d.date for d in fri})
    mid_i = len(dts) // 2
    h1 = [d for d in fri if d.date <= dts[mid_i]]
    h2 = [d for d in fri if d.date > dts[mid_i]]
    for lbl, grp in ((f"first half (to {dts[mid_i].date()})", h1), ("second half", h2)):
        s = pin.by_date(grp, "signed_bp")
        t_ = s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))
        print(f"    {lbl:32} dates={len(s):4}  delta-trade {s.mean():+7.1f} bp  t {t_:+5.2f}")
    top = np.sort(fs)[::-1][:5]
    print(f"    the 5 best Fridays contribute {100*top.sum()/max(fs.sum(),1e-9):.0f}% of the total, "
          f"out of {len(fs)} dates")


def section_b() -> None:
    print("\n" + "=" * 108)
    print("B. IS K* THE NEAREST STRIKE? (CBOE's own gamma and open interest, live chain)")
    print("=" * 108)
    print("  Section A proxies K* with the nearest $5 grid point. If the brief's gamma-weighted-OI strike")
    print("  is usually that same strike, the proxy is tight. If not, the proxy is a DIFFERENT rule and")
    print("  section A can only falsify the family, not price the specific trade.\n")
    print(f"  {'name':6} {'spot':>9} {'K* (max gxOI)':>14} {'nearest':>9} {'match':>6} "
          f"{'|S-K*| bp':>10} {'in band':>8} {'K* share of band gamma':>23}")
    agree = 0
    tot = 0
    for n in pin.NAMES:
        try:
            df, spot, ts = cboe.chain(n)
        except Exception as e:
            print(f"  {n:6} fetch failed: {e}")
            continue
        exps = sorted(df.expiry.unique())
        # The front expiration is the one that pins; anything later has no expiry-day gamma.
        r = cboe.pin_strike(df, spot, pd.Timestamp(exps[0]))
        if not r:
            print(f"  {n:6} no chain in band")
            continue
        tot += 1
        same = abs(r["kstar"] - r["nearest"]) < 1e-9
        agree += same
        inb = 1e4 * pin.LO <= r["dist_bp"] <= 1e4 * pin.HI
        print(f"  {n:6} {spot:9.2f} {r['kstar']:14.1f} {r['nearest']:9.1f} {str(same):>6} "
              f"{r['dist_bp']:10.1f} {str(inb):>8} {100*r['share_of_band_gamma']:22.0f}%")
    if tot:
        print(f"\n  K* equals the nearest strike in {agree} of {tot} names. Where they differ, the brief's")
        print("  rule points at a strike the grid proxy never selects, so the two are not interchangeable.")
    print("  Note the last column: if one strike holds only a modest share of the gamma inside +-1.5%,")
    print("  there is no single dominant magnet to be pulled toward in the first place.")


def section_c() -> None:
    print("\n" + "=" * 108)
    print("C. DO OPTION SPREADS EAT THE DECAY? (real 0DTE quotes, 2026-09-25, with the outcome known)")
    print("=" * 108)
    try:
        snap = pickle.load(open(SNAP, "rb"))
    except OSError:
        print(f"  {SNAP} missing — the intraday capture cannot be recreated once the chain expires.")
        return
    import yfinance as yf

    print("  Captured at 12:10 ET with ~4h to expiry. The trade: sell the straddle at the nearest strike.")
    print("  You RECEIVE the bid on both legs, not the mid. Maximum possible gain is the full mid premium")
    print("  (spot settling exactly on the strike). So mid-minus-bid, as a share of mid, is the fraction of")
    print("  the entire theoretical decay handed to the market maker before the position has any view.\n")
    rows = []
    for n, d in snap.items():
        ch, s0 = d["chain"], d["S"]
        k = round(s0 / pin.GRID) * pin.GRID
        c = ch[(ch.kind == "C") & (ch.K == k)]
        p = ch[(ch.kind == "P") & (ch.K == k)]
        if c.empty or p.empty:
            continue
        cb, ca = float(c.bid.iloc[0]), float(c.ask.iloc[0])
        pb, pa = float(p.bid.iloc[0]), float(p.ask.iloc[0])
        mid = (cb + ca) / 2 + (pb + pa) / 2
        recv = cb + pb                                  # what a crossing seller actually collects
        cross = mid - recv
        close = float(yf.Ticker(n).history(period="5d", interval="1d").Close.iloc[-1])
        settle = abs(close - k)                          # straddle intrinsic at expiry
        rows.append({"name": n, "S_1210": s0, "K": k, "mid": mid, "recv": recv, "cross": cross,
                     "cross_pct": 100 * cross / mid, "close": close, "settle": settle,
                     "pnl_mid": mid - settle, "pnl_real": recv - settle,
                     "pnl_real_bp": 1e4 * (recv - settle) / s0})
    if not rows:
        print("  no matching strikes in the snapshot")
        return
    t = pd.DataFrame(rows)
    print(f"  {'name':6} {'S@12:10':>9} {'K':>8} {'mid':>7} {'you get':>8} {'cross':>7} {'cross%':>7} "
          f"{'close':>9} {'|close-K|':>10} {'P&L@mid':>8} {'P&L real':>9}")
    for _, r in t.iterrows():
        print(f"  {r['name']:6} {r.S_1210:9.2f} {r.K:8.1f} {r['mid']:7.2f} {r.recv:8.2f} {r.cross:7.2f} "
              f"{r.cross_pct:6.1f}% {r.close:9.2f} {r.settle:10.2f} {r.pnl_mid:+8.2f} {r.pnl_real:+9.2f}")
    print(f"\n  mean spread cost as a share of the maximum theoretical decay: {t.cross_pct.mean():.0f}%")
    print(f"  straddles profitable at the MID: {(t.pnl_mid>0).sum()} of {len(t)}   "
          f"after crossing: {(t.pnl_real>0).sum()} of {len(t)}")
    print(f"  mean P&L per straddle at mid {t.pnl_mid.mean():+.2f}, after crossing {t.pnl_real.mean():+.2f}")
    print("\n  QUESTION 3 IS ANSWERED NO, AND IT IS THE ONE PLACE THE BRIEF'S PESSIMISM WAS MISPLACED.")
    print(f"  Crossing both legs costs {t.cross_pct.mean():.0f}% of the straddle mid, not 100%. These are the")
    print("  five most liquid option books in the US market, quoted a penny or two wide on 0DTE; 'wide")
    print("  Friday retail spreads' is a real problem in thin single names and simply is not one here.")
    print("  The cost structure is a fact about the quote and needs no sample. What DOES need a sample is")
    print("  whether the straddle wins, and five names on one quiet afternoon is not one: all five share")
    print("  the same market move, so this is closer to n=1. The +2.05 mean is short gamma being paid on a")
    print("  calm day, which is a volatility risk premium available every day of the week, not a pin.")
    print("  Panel 5 is the actual historical test of whether Friday settlement is tighter.")

    print("\n  POST-CLOSE CBOE CHECK on the same names, for spread width with the exchange's own quotes:")
    for n in list(snap)[:5]:
        try:
            df, spot, _ = cboe.chain(n)
        except Exception:
            continue
        exps = sorted(df.expiry.unique())
        nxt = pd.Timestamp(exps[1]) if len(exps) > 1 else pd.Timestamp(exps[0])
        k = min(df[df.expiry == nxt].strike.unique(), key=lambda x: abs(x - spot))
        r = cboe.atm_cost(df, spot, nxt, float(k))
        if r:
            print(f"    {n:6} next expiry {nxt.date()}  K {r['strike']:8.1f}  straddle mid "
                  f"{r['straddle_mid']:7.2f}  one-way cross {r['one_way_cross']:5.2f} "
                  f"({r['cross_pct_of_mid']:4.1f}% of mid)  sizes C{r['call'][2]:.0f}x{r['call'][3]:.0f} "
                  f"P{r['put'][2]:.0f}x{r['put'][3]:.0f}")


def verdict() -> None:
    print("\n" + "=" * 108)
    print("VERDICT")
    print("=" * 108)
    print("""
  QUESTION 1 — DOES SPOT FINISH NEARER K* MORE THAN 50% OF THE TIME? NO, AND THE 50% BAR WAS UNREACHABLE.
  On expiration Fridays the close is nearer the strike than the 10:30 price was on 32% of name-days. On
  control Mon-Thu days it is 33%. Mean distance INCREASES by 53 bp on Fridays and 60 bp on control days.
  The reason 32% is not evidence of anything is that a driftless random walk cannot produce 50%: starting
  at distance d, the close is nearer only if the move lands in the interval (-2d, 0), and the expected
  absolute distance strictly increases by Jensen. Measured against the only correct null -- an ordinary
  weekday -- expiration Friday converges 6.5 bp more, t = 1.12. That is nothing. Had this been scored
  against the brief's 50%, every day of the week would have "failed", which would have told you about the
  absolute-value function rather than about dealers.

  QUESTION 2 — DOES THE DELTA TRADE SURVIVE A 1-2 bp FEE? THE FEE IS IRRELEVANT, AND THE TRADE STILL FAILS.
  Gross, collapsing the five names into one observation per date, Fridays earn +14.4 bp (t = 2.32) against
  -1.4 bp on control days, a difference of +15.8 bp at t = 2.10. Net of 2 bp the Friday book is +7.5 bp at
  t = 1.08. So the answer to the question as asked is yes, it survives the fee -- the fee is 1.4% of one
  standard deviation and was never the obstacle. The trade fails four validity checks instead, and any
  one of them is disqualifying:

    THE DOSE-RESPONSE GOES BACKWARDS. Third Fridays carry five to ten times the open interest of a weekly
    and are where dealer gamma is overwhelmingly concentrated. They earn +2.9 bp (t = 0.26) against
    +17.7 bp for weeklies; monthly minus weekly is -14.9 bp. An effect caused by gamma must scale with
    gamma. This one is absent exactly where gamma is largest.

    THE PLACEBO REPRODUCES IT EXACTLY, WHICH IS THE SINGLE MOST DAMNING NUMBER HERE. Shift the grid by
    $0.65 so that it lands on no listed strike in any of these five names -- no open interest, no dealer
    position, nothing to hedge -- and the same trade earns +13.9 bp at t = 2.08 against the real grid's
    +14.4 bp at t = 2.32. Friday minus control is actually LARGER on the fake grid (+19.5, t = 2.58). An
    effect that does not care whether options are listed at the price cannot be produced by option
    dealers. A $1.30 offset gives +6.4 bp, so the variation between arbitrary grids exceeds the distance
    from any of them to zero. (The $5 + $2.50 grid is not a placebo at all and is reported only as a
    consistency check: its points are the exact midpoints of the $5 strikes, so the long/short call flips
    on 100% of days and its -20.5 bp is arithmetically forced to mirror the real grid.)

    FRIDAY DOES NOT STAND OUT AMONG WEEKDAYS. Running the identical rule day by day: Mon +4.0 (t 0.56),
    Tue -17.6 (t -2.04), Wed +3.6 (t 0.46), Thu +5.3 (t 0.52), Fri +14.4 (t 2.32). Tuesday is as
    significant as Friday and points the opposite way. Five weekdays times four grid offsets is twenty
    tests of a signal with a mechanism behind it in only one cell, and one |t| > 2 in each direction is
    the expected harvest of that search rather than a discovery.

    IT IS FIVE DAYS AND IT IS DECAYING. The best 5 of 146 Fridays are 50% of the total. First half
    +19.6 bp (t = 2.05), second half +9.0 bp (t = 1.14) -- on a sample only three years long, because
    yfinance serves no hourly bars before 2023-10-27.

  QUESTION 3 — DO RETAIL OPTION SPREADS CONSUME 100% OF THE DECAY? NO. THIS IS THE ONE PLACE THE BRIEF WAS
  TOO PESSIMISTIC. Measured on real 0DTE quotes captured at 12:10 ET today with the settlement now known,
  crossing both legs of the ATM straddle costs 2% of the straddle mid, ranging from 0.4% on SPY to 3.2% on
  AMD. A penny or two wide on a $2-7 straddle. The premise that Friday spreads are prohibitive is true of
  thin single-name books and false of the five most liquid option books in the United States. If a pin
  existed in these names, execution cost would not be what stopped you from harvesting it.

  WHAT IS ACTUALLY THERE, STATED FAIRLY, BECAUSE IT IS NOT NOTHING. Variant 2 has the better claim and the
  brief had it second. Realised settlement distance |S_close - K| is 113.5 bp on Fridays against 123.1 bp
  on control days (t = -1.37), and -- unlike the delta trade -- the monthlies are the TIGHTEST of all at
  93.8 bp, which is the direction a gamma story requires. So the distribution around the strike does look
  mildly compressed on high-OI expirations. That is the dispersion effect, and it is the effect dealer
  hedging actually predicts. It is not significant at this sample size, and it is a short-volatility
  premium rather than a pin: you would be paid for selling gamma into a quiet afternoon, which is
  available every day of the week and carries the tail risk that goes with it.

  THE CONCEPTUAL ERROR THE BRIEF INHERITED FROM THE LITERATURE. Buying dips and selling rips compresses
  the DISTRIBUTION of the close around K*. It does not create DRIFT toward K*. A perfectly pinned stock
  that is equally likely to approach its strike from either side has a tighter variance and an expected
  return of exactly zero. Variant 1 is a drift bet and variant 2 is a dispersion bet, and only the second
  is what the mechanism describes. The tradeable expression of gamma pinning is short gamma, not long
  delta -- and short gamma is a crowded, well-priced trade rather than a structural inefficiency.

  DATA LIMITS THAT ARE REAL AND WHICH WAY THEY CUT. There is no free historical per-strike open interest
  from any source -- yfinance serves unexpired chains only, DoltHub has neither an OI column nor the
  expiring front week, CBOE's CDN is a current snapshot, and Kaggle, Polygon, ThetaData, Tradier and IBKR
  all require an account this machine does not have. So K* itself cannot be reconstructed historically,
  and the test above uses the $5 grid as a superset: every candidate K* is a point on it. Section B shows
  the substitution is loose -- K* was the nearest strike in only 3 of 5 names live, and the top strike
  holds as little as 17% of the gamma inside +-1.5% on SPY, meaning there is often no dominant magnet to
  be pulled toward. A superset test can only falsify, and this is a falsification: if spot will not
  converge on the grid that CONTAINS every K*, no rule for choosing among those grid points helps. The
  reference price is also 10:30 rather than the specified 10:00, because a 60-minute bar opening at 09:30
  closes at 10:30 and 30-minute bars reach back only 60 days. That error runs against the strategy -- the
  pin has had an extra half hour to act before the starting distance is measured -- so it cannot be
  manufacturing the negative result.

  BOTTOM LINE. Not tradeable. The delta trade's apparent Friday edge survives being shifted onto a grid
  with no options on it, is matched in size by an equally significant Tuesday effect of the opposite sign,
  is absent on the monthly expirations where dealer gamma is concentrated, and lives in five of 146 dates.
  The honest version of the mechanism is short gamma, whose premium is already priced and whose tail is
  the reason it is priced. Costs are not the binding constraint in either variant -- the 2 bp equity fee is
  1.4% of a standard deviation and the real 0DTE straddle spread is 2% of mid. That is the same conclusion
  the index reconstitution study reached: what fails is the signal, and cheap execution cannot rescue a
  signal that is not there.""")


def main() -> int:
    print("=" * 108)
    print("OPEX GAMMA PINNING / DEALER MAGNET — is the pin a tradeable drift or a compressed distribution?")
    print("=" * 108)
    section_a()
    section_b()
    section_c()
    verdict()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
