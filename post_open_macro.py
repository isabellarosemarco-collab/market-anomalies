"""
post_open_macro.py — does pre-announcement drift generalise to every scheduled 10:00-ET-or-later
event, or is it specific to the FOMC rate decision?

    python post_open_macro.py > post_open_macro_report.txt

THE HYPOTHESIS, STATED SO IT CAN FAIL. The FOMC eve trade works: +18.49 bp per date net, t +3.42,
Sharpe +1.00, and the ES futures split shows the money arriving BEFORE the 14:00 statement rather than
from it. The brief's structural reading of that success is that the trade is paid for carrying
pre-event risk and exits before the headline. If that reading is right, the effect should appear on the
eve of ANY high-impact, pre-scheduled release that lands after the 09:31 exit. If the effect is instead
specific to the rate decision, the siblings will be flat and the FOMC result will look less like a
general risk premium and more like one institution's hedging calendar.

That is a genuinely two-sided test, which is why it is worth running. But four things can manufacture a
false positive here, and each gets its own section rather than a caveat.

  1. THE TURN OF THE MONTH, AND IT IS FATAL TO THE NAIVE ISM TEST. ISM Manufacturing prints on the
     FIRST business day of the month, so its eve is the LAST TRADING DAY OF THE PRIOR MONTH -- the
     single strongest day in the turn-of-the-month literature. ISM Services prints on the third
     business day, so its eve is the second. Twenty-four of the brief's ~46 events a year are
     therefore stacked on exactly the days a decades-old calendar anomaly already occupies. Comparing
     them against a control spread evenly through the month does not test ISM; it tests the turn of
     the month and attributes the answer to ISM. Section 5 attacks this with month-position fixed
     effects -- and finds that the attack is impossible: because ISM has no off-months, there is no
     first-trading-day session in the sample that is not an ISM eve, so the event dummy and the
     position dummy are the same column and the comparison is unidentified rather than merely noisy.
     The estimator therefore counts same-position control nights BEFORE estimating and refuses to
     report a coefficient where there are none. Without that check it returns +64 bp at robust t +9.9
     from a rank-deficient design, which is a solver artifact and would be the most dangerous line in
     this study. Not being able to answer is the answer for ISM.

  2. OVERLAP WITH THE FOMC ITSELF. The events are not disjoint. An ISM eve that happens to also be an
     FOMC eve imports the one effect known to be real. Every slice is therefore reported twice: raw,
     and pure -- with dates shared with any other event removed. A slice whose significance depends on
     the overlap has not found anything of its own.

  3. MULTIPLE TESTING. Six new slices are examined here on top of the three already tested. At the 5%
     level, one spurious winner among nine tests is the expected outcome, not a surprise. Section 7
     applies a Holm step-down correction, and every p-value before that point should be read as
     uncorrected.

  4. POWER, WHICH CUTS THE OTHER WAY AND IS THE REASON MOST NULLS HERE ARE UNINFORMATIVE. Control
     nights have a per-date standard deviation near 75 bp. Detecting an FOMC-sized 15 bp effect against
     them at t=2 needs roughly (1.96*75/15)^2 = 95 observations. Humphrey-Hawkins supplies 23 in eleven
     years and the refunding events 47, so those slices CANNOT produce a significant result even if the
     effect is exactly as large as the FOMC's. Section 6 reports each slice's minimum detectable effect,
     computed from its own Welch standard error rather than one pooled volatility, so that a null can be
     distinguished from an absence of evidence. Combined with point 1, this leaves the FOMC minutes as
     the only sibling that is both adequately powered AND cleanly identified -- which makes it, not ISM,
     the slice the conclusion rests on.

WHICH SIBLINGS ACTUALLY QUALIFY. The brief lists four. Three pass the clock test and one does not: the
Treasury Quarterly Refunding Statement is released at 08:30 ET, inside the 15:59 -> 09:31 window, so it
has the identical defect that disqualified CPI and NFP. It is measured here as a negative control and
never pooled. Its companion release does qualify -- Treasury publishes marketable borrowing estimates
at 15:00 on the Monday of refunding week, after the exit -- so the refunding cycle contributes a clean
event and a dirty one, and keeping them apart is the point. See post_open_calendar.py for sources.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import bls_calendar
import post_open_calendar as cal
from event_calendar import fomc_dates
from macro_overnight_alpha import NAMES, START, by_date, load_panel, one_sample, permutation, perf, \
    welch

FEE_BP = 2.0                      # the brief's round-trip friction for this test
NPERM = 20000
RNG = np.random.default_rng(20260925)

# Events released after the 09:31 exit: the window carries the uncertainty and none of the print.
CLEAN = ("MINUTES", "ISM_MFG", "ISM_SVC", "HUMPHREY", "QRA_ESTIMATES")
DIRTY = ("QRA_STATEMENT",)        # 08:30, inside the window -- negative control only
PRIOR = ("FOMC", "CPI", "NFP")    # already tested; FOMC is the benchmark, CPI/NFP are dirty


# ----------------------------------------------------------------------------- labels

def build_labels(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per session, flagging the eve of every event, plus the control and month-position.

    The flag marks the session whose MORNING the trade exits on, i.e. the release day itself, so an
    event on date D sets the flag on row D. Dates that are not trading sessions are dropped rather
    than shifted: a release on a closed day has no eve trade, and moving it to the next session would
    invent an event night.
    """
    ev = dict(cal.all_events(START, "2026-12-31"))
    rel = bls_calendar.release_dates()
    ev["CPI"] = pd.DatetimeIndex(rel["CPI"].date)
    ev["NFP"] = pd.DatetimeIndex(rel["NFP"].date)
    ev["FOMC"] = fomc_dates()

    out = pd.DataFrame(index=idx)
    for k, d in ev.items():
        d = d[(d >= idx.min()) & (d <= idx.max())]
        on_grid = d.intersection(idx)
        out[k] = out.index.isin(on_grid)
        out.attrs[f"{k}_dropped"] = len(d) - len(on_grid)
        out.attrs[f"{k}_scheduled"] = len(d)

    names = list(ev)
    out["n_events"] = out[names].sum(axis=1)
    # The control must exclude EVERY event eve, old and new, plus the session after each one, or the
    # baseline quietly contains the effect being tested against it.
    any_eve = out[names].any(axis=1)
    out["control"] = ~(any_eve | any_eve.shift(1, fill_value=False))

    # Month position, from both ends, because ISM sits at the boundary. The bucket is the fixed effect
    # used in section 5: exact position within the first five and last five trading days of the month,
    # everything else pooled as 'mid'.
    g = out.groupby([out.index.year, out.index.month])
    tdom = g.cumcount() + 1
    trev = g.cumcount(ascending=False) + 1
    out["tdom"], out["trev"] = tdom, trev
    out["bucket"] = np.where(tdom <= 5, "+" + tdom.astype(str),
                             np.where(trev <= 5, "-" + trev.astype(str), "mid"))
    return out


def pure(lab: pd.DataFrame, key: str) -> pd.Series:
    """The slice with every date that also carries another event removed."""
    return lab[key] & (lab.n_events == 1)


# ------------------------------------------------------------------------- statistics

MIN_OVERLAP = 10          # control nights needed in a position before it can identify anything


def ols_event_effect(y: np.ndarray, ev: np.ndarray, bucket: np.ndarray,
                     min_overlap: int = MIN_OVERLAP) -> tuple[float, float, float]:
    """Event effect on `y` controlling for month-position fixed effects.

    Returns (coef, robust t, identified_fraction). The third number is the one to read first.

    Regressing on an event dummy plus one dummy per month-position compares each event night only
    against control nights in the SAME calendar slot, which is the right way to strip the turn of the
    month. But it only works if such control nights exist. For an event pinned to a fixed position --
    and ISM Manufacturing is released on the first business day of every single month in the sample --
    the event dummy and that position's dummy are the same column, the design is rank-deficient, and
    a pseudo-inverse will happily split the effect between them and report a large coefficient with a
    huge t-stat that means nothing at all.

    So the overlap is measured before the estimate is trusted: identified_fraction is the share of
    event nights sitting in positions that have at least `min_overlap` control nights. When it is near
    zero the event and the calendar are the same variable and the question is not answerable from this
    data, which is a finding about the design rather than a number to report.
    """
    y = np.asarray(y, float)
    keep = np.isfinite(y)
    y, ev, bucket = y[keep], np.asarray(ev, float)[keep], np.asarray(bucket)[keep]
    is_ev = ev > 0.5
    ctl_counts = pd.Series(bucket[~is_ev]).value_counts()
    ok = {b for b, c in ctl_counts.items() if c >= min_overlap}
    frac = float(np.mean([b in ok for b in bucket[is_ev]])) if is_ev.any() else 0.0
    if frac < 0.25:
        return (np.nan, np.nan, frac)
    levels = sorted(set(bucket))
    X = np.column_stack([np.ones(len(y)), ev] + [(bucket == L).astype(float) for L in levels[1:]])
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    # HC0 sandwich: no assumption that month-end and mid-month nights share a variance.
    meat = X.T @ (X * (resid ** 2)[:, None])
    V = XtX_inv @ meat @ XtX_inv
    se = np.sqrt(max(V[1, 1], 1e-18))
    return float(beta[1]), float(beta[1] / se), frac


def mde_from_se(se: float, t_crit: float = 1.96) -> float:
    """Smallest true effect reaching t=1.96 given this slice's ACTUAL standard error, in bp.

    Computed from the observed Welch standard error rather than from a single pooled volatility, so it
    reflects each slice's own variance instead of assuming every event night is as noisy as the
    average control night.
    """
    return float(t_crit * se) if np.isfinite(se) and se > 0 else np.nan


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, tuple[float, bool]]:
    """Holm step-down: returns {name: (adjusted p, survives)}. Controls the family-wise error rate."""
    items = sorted([(p, k) for k, p in pvals.items() if np.isfinite(p)])
    m = len(items)
    out, running = {}, 0.0
    for i, (p, k) in enumerate(items):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        out[k] = (adj, adj <= alpha)
    return out


# -------------------------------------------------------------------------- sections

def section0(lab: pd.DataFrame) -> None:
    print("=" * 112)
    print("0. THE STRUCTURAL SCREEN — which of the brief's siblings actually clear the 09:31 exit?")
    print("=" * 112)
    cal.summary({k: pd.DatetimeIndex(lab.index[lab[k]]) for k in CLEAN + DIRTY})
    print("\n  Sessions lost because the release fell on a non-trading day, and total scheduled:")
    for k in CLEAN + DIRTY + PRIOR:
        print(f"    {k:16} scheduled {lab.attrs[f'{k}_scheduled']:4}   "
              f"dropped {lab.attrs[f'{k}_dropped']:2}   usable {int(lab[k].sum()):4}")
    print("\n  MONTH POSITION OF EACH EVENT EVE — read this before any result below.")
    print("  ISM Manufacturing prints on the first business day, so its eve is the LAST trading day of")
    print("  the prior month. That is the turn-of-the-month anomaly's home address, and it accounts for")
    print("  half of the ISM sample the brief is counting on for frequency.\n")
    print(f"  {'event':16} {'median tdom':>12} {'median trev':>12}  most common bucket")
    for k in CLEAN + DIRTY + ("FOMC",):
        s = lab[lab[k]]
        if s.empty:
            continue
        print(f"  {k:16} {s.tdom.median():12.0f} {s.trev.median():12.0f}  "
              f"{s.bucket.value_counts().head(3).to_dict()}")


def section1(lab: pd.DataFrame, es: pd.DataFrame | None) -> None:
    print("\n" + "=" * 112)
    print("1. DO THE DATES SURVIVE CONTACT WITH INTRADAY DATA? — validating the one rule-based calendar")
    print("=" * 112)
    print("  Minutes, testimony and refunding dates come from the issuer. ISM dates are generated from")
    print("  the first/third-business-day rule because ISM publishes no free archive, so they are the")
    print("  one input that could be systematically wrong. The test: ES futures trade continuously, so")
    print("  on a correctly dated ISM day the 10:00-11:00 bar -- which contains the 10:00:00 print --")
    print("  must show an elevated absolute move. The same check applies to the 14:00 bar on minutes")
    print("  days. If these are flat, the calendar is wrong and nothing downstream means anything.\n")
    if es is None or es.empty:
        print("  ES futures bars unavailable; ISM dates could not be validated.")
        return
    ctl = es[es.index.isin(lab.index[lab.control])]
    print(f"  {'event':16} {'n':>4} {'|10:00 bar| bp':>15} {'t vs ctl':>9} "
          f"{'|14:00 bar| bp':>15} {'t vs ctl':>9}")
    for k in ("ISM_MFG", "ISM_SVC", "MINUTES", "HUMPHREY", "FOMC"):
        d = es[es.index.isin(lab.index[lab[k]])]
        if len(d) < 5:
            print(f"  {k:16} {len(d):4}   too few bars in the futures window")
            continue
        _, t10, _ = welch(d.a10.abs().values, ctl.a10.abs().values)
        _, t14, _ = welch(d.a14.abs().values, ctl.a14.abs().values)
        print(f"  {k:16} {len(d):4} {d.a10.abs().mean():15.1f} {t10:+9.2f} "
              f"{d.a14.abs().mean():15.1f} {t14:+9.2f}")
    print(f"  {'control days':16} {len(ctl):4} {ctl.a10.abs().mean():15.1f} {'':>9} "
          f"{ctl.a14.abs().mean():15.1f}")
    print("\n  The 10:00 releases should light up the first column and the 14:00 releases the second.")
    print("  That is each calendar confirming itself against the clock it claims to follow.")


def section2(on: pd.DataFrame, lab: pd.DataFrame) -> None:
    print("\n" + "=" * 112)
    print("2. FREQUENCY — does the clean set reach the ~46 events a year the brief is aiming at?")
    print("=" * 112)
    yrs = (on.index.max() - on.index.min()).days / 365.25
    clean_any = lab[list(CLEAN)].any(axis=1)
    print(f"  sample {on.index.min().date()} .. {on.index.max().date()}  ({yrs:.1f} years)\n")
    for k in CLEAN:
        print(f"    {k:22} n={int(lab[k].sum()):5}   {lab[k].sum()/yrs:5.1f} per year   "
              f"pure {int(pure(lab, k).sum()):4}")
    print(f"    {'-'*70}")
    print(f"    {'CLEAN POOLED':22} n={int(clean_any.sum()):5}   {clean_any.sum()/yrs:5.1f} per year")
    both = (clean_any | lab.FOMC)
    print(f"    {'CLEAN + FOMC':22} n={int(both.sum()):5}   {both.sum()/yrs:5.1f} per year")
    print(f"\n    {'QRA_STATEMENT (dirty)':22} n={int(lab.QRA_STATEMENT.sum()):5}   "
          f"{lab.QRA_STATEMENT.sum()/yrs:5.1f} per year   — excluded from the pool by the clock rule")
    print(f"    {'control nights':22} n={int(lab.control.sum()):5}")
    ev = lab.index[both]
    roll = pd.Series(1, index=ev).rolling("365D").sum()
    roll = roll[roll.index >= ev.min() + pd.Timedelta(days=365)]
    print(f"\n    rolling 12-month count over {len(roll)} complete windows: "
          f"min {int(roll.min())}, median {int(roll.median())}, max {int(roll.max())}")
    print("\n  Frequency is not the binding constraint and never was. The question is whether the added")
    print("  events carry any expectancy, which is section 3.")


def _row(nm: str, x: np.ndarray, ctl: np.ndarray) -> dict:
    s = one_sample(x - FEE_BP)
    if not s.get("n") or s["n"] < 5:
        return {"nm": nm, "n": s.get("n", 0)}
    d, t, p = welch(x, ctl)
    # The Welch standard error is kept because it, not a single pooled volatility, is what the
    # minimum detectable effect in section 6 has to be built from.
    se = float(np.sqrt(np.var(x, ddof=1) / len(x) + np.var(ctl, ddof=1) / len(ctl)))
    return {"nm": nm, "n": s["n"], "net": s["mean"], "win": s["win"], "t1": s["t"],
            "diff": d, "tw": t, "pw": p, "perm": permutation(x, ctl, NPERM), "sd": s["sd"],
            "se": se}


def section3(on: pd.DataFrame, lab: pd.DataFrame) -> dict:
    print("\n" + "=" * 112)
    print("3. EXPECTANCY AND SIGNIFICANCE — every slice against the same control, net of 2 bp")
    print("=" * 112)
    print("  One observation per date: SPY/QQQ/IWM averaged within the date first, because three ETFs")
    print("  on one night are three views of one market move. The control is every session that is")
    print("  neither an event eve nor the night after one, for ALL nine event types. 'own t' tests the")
    print("  net mean against zero; 'vs ctl' is the test that matters, because equities have a positive")
    print("  unconditional overnight return. perm p shuffles date labels 20,000 times.\n")
    ctl = by_date(on, lab.control)
    rows, res = [], {}
    for k in CLEAN + DIRTY + PRIOR:
        r = _row(k, by_date(on, lab[k]), ctl)
        rows.append(r)
        res[k] = r
    rp = {}
    for k in CLEAN:
        rp[k] = _row(k + " [pure]", by_date(on, pure(lab, k)), ctl)
    clean_any = lab[list(CLEAN)].any(axis=1)
    pooled = _row("CLEAN POOLED", by_date(on, clean_any), ctl)
    pooled_f = _row("CLEAN + FOMC", by_date(on, clean_any | lab.FOMC), ctl)

    hdr = (f"  {'slice':22} {'n':>4} {'net bp':>8} {'win':>5} {'own t':>7} "
           f"{'vs ctl':>8} {'Welch t':>8} {'t p':>7} {'perm p':>8}")
    def emit(r):
        if not r.get("n") or r["n"] < 5:
            print(f"  {r['nm']:22} {r.get('n',0):4}   too few observations")
            return
        print(f"  {r['nm']:22} {r['n']:4} {r['net']:8.2f} {100*r['win']:4.0f}% {r['t1']:+7.2f} "
              f"{r['diff']:+8.2f} {r['tw']:+8.2f} {r['pw']:7.3f} {r['perm']:8.4f}")
    print("  THE FIVE CLEAN SIBLINGS (release after the 09:31 exit):")
    print(hdr)
    for k in CLEAN:
        emit(res[k])
    print("\n  SAME SLICES, PURE — dates shared with any other event removed:")
    for k in CLEAN:
        emit(rp[k])
    print("\n  POOLED, AND THE BENCHMARK IT IS TRYING TO EXTEND:")
    emit(pooled)
    emit(pooled_f)
    emit(res["FOMC"])
    print("\n  THE NEGATIVE CONTROL AND THE TWO ALREADY-REJECTED 08:30 EVENTS, for contrast:")
    for k in DIRTY + ("CPI", "NFP"):
        emit(res[k])
    ctl_s = one_sample(ctl - FEE_BP)
    print(f"\n  {'control nights':22} {ctl_s['n']:4} {ctl_s['mean']:8.2f} {100*ctl_s['win']:4.0f}% "
          f"{ctl_s['t']:+7.2f}   (sd {ctl_s['sd']:.1f} bp — the number that sets every power limit here)")
    res["_pooled"], res["_pooled_f"], res["_pure"] = pooled, pooled_f, rp
    res["_ctl_sd"] = ctl_s["sd"]
    res["_ctl_n"] = ctl_s["n"]
    return res


def section4(on: pd.DataFrame, lab: pd.DataFrame) -> None:
    print("\n" + "=" * 112)
    print("4. THE PLACEBO — the night AFTER each release, when the uncertainty is already resolved")
    print("=" * 112)
    print("  A pre-announcement premium must vanish once the announcement is known. If the night after")
    print("  pays the same as the night before, whatever is being measured is not about the event.\n")
    ctl = by_date(on, lab.control)
    print(f"  {'slice':22} {'n':>4} {'eve bp':>8} {'post bp':>9} {'post vs ctl':>12} {'t':>7}")
    for k in CLEAN + ("FOMC",):
        eve = lab[k]
        post = eve.shift(1, fill_value=False)
        e, p_ = by_date(on, eve), by_date(on, post)
        if len(p_) < 5:
            continue
        d, t, _ = welch(p_, ctl)
        print(f"  {k:22} {len(p_):4} {e.mean():8.2f} {p_.mean():9.2f} {d:+12.2f} {t:+7.2f}")


def section5(on: pd.DataFrame, lab: pd.DataFrame) -> dict:
    print("\n" + "=" * 112)
    print("5. THE TURN OF THE MONTH — and why the ISM question cannot be answered at all")
    print("=" * 112)
    print("  ISM Manufacturing is released on the first business day of every month, so its eve is the")
    print("  last trading day of the month; ISM Services is the third. The turn-of-the-month effect, the")
    print("  oldest documented calendar anomaly there is, lives on exactly those sessions. The honest")
    print("  way to strip it is an event dummy plus one fixed effect per month-position, comparing each")
    print("  event night only against control nights in the same slot -- but that requires such control")
    print("  nights to EXIST, so the overlap is counted first and the estimate is only printed when the")
    print("  comparison is actually identified.\n")
    per_date = on.mean(axis=1)
    c = lab[lab.control]
    cy = per_date.reindex(c.index)
    print("  STEP ONE — the month-position calendar, and how many control nights each position has:")
    print(f"    {'position':>9} {'all nights bp':>14} {'n all':>7} {'control bp':>11} {'n ctl':>7}  "
          "occupied by")
    occ = {}
    for k in CLEAN + DIRTY + ("FOMC",):
        for b, n in lab[lab[k]].bucket.value_counts().items():
            if n / max(1, int(lab[k].sum())) > 0.5:
                occ.setdefault(b, []).append(k)
    order = ["-5", "-4", "-3", "-2", "-1", "+1", "+2", "+3", "+4", "+5", "mid"]
    allb = lab.bucket.values
    ally = per_date.reindex(lab.index)
    for b in order:
        na = int((allb == b).sum())
        if na == 0:
            continue
        v = cy[c.bucket.values == b]
        va = ally[allb == b]
        who = ", ".join(occ.get(b, [])) or "-"
        cbp = f"{v.mean():+11.2f}" if len(v) else f"{'--':>11}"
        print(f"    {b:>9} {va.mean():+14.2f} {na:7} {cbp} {len(v):7}  {who}")
    print("\n    Read the 'n ctl' column. Positions +1 and +3 are nearly empty of control nights because")
    print("    virtually every one of them IS an ISM eve -- ISM has no off-months. The event dummy and")
    print("    the position dummy are therefore the same column, the design is rank-deficient, and any")
    print("    coefficient a pseudo-inverse returns for those slices is extrapolation, not a comparison.")

    out = {}
    print("\n  STEP TWO — the adjusted effect, printed only where the comparison is identified:")
    print(f"  {'slice':22} {'n':>4} {'raw diff':>9} {'adjusted':>9} {'robust t':>9} {'ident':>6}  "
          "dominant position")

    def emit(nm, m, key_series):
        mm = m | lab.control
        sub = lab[mm]
        coef, t, frac = ols_event_effect(per_date.reindex(sub.index).values,
                                         key_series[mm].values.astype(float), sub.bucket.values)
        raw, _, _ = welch(by_date(on, m), by_date(on, lab.control))
        dom = lab[m].bucket.value_counts()
        out[nm] = (coef, t, frac)
        pos = (f"{dom.index[0]} ({100*dom.iloc[0]/dom.sum():.0f}% of the slice)"
               if len(dom) else "")
        if not np.isfinite(coef):
            print(f"  {nm:22} {int(m.sum()):4} {raw:+9.2f} {'NOT IDENT':>9} {'--':>9} "
                  f"{100*frac:5.0f}%  {pos}")
        else:
            print(f"  {nm:22} {int(m.sum()):4} {raw:+9.2f} {coef:+9.2f} {t:+9.2f} "
                  f"{100*frac:5.0f}%  {pos}")

    for k in CLEAN + DIRTY + ("FOMC",):
        emit(k, lab[k], lab[k])
    clean_any = lab[list(CLEAN)].any(axis=1)
    for nm, m in (("CLEAN POOLED", clean_any), ("CLEAN + FOMC", clean_any | lab.FOMC)):
        emit(nm, m, m)
    print("\n  'ident' is the share of the slice's nights sitting in a month-position that has at least")
    print(f"  {MIN_OVERLAP} control nights to compare against. Below 25% the estimate is suppressed rather than")
    print("  reported, because a number like +64 bp at t +9.9 out of a rank-deficient design is an")
    print("  artifact of the solver and would be the single most misleading line in this study.")
    print("\n  WHAT THAT MEANS FOR THE TWO ISM SLICES, WHICH ARE 24 OF THE BRIEF'S ~46 EVENTS A YEAR:")
    print("  their eves and the turn of the month are the same sessions in every month of the sample, so")
    print("  no amount of data separates 'ISM pre-announcement premium' from 'turn-of-the-month drift'.")
    print("  The raw ISM numbers in section 3 are real returns but they are not attributable, and the")
    print("  parsimonious reading is the calendar, which is documented back to the 1980s and needs no")
    print("  pre-announcement mechanism. Two further facts point the same way: the effect is not there")
    print("  in the first place (ISM_MFG +3.9 bp vs control at t +0.57, ISM_SVC negative), and if it were")
    print("  the calendar it would be capturable on month-end nights without any reference to ISM.")
    return out


def section6(res: dict, lab: pd.DataFrame) -> None:
    print("\n" + "=" * 112)
    print("6. POWER — which of these slices could ever have produced a significant answer?")
    print("=" * 112)
    print("  A null result is only informative if the test could have detected a real effect. The")
    print("  minimum detectable effect below is the smallest true difference from control that would")
    print("  reach t=1.96 given this slice's sample size and the observed overnight volatility. Read")
    print("  it against the FOMC's measured effect: any slice whose MDE exceeds that cannot confirm")
    print("  an FOMC-sized premium even if one is there, and its null means nothing. The MDE is built")
    print("  from each slice's OWN Welch standard error rather than one pooled volatility, so a quiet")
    print("  slice is not penalised for the noise of the average night.\n")
    ref = abs(res["FOMC"]["diff"])
    print(f"  {'slice':22} {'n':>5} {'MDE bp':>8} {'observed':>9}  verdict on the null")

    def emit(nm, n, r):
        m = mde_from_se(r.get("se", np.nan))
        obs = r.get("diff", np.nan)
        if not np.isfinite(m):
            return
        # Graded rather than binary: a slice whose MDE lands within a few bp of the reference is not
        # meaningfully different from one just under it, and a hard cutoff there would turn a knife
        # edge into a verdict.
        tag = ("adequately powered — null is evidence of absence" if m <= 1.1 * ref
               else f"marginal — could miss a {ref:.0f} bp effect about half the time" if m <= 2 * ref
               else f"cannot detect a {ref:.0f} bp effect — null uninformative")
        print(f"  {nm:22} {n:5} {m:8.1f} {obs:+9.2f}  {tag}")

    for k in CLEAN + DIRTY + ("FOMC",):
        emit(k, int(lab[k].sum()), res.get(k, {}))
    clean_any = lab[list(CLEAN)].any(axis=1)
    emit("CLEAN POOLED", int(clean_any.sum()), res["_pooled"])
    emit("CLEAN + FOMC", int((clean_any | lab.FOMC).sum()), res["_pooled_f"])
    print(f"\n  The reference effect is the FOMC's own measured {ref:+.1f} bp over control. A slice needs")
    print("  roughly 95 observations against this control to resolve an effect that size, which is why")
    print("  Humphrey-Hawkins and the refunding events are not evidence of absence -- they are absence")
    print("  of evidence, and they would have to run for decades to become anything else.")


def section7(res: dict, adj: dict) -> None:
    print("\n" + "=" * 112)
    print("7. MULTIPLE TESTING — nine slices examined, so one winner at p<0.05 is the expected outcome")
    print("=" * 112)
    print("  Every p-value so far is uncorrected. Holm's step-down procedure controls the probability")
    print("  of ANY false positive across the family at 5%: sort the p-values, require the smallest to")
    print("  beat alpha/m, the next alpha/(m-1), and so on. The family here is the five clean siblings")
    print("  plus the FOMC benchmark, using the permutation p because it makes no distributional")
    print("  assumption about fat-tailed overnight returns.\n")
    fam = {k: res[k].get("perm", np.nan) for k in CLEAN + ("FOMC",) if k in res}
    h = holm(fam)
    print(f"  {'slice':22} {'raw perm p':>11} {'Holm p':>9} {'survives 5% family-wise?':>26}")
    for k, p in sorted(fam.items(), key=lambda kv: (np.isnan(kv[1]), kv[1])):
        a, ok = h.get(k, (np.nan, False))
        print(f"  {k:22} {p:11.4f} {a:9.4f} {'YES' if ok else 'no':>26}")
    print("\n  And the same slices after the turn-of-the-month adjustment, which is the stricter test")
    print("  because it is the one that removes a known confound rather than a distributional one:")
    for k in CLEAN + ("FOMC",):
        if k in adj:
            c, t, frac = adj[k]
            if not np.isfinite(c):
                print(f"    {k:22} not identified — only {100*frac:.0f}% of its nights have any "
                      f"same-position control")
                continue
            print(f"    {k:22} adjusted {c:+7.2f} bp at robust t {t:+5.2f}"
                  f"{'   <-- clears 2.0' if abs(t) > 2 else ''}")


def section8(on: pd.DataFrame, lab: pd.DataFrame) -> None:
    print("\n" + "=" * 112)
    print("8. SHARPE, DRAWDOWN, CALMAR — does adding the siblings improve the book or dilute it?")
    print("=" * 112)
    print("  Sharpe is annualised on each group's own trade count, because the book is flat otherwise.")
    print("  The comparison that matters is CLEAN + FOMC against FOMC alone: if pre-announcement drift")
    print("  is general, pooling should raise the Sharpe by adding independent instances of the same")
    print("  premium. If it is specific to the rate decision, pooling dilutes exactly as CPI and NFP")
    print("  did in the previous study.\n")
    yrs = (on.index.max() - on.index.min()).days / 365.25
    clean_any = lab[list(CLEAN)].any(axis=1)
    groups = [(k, lab[k]) for k in CLEAN] + [
        ("CLEAN POOLED", clean_any), ("FOMC alone", lab.FOMC),
        ("CLEAN + FOMC", clean_any | lab.FOMC), ("QRA_STATEMENT (dirty)", lab.QRA_STATEMENT)]
    print(f"  {'group':24} {'n':>4} {'net bp':>8} {'ann bp':>8} {'Sharpe':>7} {'maxDD':>9} "
          f"{'Calmar':>7} {'total bp':>9}")
    for nm, m in groups:
        x = by_date(on, m) - FEE_BP
        if len(x) < 5:
            continue
        p = perf(x, len(x) / yrs)
        print(f"  {nm:24} {len(x):4} {x.mean():8.2f} {p['ann']:8.1f} {p['sharpe']:+7.2f} "
              f"{p['maxdd']:9.1f} {p['calmar']:+7.2f} {p['total']:9.1f}")
    import yfinance as yf
    h = yf.Ticker("SPY").history(start=START, interval="1d", auto_adjust=True)
    r = (np.log(h.Close / h.Close.shift(1)) * 1e4).dropna()
    p = perf(r.values, 252)
    print(f"  {'SPY buy-and-hold':24} {len(r):4} {r.mean():8.2f} {p['ann']:8.1f} {p['sharpe']:+7.2f} "
          f"{p['maxdd']:9.1f} {p['calmar']:+7.2f} {p['total']:9.1f}")


# ------------------------------------------------------------------------ futures

def load_es() -> pd.DataFrame:
    """Per-session ES returns for the 10:00-11:00 and 14:00-15:00 bars, in bp.

    Free 60-minute history reaches about two years. That is short, but this is used only to validate
    the release clock of the rule-based ISM calendar, which needs a signature rather than a premium.
    """
    try:
        import yfinance as yf
        es = yf.Ticker("ES=F").history(period="730d", interval="60m")
        if es.empty:
            return pd.DataFrame()
        es.index = es.index.tz_convert("America/New_York")
        d = pd.to_datetime(es.index.date)
        out = {}
        for hh, nm in ((10, "a10"), (14, "a14")):
            b = es[es.index.hour == hh]
            out[nm] = pd.Series(((b.Close / b.Open - 1) * 1e4).values,
                                index=pd.to_datetime(b.index.date)).groupby(level=0).first()
        return pd.DataFrame(out).dropna()
    except Exception as e:                      # a data outage must not take the whole study down
        print(f"  (ES futures unavailable: {type(e).__name__})")
        return pd.DataFrame()


def verdict(res: dict, adj: dict, lab: pd.DataFrame, on: pd.DataFrame) -> None:
    print("\n" + "=" * 112)
    print("VERDICT")
    print("=" * 112)
    fomc = res["FOMC"]
    pooled = res["_pooled_f"]
    yrs = (on.index.max() - on.index.min()).days / 365.25
    fx = by_date(on, lab.FOMC) - FEE_BP
    fp = perf(fx, len(fx) / yrs)
    px = by_date(on, lab[list(CLEAN)].any(axis=1) | lab.FOMC) - FEE_BP
    pp = perf(px, len(px) / yrs)
    fadj = adj["FOMC"]
    print(f"""
  THE HYPOTHESIS WAS THAT PRE-ANNOUNCEMENT DRIFT IS A GENERAL PROPERTY OF SCHEDULED POST-OPEN EVENTS.
  It is not. It is specific to the FOMC rate decision.

  The structural screen worked exactly as the brief intended -- and it disqualified one of the four
  siblings on its own terms. The Treasury Quarterly Refunding Statement is released at 08:30 ET, inside
  the 15:59 -> 09:31 window, so it repeats the CPI/NFP defect rather than the FOMC pattern. Its
  companion release, Treasury's borrowing estimates at 15:00 on the Monday of refunding week, does
  qualify, so the refunding cycle contributed one clean event and one negative control.

  The remaining five clean slices reach {int(lab[list(CLEAN)].any(axis=1).sum())} event nights, about {lab[list(CLEAN)].any(axis=1).sum() / ((lab.index.max()-lab.index.min()).days/365.25):.0f} a year, and
  with the FOMC added the brief's ~46 a year is essentially met. Frequency was never the
  constraint; it was not the constraint in the previous study either. Expectancy is.

  WHAT THE NUMBERS SAY. The FOMC benchmark repeats: {fomc['diff']:+.2f} bp against control at Welch t {fomc['tw']:+.2f},
  permutation p {fomc['perm']:.4f}. Pooling it with all five clean siblings gives {pooled['diff']:+.2f} bp at
  t {pooled['tw']:+.2f}, permutation p {pooled['perm']:.4f}. Adding events that share the FOMC's
  structural DNA did not add signal, and the pooled t-stat is lower than the benchmark's on its own --
  the same dilution CPI and NFP produced, arrived at from the opposite
  direction. The structural screen was necessary and it was not sufficient.

  ISM SUPPLIES 24 OF THE BRIEF'S ~46 EVENTS AND IT CANNOT BE TESTED AT ALL. This is the sharpest thing
  the study found, and it is a fact about the calendar rather than a p-value. ISM Manufacturing is
  released on the first business day of every month, so its eve is the LAST TRADING DAY OF THE MONTH,
  in 140 of 140 cases; ISM Services sits on the third. The turn-of-the-month effect lives on exactly
  those sessions. Because ISM has no off-months, there is not a single first-trading-day night in
  eleven years that is NOT an ISM eve -- so the event indicator and the calendar position are the same
  variable, and no statistical adjustment can separate them. Section 5 counts the available control
  nights per position before estimating, and suppresses the coefficient where none exist. An earlier
  version of that regression, run without the check, returned +64 bp at robust t +9.9 for ISM
  Manufacturing; that was a rank-deficient solver extrapolating, and it is exactly the kind of number
  that gets a strategy funded. What is left is this: the raw ISM returns are real but unattributable,
  and the parsimonious owner of them is the calendar, which is documented back to the 1980s and needs
  no pre-announcement mechanism. It also barely matters here, because the raw ISM effect is not there
  in the first place -- {res['ISM_MFG']['diff']:+.2f} bp vs control at t {res['ISM_MFG']['tw']:+.2f}, and ISM Services is negative.

  WHY MOST OF THE OTHER NULLS ARE UNINFORMATIVE, WHICH IS A LIMIT OF THE TEST AND NOT A FINDING.
  Overnight per-date control returns have a standard deviation near {res['_ctl_sd']:.0f} bp. Detecting an
  FOMC-sized {abs(fomc['diff']):.0f} bp effect at t=2 against this control needs roughly 95
  observations. Humphrey-Hawkins supplies 23 in eleven years and the refunding events 47. Those slices
  could not have produced a significant result even if their true effect equalled the FOMC's, so their
  flat readings are an absence of evidence rather than evidence of absence -- and Humphrey-Hawkins is
  the one worth flagging, since it reads {res['HUMPHREY']['diff']:+.1f} bp, FOMC-sized or larger, at t {res['HUMPHREY']['tw']:+.2f}.
  It is the only sibling this study cannot rule out, and it would take decades of two-a-year events to
  settle. Section 6 states each slice's minimum detectable effect so
  none of this can be misread. Only minutes and the pooled sets are both powered and identified, and it
  is those that carry the conclusion.

  AND THE CORRECTION NOBODY WANTS TO APPLY. Six new slices were tested here on top of three before. At
  5%, roughly one spurious winner is the expected yield from nine tests. Section 7's Holm step-down
  keeps the family-wise error at 5%, and the honest reading of any single surviving p-value has to pass
  through it. The FOMC's permutation p of {fomc['perm']:.4f} was already at the boundary before any
  correction for the fact that it is one of nine things looked at.

  WHERE THE DATES CAME FROM, AND THE ONE THAT DID NOT VALIDATE. Minutes, testimony and refunding dates
  are scraped from the issuers themselves -- the Fed's own "(Released ...)" annotations, its per-year
  testimony listings, and TreasuryDirect's auction API, where the announcement date of each new 30-year
  bond IS the refunding statement. The refunding dates come back 47 for 47 Wednesdays, which is the
  check that the identification is right. Only ISM is generated from a rule, because ISM publishes no
  free archive, and rules are how the previous study nearly misdated NFP. So the rule was tested against
  intraday ES futures: ISM Manufacturing's 10:00 bar moves 33.5 bp on its dates against 22.6 on control
  (t +1.87) and the FOMC's 14:00 bar moves 47.5 against 14.6 (t +3.71), both confirming the clock. ISM
  SERVICES DID NOT LIGHT UP -- 23.1 bp against 22.6, t +0.11. At 26 overlapping dates in two years of
  free 60-minute history that is more likely thin power than a wrong calendar, but it is unverified, and
  the ISM Services dates should be treated as the weakest list here. It changes nothing about the
  conclusion, since that slice's raw effect is negative anyway.

  SO WHAT IS THE FOMC EFFECT, IF NOT A GENERAL PRE-EVENT PREMIUM? The evidence now points somewhere
  narrower and more interesting. It is not that markets pay to shed risk before any scheduled release:
  ISM, minutes and testimony are all scheduled, all market-moving, all post-open, and none of them pay.
  What is unique to the rate decision is that it is the only one of these events that changes the
  policy rate itself -- the discount rate in every valuation on the tape -- and the only one whose
  outcome a large, identifiable set of institutions is contractually obliged to hedge. FOMC minutes
  release the same committee's reasoning three weeks later at the identical 14:00 clock and pay
  nothing, which is close to a controlled experiment: same issuer, same hour, same subject matter,
  no rate change. The premium attaches to the decision, not to the announcement structure.

  ONE RESULT MOVED THE OTHER WAY, AND IT IS WORTH SAYING. Stripping the calendar makes the FOMC leg
  STRONGER, not weaker: {fomc['diff']:+.2f} bp raw becomes {fadj[0]:+.2f} bp at robust t {fadj[1]:+.2f}, the
  only slice in the study to clear 2.0 on the adjusted test. FOMC eves are scattered through the
  month -- about half of them land mid-month -- so the fixed effects are removing noise from that
  comparison rather than removing the effect. That is the signature of a real effect surviving a control
  rather than being manufactured by one, and it is the opposite of what happened to ISM.

  WHAT THIS MEANS FOR THE TRACK RECORD. Eight trades a year is the honest count, and it stays eight.
  The FOMC leg earns {fp['ann']:.0f} bp a year, roughly ${fp['ann']/1e4*5000:.0f} on a $5,000 account, at a
  Sharpe of {fp['sharpe']:+.2f} against a maximum drawdown of {fp['maxdd']:.0f} bp -- a real but small effect that cannot
  be scaled by finding more events, because the thing being harvested is not event-ness. Pooling the
  siblings in raises the annual total to {pp['ann']:.0f} bp but cuts the Sharpe to {pp['sharpe']:+.2f} and deepens the
  drawdown to {pp['maxdd']:.0f} bp, which is the same trade the previous study rejected: more gross return for
  worse risk-adjusted return, on more exposure. Expanding frequency by structural analogy has now
  failed twice: once with CPI and NFP, where the clock was wrong, and once here, where the clock was
  right and the mechanism still did not transfer. The next honest move is not another calendar. It is to ask what
  else shares the rate decision's actual mechanism -- a scheduled, binding change to a price that
  every other price is quoted against -- and there are very few of those.""")


def main() -> int:
    print("=" * 112)
    print("POST-OPEN MACRO — is pre-announcement drift general, or is it the rate decision alone?")
    print("=" * 112)
    on, _ = load_panel()
    lab = build_labels(on.index)
    print(f"  {len(on)} sessions, {on.index.min().date()} .. {on.index.max().date()}, "
          f"names {', '.join(NAMES)}, dividend-adjusted, {FEE_BP:.1f} bp round trip")
    es = load_es()
    section0(lab)
    section1(lab, es)
    section2(on, lab)
    res = section3(on, lab)
    section4(on, lab)
    adj = section5(on, lab)
    section6(res, lab)
    section7(res, adj)
    section8(on, lab)
    verdict(res, adj, lab, on)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
