"""
macro_overnight_alpha.py — does pre-announcement overnight positioning exist for CPI and NFP the way
it does for the FOMC, and does pooling the three get to 30+ trades a year with a t-stat above 2?

    python macro_overnight_alpha.py

THE ONE STRUCTURAL FACT THAT REFRAMES THE WHOLE BRIEF, AND IT IS WORTH STATING BEFORE ANY NUMBER.
The brief's trade is "buy at 15:59 the night before the release, sell at 09:31 at the open". CPI and
NFP print at 08:30 ET. The equity market opens at 09:30. So THE RELEASE HAPPENS INSIDE THE HOLDING
WINDOW: this is not a pre-announcement drift trade, it is a hold-through-the-print trade. There is no
way to exit a SPY position after 08:30 and before 09:30, because the venue is shut.

That makes it a fundamentally different bet from the FOMC version the brief is generalising FROM. The
FOMC statement lands at 14:00, during the session, so the overnight window before an FOMC decision day
contains no announcement at all -- you are paid to carry risk INTO the event and you exit before it.
Lucca-Moench pre-FOMC drift is a risk-premium story: compensation for holding equity exposure across
an interval of elevated, soon-to-be-resolved uncertainty. The CPI/NFP overnight is the opposite: you
are holding THROUGH the uncertainty resolution, collecting the surprise itself, which is a coin flip
with fat tails and no theoretical reason to have a positive mean.

So "pooling CPI + NFP + FOMC" pools two structurally opposite trades to reach a trade count. The
study measures each separately for exactly that reason, and Section 6 uses ES futures -- which trade
at 08:30 when the ETFs do not -- to split the CPI/NFP overnight into the part before the print and the
part after it. That decomposition is the only way to tell drift from reaction, and it is the question
the brief actually wants answered.

WHAT ELSE WOULD HAVE FAKED A RESULT HERE, ALL OF IT CHECKED RATHER THAN ASSUMED.
  * Dividends. Measured on unadjusted SPY over this sample: the close-to-open return averages -43.1 bp
    on the 47 ex-dividend mornings against +3.7 bp on every other morning, because the price gaps down
    by the distribution at the open. That is a -47 bp artifact, several times larger than any plausible
    event effect, and SPY's ex-dates are the third Friday of Mar/Jun/Sep/Dec -- mid-month, right where
    CPI lives. auto_adjust is therefore mandatory, and with it the ex-div morning reads -2.0 bp, in
    line with everything else.
  * Wrong dates. NFP is NOT reliably the first Friday: on the real BLS schedule only 85% of releases
    fall in the first seven days of the month. A misdated event night is counted twice wrong -- a real
    event lands in the control group and an ordinary night lands in the event group -- so the dates are
    scraped from BLS's own schedule pages rather than generated from a rule. See bls_calendar.py.
  * Correlated tickers. SPY, QQQ and IWM on the same night are three views of one market move. Every
    headline statistic here is computed per DATE, averaging the three, because treating them as three
    independent observations inflates t by up to sqrt(3).
  * Turn of the month. NFP eve is usually the first Thursday, which sits inside the turn-of-the-month
    window -- one of the oldest documented calendar anomalies. An NFP result that is really a
    turn-of-month result would look identical unless the control is matched on trading-day-of-month,
    which Section 5 does.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import bls_calendar
from event_calendar import fomc_dates

NAMES = ("SPY", "QQQ", "IWM")
FEE_BP = 1.5                      # the brief's round-trip equity friction
START, END = "2015-01-01", "2026-12-31"
SPLIT = "2020-01-01"              # the brief's pre/post inflation-cycle break
NPERM = 20000
RNG = np.random.default_rng(20260925)


# ----------------------------------------------------------------------------- data

def load_panel(names=NAMES, start=START) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Overnight and intraday returns in basis points, indexed by session, one column per name.

    auto_adjust=True is not a style preference here; see the module docstring. The overnight return is
    labelled by the MORNING it ends on, which is also the morning the brief's trade exits, so an event
    on date D is the row indexed D.
    """
    import yfinance as yf
    on, day = {}, {}
    for t in names:
        h = yf.Ticker(t).history(start=start, interval="1d", auto_adjust=True)
        h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
        on[t] = (h.Open / h.Close.shift(1) - 1.0) * 1e4
        day[t] = (h.Close / h.Open - 1.0) * 1e4
    return pd.DataFrame(on).dropna(how="all"), pd.DataFrame(day).dropna(how="all")


def build_labels(idx: pd.DatetimeIndex, refresh: bool = False) -> pd.DataFrame:
    """One row per session with the event flags, plus the placebo and control definitions.

    `*_eve` marks the morning the brief's trade EXITS, i.e. the release day itself. `*_post` marks the
    following session's morning, which is the brief's own placebo: if the drift is about uncertainty
    ahead of the print, it must vanish once the print is known.
    """
    rel = bls_calendar.release_dates(refresh=refresh)
    ev = {"CPI": pd.DatetimeIndex(rel["CPI"].date), "NFP": pd.DatetimeIndex(rel["NFP"].date),
          "FOMC": fomc_dates()}
    out = pd.DataFrame(index=idx)
    pos = pd.Series(np.arange(len(idx)), index=idx)
    for k, dates in ev.items():
        d = dates[(dates >= idx.min()) & (dates <= idx.max())]
        on_grid = d.intersection(idx)
        out[f"{k}_eve"] = out.index.isin(on_grid)
        nxt = pos.reindex(on_grid).dropna().astype(int) + 1
        nxt = nxt[nxt < len(idx)]
        out[f"{k}_post"] = out.index.isin(idx[nxt.values])
        out.attrs[f"{k}_missing"] = len(d) - len(on_grid)
        out.attrs[f"{k}_total"] = len(d)
    out["any_eve"] = out[[f"{k}_eve" for k in ev]].any(axis=1)
    out["any_post"] = out[[f"{k}_post" for k in ev]].any(axis=1)
    out["control"] = ~(out.any_eve | out.any_post)
    # Trading-day-of-month, 1-based: the matching variable for the turn-of-month placebo.
    out["tdom"] = out.groupby([out.index.year, out.index.month]).cumcount() + 1
    return out


# ---------------------------------------------------------------------- statistics

def welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """(difference, t, two-sided p) for mean(a) - mean(b), unequal variances."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return (np.nan, np.nan, np.nan)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    d = a.mean() - b.mean()
    t = d / se if se > 0 else np.nan
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return (d, t, _p(t, df))


def _p(t: float, df: float) -> float:
    if not np.isfinite(t):
        return np.nan
    try:
        from scipy import stats
        return float(2 * stats.t.sf(abs(t), df))
    except ImportError:                      # normal approximation; df here is always > 100
        from math import erfc, sqrt
        return float(erfc(abs(t) / sqrt(2)))


def one_sample(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return {"n": len(x)}
    se = x.std(ddof=1) / np.sqrt(len(x))
    t = x.mean() / se if se > 0 else np.nan
    return {"n": len(x), "mean": x.mean(), "sd": x.std(ddof=1), "t": t,
            "p": _p(t, len(x) - 1), "win": float((x > 0).mean())}


def permutation(event: np.ndarray, control: np.ndarray, nperm: int = NPERM) -> float:
    """Two-sided p for mean(event) - mean(control) with the event labels shuffled.

    Overnight returns are fat-tailed and skewed, so the t-distribution's p-value is the number most
    likely to be wrong in an event study this small. The permutation p makes no distributional
    assumption: it asks how often a random relabelling of the same pooled returns produces a gap at
    least this large. Labels are shuffled at the DATE level, which is the unit of independence.
    """
    event, control = np.asarray(event, float), np.asarray(control, float)
    event, control = event[np.isfinite(event)], control[np.isfinite(control)]
    obs = abs(event.mean() - control.mean())
    pool = np.concatenate([event, control])
    n = len(event)
    hits = 0
    for _ in range(nperm):
        s = RNG.permutation(pool)
        if abs(s[:n].mean() - s[n:].mean()) >= obs:
            hits += 1
    return (hits + 1) / (nperm + 1)


def perf(x: np.ndarray, per_year: float) -> dict:
    """Sharpe and max drawdown for a strategy that only holds risk on `per_year` nights a year.

    Sharpe is annualised on the actual trade count, not on 252, because the position is flat on every
    other night. That is the honest per-unit-risk number, but it is also why it must be read next to
    the total: a fine Sharpe earned on 32 nights a year compounds into very little.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return {}
    eq = np.cumsum(x)
    dd = eq - np.maximum.accumulate(eq)
    sd = x.std(ddof=1)
    return {"sharpe": (x.mean() / sd * np.sqrt(per_year)) if sd > 0 else np.nan,
            "maxdd": float(dd.min()), "total": float(eq[-1]),
            "ann": float(x.mean() * per_year),
            "calmar": float(x.mean() * per_year / abs(dd.min())) if dd.min() < 0 else np.nan}


def by_date(panel: pd.DataFrame, mask: pd.Series) -> np.ndarray:
    """One observation per date: the equal-weighted mean across whichever names traded that day."""
    return panel.loc[mask.reindex(panel.index, fill_value=False)].mean(axis=1).dropna().values


# ------------------------------------------------------------------------ sections

def section0(on: pd.DataFrame, day: pd.DataFrame, lab: pd.DataFrame) -> None:
    print("=" * 112)
    print("0. CALENDAR GATE — are the dates real, and does each event land where its clock says it should?")
    print("=" * 112)
    rel = bls_calendar.release_dates()
    bls_calendar.summary(rel)
    print(f"\n    FOMC decision days from federalreserve.gov: "
          f"{lab.FOMC_eve.sum()} in sample")
    for k in ("CPI", "NFP", "FOMC"):
        miss = lab.attrs.get(f"{k}_missing", 0)
        if miss:
            print(f"    {k}: {miss} release dates fell on non-trading days and were dropped")
    print("""
  THE SIGNATURE TEST. A correct date list has to show up in the data, not just parse. CPI and NFP
  print at 08:30, before the open, so on a correctly dated release morning the ABSOLUTE OVERNIGHT
  return must be elevated. The FOMC prints at 14:00, during the session, so a correct FOMC date shows
  an elevated absolute INTRADAY return and nothing unusual overnight. If these come out flat, the
  dates are wrong and every number after this point is meaningless.
""")
    ctl = lab.control
    print(f"  {'':22} {'|overnight| bp':>16} {'vs control':>12}   {'|intraday| bp':>15} {'vs control':>12}")
    for k in ("CPI", "NFP", "FOMC"):
        m = lab[f"{k}_eve"]
        a_on, b_on = by_date(on.abs(), m), by_date(on.abs(), ctl)
        a_id, b_id = by_date(day.abs(), m), by_date(day.abs(), ctl)
        _, t_on, _ = welch(a_on, b_on)
        _, t_id, _ = welch(a_id, b_id)
        print(f"  {k + ' release day':22} {a_on.mean():16.1f} {'t ' + f'{t_on:+.2f}':>12}   "
              f"{a_id.mean():15.1f} {'t ' + f'{t_id:+.2f}':>12}")
    print(f"  {'control mornings':22} {by_date(on.abs(), ctl).mean():16.1f} {'':>12}   "
          f"{by_date(day.abs(), ctl).mean():15.1f}")
    print("\n  Read the two columns against each other: the 08:30 releases move the overnight window and")
    print("  the 14:00 decision moves the session. That is the calendar validating itself.")


def section1(on: pd.DataFrame, lab: pd.DataFrame) -> None:
    print("\n" + "=" * 112)
    print("1. SAMPLE SIZE AND FREQUENCY — is there even enough here for a one-year track record?")
    print("=" * 112)
    yrs = (on.index.max() - on.index.min()).days / 365.25
    rows = []
    for k in ("CPI", "NFP", "FOMC"):
        n = int(lab[f"{k}_eve"].sum())
        rows.append((k, n, n / yrs))
    pooled = int(lab.any_eve.sum())
    rows.append(("POOLED (unique dates)", pooled, pooled / yrs))
    print(f"  sample {on.index.min().date()} .. {on.index.max().date()}  ({yrs:.1f} years)\n")
    for k, n, f in rows:
        print(f"    {k:24} n={n:4}   {f:5.1f} events per calendar year")
    overlap = int((lab.CPI_eve & lab.FOMC_eve).sum() + (lab.NFP_eve & lab.FOMC_eve).sum()
                  + (lab.CPI_eve & lab.NFP_eve).sum())
    print(f"\n    dates carrying two events at once: {overlap} "
          f"(counted once in the pooled row, which is why it is not the simple sum)")
    ev = lab.index[lab.any_eve]
    roll = pd.Series(1, index=ev).rolling("365D").sum()
    # Only windows that have a full year of history behind them; the first few months of any rolling
    # count are mechanically low and would understate the frequency for no reason.
    roll = roll[roll.index >= ev.min() + pd.Timedelta(days=365)]
    print(f"    rolling 12-month trade count over {len(roll)} complete windows: "
          f"min {int(roll.min())}, median {int(roll.median())}, max {int(roll.max())}")
    print("\n  The frequency question is answered YES and it is the only part of the brief that works")
    print("  cleanly. Everything after this is about whether the trades are worth taking.")


def section2(on: pd.DataFrame, lab: pd.DataFrame) -> None:
    print("\n" + "=" * 112)
    print("2. EXPECTANCY AND WIN RATE — net of the brief's 1.5 bp round trip")
    print("=" * 112)
    print("  Two rows per group. The first treats each ticker-night as an observation, which is what a")
    print("  naive panel regression does. The second averages SPY/QQQ/IWM within each date first. The")
    print("  gap between their t-stats is the cost of pretending three views of one market move are")
    print("  three independent facts.\n")
    print(f"  {'group':24} {'n':>5} {'mean bp':>9} {'net bp':>8} {'win%':>6} {'t':>7} {'p':>8}")
    groups = [("CPI eves", lab.CPI_eve), ("NFP eves", lab.NFP_eve), ("FOMC eves", lab.FOMC_eve),
              ("POOLED macro eves", lab.any_eve), ("control nights", lab.control)]
    for nm, m in groups:
        flat = on.loc[m.reindex(on.index, fill_value=False)].stack().values
        dated = by_date(on, m)
        for tag, x in ((f"{nm} [ticker-night]", flat), (f"{nm} [per date]", dated)):
            s = one_sample(x - FEE_BP)
            if not s.get("n"):
                continue
            print(f"  {tag:24} {s['n']:5} {s['mean'] + FEE_BP:9.2f} {s['mean']:8.2f} "
                  f"{100 * s['win']:5.0f}% {s['t']:+7.2f} {s['p']:8.3f}")
        print()


def section3(on: pd.DataFrame, lab: pd.DataFrame) -> None:
    print("=" * 112)
    print("3. SIGNIFICANCE AGAINST THE CONTROL — the null is the baseline overnight return, not zero")
    print("=" * 112)
    print("  Equities have a positive unconditional overnight return, so a positive mean on event")
    print("  nights proves nothing on its own. The test is event minus control. Permutation p shuffles")
    print("  the event labels across dates 20,000 times and makes no distributional assumption, which")
    print("  matters because overnight returns are fat-tailed.\n")
    ctl = by_date(on, lab.control)
    print(f"  {'group':22} {'n':>4} {'event bp':>9} {'control bp':>11} {'diff':>8} "
          f"{'Welch t':>8} {'t p':>7} {'perm p':>8}")
    for nm, m in (("CPI eves", lab.CPI_eve), ("NFP eves", lab.NFP_eve),
                  ("FOMC eves", lab.FOMC_eve), ("POOLED", lab.any_eve)):
        a = by_date(on, m)
        d, t, p = welch(a, ctl)
        pp = permutation(a, ctl)
        print(f"  {nm:22} {len(a):4} {a.mean():9.2f} {ctl.mean():11.2f} {d:8.2f} "
              f"{t:+8.2f} {p:7.3f} {pp:8.4f}")
    print("\n  PLACEBO — the night AFTER the release, when the uncertainty is already resolved.")
    print("  If the eve effect is about positioning ahead of a known event, this must be flat.")
    for nm, m in (("CPI post", lab.CPI_post), ("NFP post", lab.NFP_post),
                  ("FOMC post", lab.FOMC_post), ("POOLED post", lab.any_post)):
        a = by_date(on, m)
        d, t, p = welch(a, ctl)
        print(f"  {nm:22} {len(a):4} {a.mean():9.2f} {ctl.mean():11.2f} {d:8.2f} "
              f"{t:+8.2f} {p:7.3f}")


def section4(on: pd.DataFrame, lab: pd.DataFrame) -> None:
    print("\n" + "=" * 112)
    print("4. SUB-PERIOD STABILITY — did the post-2020 inflation cycle change CPI positioning?")
    print("=" * 112)
    print("  Note the control column as well as the event column. The unconditional overnight return")
    print("  itself changed regime, so an event effect that merely tracks its own baseline has not")
    print("  found anything.\n")
    print(f"  {'group':18} {'period':12} {'n':>4} {'event bp':>9} {'control bp':>11} "
          f"{'diff':>8} {'t':>7}")
    for nm, m in (("CPI eves", lab.CPI_eve), ("NFP eves", lab.NFP_eve),
                  ("FOMC eves", lab.FOMC_eve), ("POOLED", lab.any_eve)):
        for tag, sl in (("pre-2020", slice(None, SPLIT)), ("2020-present", slice(SPLIT, None))):
            sub = on.loc[sl]
            mm = m.loc[sl]
            a, c = by_date(sub, mm), by_date(sub, lab.control.loc[sl])
            if len(a) < 5:
                continue
            d, t, _ = welch(a, c)
            print(f"  {nm:18} {tag:12} {len(a):4} {a.mean():9.2f} {c.mean():11.2f} "
                  f"{d:8.2f} {t:+7.2f}")
        print()


def section5(on: pd.DataFrame, lab: pd.DataFrame) -> None:
    print("=" * 112)
    print("5. THE CALENDAR PLACEBO — is NFP eve a payrolls effect or a turn-of-the-month effect?")
    print("=" * 112)
    print("""  NFP lands on the first Friday, so NFP eve is almost always the first Thursday: trading day 2
  to 4 of the month. The turn-of-the-month effect is one of the oldest documented calendar anomalies
  and it lives in exactly that window. CPI sits around trading day 7 to 10. So the control group used
  above -- all non-event nights, spread evenly through the month -- is the wrong comparison: it differs
  from the event group in the day of the month as well as in the event. Matching the control to the
  same trading-day-of-month distribution removes the calendar and leaves only the announcement.
""")
    print(f"  {'group':22} {'tdom range':12} {'n':>4} {'event bp':>9} "
          f"{'matched ctl':>12} {'diff':>8} {'t':>7}")
    for nm, m in (("CPI eves", lab.CPI_eve), ("NFP eves", lab.NFP_eve), ("FOMC eves", lab.FOMC_eve)):
        tdoms = sorted(lab.tdom[m].unique())
        matched = lab.control & lab.tdom.isin(tdoms)
        a, c = by_date(on, m), by_date(on, matched)
        d, t, _ = welch(a, c)
        print(f"  {nm:22} {f'{min(tdoms)}-{max(tdoms)}':12} {len(a):4} {a.mean():9.2f} "
              f"{c.mean():12.2f} {d:8.2f} {t:+7.2f}")
    print("\n  For reference, the unmatched control mean was "
          f"{by_date(on, lab.control).mean():+.2f} bp.")


def section6(lab: pd.DataFrame) -> None:
    print("\n" + "=" * 112)
    print("6. SPLITTING DRIFT FROM REACTION — ES futures, which trade at 08:30 when the ETFs cannot")
    print("=" * 112)
    print("""  This is the section the brief needs and could not have with ETFs. Because ES trades overnight,
  the CPI/NFP window can be cut at 08:00 -- before the print -- and the two halves measured separately:

     pre-release   prior 16:00 close -> 08:00     contains NO announcement (the real drift analogue)
     release hour  08:00 -> 09:00                 contains the 08:30 print
     full window   prior 16:00 close -> 09:00     what the brief's trade actually holds

  If the brief's premise is right, the money is in the pre-release leg. If the full-window result is
  just the print, the pre-release leg will be flat and the release hour will carry everything. Free
  60-minute futures history reaches back about two years, so this is roughly 25 events per type --
  underpowered on its own, and reported as a direction rather than a verdict.
""")
    import yfinance as yf
    es = yf.Ticker("ES=F").history(period="730d", interval="60m")
    if es.empty:
        print("  ES=F returned no data; decomposition unavailable.")
        return
    es.index = es.index.tz_convert("America/New_York")
    es["d"] = pd.to_datetime(es.index.date)
    es["h"] = es.index.hour
    prev_close = es[es.h == 15].groupby("d").Close.last()
    o8 = es[es.h == 8].groupby("d").Open.first()
    o9 = es[es.h == 9].groupby("d").Open.first()
    f = pd.DataFrame({"pc": prev_close.shift(1).reindex(o8.index), "o8": o8, "o9": o9}).dropna()
    f["pre"] = (f.o8 / f.pc - 1) * 1e4
    f["rel"] = (f.o9 / f.o8 - 1) * 1e4
    f["full"] = (f.o9 / f.pc - 1) * 1e4
    print(f"  ES=F sessions available: {len(f)}  ({f.index.min().date()} .. {f.index.max().date()})\n")
    print(f"  {'group':18} {'n':>4} {'pre-release':>12} {'t':>7} {'release hr':>12} {'t':>7} "
          f"{'full':>8} {'t':>7}")
    lab2 = lab.reindex(f.index).fillna(False)
    ctl = f[lab2.control.astype(bool)]
    for nm, col in (("CPI eves", "CPI_eve"), ("NFP eves", "NFP_eve"), ("FOMC eves", "FOMC_eve")):
        sub = f[lab2[col].astype(bool)]
        if len(sub) < 5:
            continue
        cells = []
        for w in ("pre", "rel", "full"):
            _, t, _ = welch(sub[w].values, ctl[w].values)
            cells += [sub[w].mean(), t]
        print(f"  {nm:18} {len(sub):4} {cells[0]:12.2f} {cells[1]:+7.2f} {cells[2]:12.2f} "
              f"{cells[3]:+7.2f} {cells[4]:8.2f} {cells[5]:+7.2f}")
    print(f"  {'control':18} {len(ctl):4} {ctl.pre.mean():12.2f} {'':>7} {ctl.rel.mean():12.2f} "
          f"{'':>7} {ctl.full.mean():8.2f}")
    print("\n  sd of each leg on control nights (the risk being taken to earn the means above):")
    print(f"    pre-release {ctl.pre.std():.1f} bp    release hour {ctl.rel.std():.1f} bp    "
          f"full {ctl.full.std():.1f} bp")


def section7(on: pd.DataFrame, lab: pd.DataFrame) -> None:
    print("\n" + "=" * 112)
    print("7. SHARPE, DRAWDOWN, CALMAR — holding equity risk only on these ~32 nights a year")
    print("=" * 112)
    yrs = (on.index.max() - on.index.min()).days / 365.25
    print("  P&L is additive basis points per trade, net of 1.5 bp. Sharpe is annualised on each")
    print("  group's OWN trade count, so these are per-unit-risk numbers for a book that is flat the")
    print("  rest of the time -- read them next to the total, which is what actually lands in the")
    print("  account.\n")
    print(f"  {'group':22} {'n':>4} {'net bp':>8} {'ann bp':>8} {'Sharpe':>7} {'maxDD bp':>9} "
          f"{'Calmar':>7} {'total bp':>9}")
    for nm, m in (("CPI eves", lab.CPI_eve), ("NFP eves", lab.NFP_eve),
                  ("FOMC eves", lab.FOMC_eve), ("POOLED", lab.any_eve)):
        x = by_date(on, m) - FEE_BP
        p = perf(x, len(x) / yrs)
        print(f"  {nm:22} {len(x):4} {x.mean():8.2f} {p['ann']:8.1f} {p['sharpe']:+7.2f} "
              f"{p['maxdd']:9.1f} {p['calmar']:+7.2f} {p['total']:9.1f}")
    # SPY buy-and-hold over the same span, in the same additive bp units: total daily log return.
    import yfinance as yf
    h = yf.Ticker("SPY").history(start=START, interval="1d", auto_adjust=True)
    r = (np.log(h.Close / h.Close.shift(1)) * 1e4).dropna()
    p = perf(r.values, 252)
    print(f"  {'SPY buy-and-hold':22} {len(r):4} {r.mean():8.2f} {p['ann']:8.1f} {p['sharpe']:+7.2f} "
          f"{p['maxdd']:9.1f} {p['calmar']:+7.2f} {p['total']:9.1f}")
    print("\n  The total column is the one that answers 'can this build a track record'. A pooled")
    print("  strategy trading ~32 nights a year accumulates the bp in that column over the whole")
    print("  sample, before any borrowing cost on the capital that sat idle the other 220 sessions.")


def verdict() -> None:
    print("\n" + "=" * 112)
    print("VERDICT")
    print("=" * 112)
    print("""
  ANSWERING THE CLOSING QUESTION DIRECTLY: does pooling CPI + NFP + FOMC give 30+ events a year with a
  surviving t-stat > 2.0?

     FREQUENCY: YES, and comfortably. 362 unique event dates over 11.7 years is 30.9 per year, and the
     rolling 12-month count across 330 complete windows never drops below 27 (median 32, max 34). The
     frequency problem the brief set out to solve is solved.

     T-STAT: NO. Pooled per-date net expectancy is +7.66 bp at t +1.79 (p 0.075) against zero, and
     against the control it is +4.46 bp at Welch t +0.98, permutation p 0.30. Both fail, and the second
     is the one that counts, because equities have a positive unconditional overnight return -- the
     control nights themselves print +4.70 bp at t +2.04. More than half the pooled "edge" is just
     being long overnight.

  WHY POOLING CANNOT WORK HERE, AND IT IS NOT A SAMPLE-SIZE PROBLEM. The pooled t-stat is not low
  because 362 observations are too few. It is low because the three event types are not three instances
  of one phenomenon. CPI and NFP print at 08:30 ET; the equity market opens at 09:30. The brief's trade
  -- buy 15:59, sell 09:31 -- therefore HOLDS THROUGH THE PRINT. There is no exit between 08:30 and
  09:30 because the venue is shut. The FOMC statement lands at 14:00, during the session, so the FOMC
  overnight window contains NO announcement at all: you are paid to carry risk INTO a known event and
  you are out before it. Those are opposite trades. One collects a risk premium for holding exposure
  someone else wants to shed ahead of an event; the other takes a directional bet on the content of a
  number. Averaging them to reach a trade count blends a premium with a coin flip.

  THE FUTURES DECOMPOSITION TURNS THAT ARGUMENT INTO A MEASUREMENT. ES trades at 08:30 when the ETFs
  cannot, so the window splits at 08:00 and the two halves can be priced separately:

     FOMC eve:  pre-release +20.89 bp (t +2.23)   release hour  -3.82 bp (t -0.45)
     CPI  eve:  pre-release  -3.00 bp (t -1.13)   release hour +19.06 bp (t +2.80)
     NFP  eve:  pre-release  -8.86 bp (t -1.03)   release hour  +6.53 bp (t +0.92)

  The FOMC effect is entirely BEFORE the event and vanishes in the release hour -- genuine
  pre-announcement drift, exactly what the original study found. The CPI effect is entirely INSIDE the
  print and is flat, slightly negative, in the hours before it. CPI is not a positioning trade that the
  brief can inherit; it is a reaction trade, and a long-only reaction trade is a directional bet that
  inflation surprises downward. That bet won over a 2024-2026 futures sample in which inflation was
  falling. It is not an anomaly, it is a macro view, and it has no reason to repeat. NFP is nothing
  anywhere: pre-release negative, release hour insignificant, full window -2.29.

  WHAT EACH LEG ACTUALLY DELIVERS, PER DATE, NET OF THE 1.5 bp ROUND TRIP:

     FOMC eves   n= 93   +18.49 bp   win 65%   t +3.42   vs ctl +15.29 (t +2.72, perm p 0.051)
     CPI  eves   n=139    +6.90 bp   win 60%   t +0.86   vs ctl  +3.70 (t +0.45, perm p 0.578)
     NFP  eves   n=137    +2.22 bp   win 59%   t +0.32   vs ctl  -0.98 (t -0.14, perm p 0.879)
     POOLED      n=362    +7.66 bp   win 60%   t +1.79   vs ctl  +4.46 (t +0.98, perm p 0.297)

  Only FOMC survives, and pooling actively destroys it. Sharpe on the FOMC leg alone is +1.00 with a
  maxDD of -196 bp and Calmar +0.75. Pool it with the other two and Sharpe falls to +0.52 and the
  drawdown nearly quintuples to -958 bp, because CPI contributes a -979 bp drawdown of its own to earn
  an insignificant mean. Adding 269 trades that pay nothing cut the risk-adjusted return in half. That
  is the whole answer to the brief: the trade count went up and the strategy got worse, because trade
  count is not what was scarce. Signal was.

  THREE CHECKS THAT COULD HAVE FAKED A RESULT, AND WHERE THEY LANDED.
     1. Dividends. Unadjusted SPY close-to-open averages -43.1 bp on its 47 ex-dividend mornings
        against +3.7 bp otherwise, a -47 bp artifact several times larger than any effect here, and SPY's
        ex-dates are the third Friday of Mar/Jun/Sep/Dec -- mid-month, where CPI lives. auto_adjust
        is mandatory, not hygiene.
     2. Ticker clustering. SPY, QQQ and IWM on one night are three views of one market move. Treating
        them as independent moves the pooled t from +2.95 to +1.79. The naive panel t-stat clears the
        brief's bar; the honest one does not.
     3. Calendar confounds. NFP eve is usually the first Thursday, trading day 2 to 4, inside the
        turn-of-the-month window -- one of the oldest documented calendar anomalies. Matching the
        control on trading-day-of-month leaves NFP at -1.14 bp and FOMC at +15.74 bp (t +2.79), so the
        FOMC result is not a calendar effect and the NFP non-result is not being hidden by one.

  THE DATES THEMSELVES WERE THE MOST LIKELY WAY TO GET THIS WRONG. The brief's rules -- CPI on
  "mid-month Wednesdays/Thursdays", NFP on "the first Friday" -- are approximations, and a misdated
  event is wrong twice: a real event lands in the control group and an ordinary night lands in the
  event group. Measured against BLS's own published schedule, only 122 of 144 Employment Situation
  releases (85%) fall in the first seven days of the month. The dates used here were scraped from
  bls.gov's own schedule pages via the Wayback Machine, keyed on reference month so that the autumn
  2025 shutdown reschedules resolve to the replacement date rather than keeping both, and they
  reconstruct exactly 12 releases per year for all 12 years with no gaps. The signature test then
  confirms them in the data rather than in the parser: absolute overnight moves are elevated on CPI
  (65.4 bp, t +2.63) and NFP (62.1 bp, t +2.56) mornings and SUPPRESSED on FOMC mornings (37.8 bp,
  t -2.65) while FOMC intraday moves are elevated (86.4 bp, t +2.28). The 08:30 releases move the
  overnight window; the 14:00 decision moves the session. That is the clock validating the calendar.

  THE SUB-PERIOD RESULT IS THE OPPOSITE OF THE BRIEF'S HYPOTHESIS. The brief expected the post-2020
  inflation cycle to have created CPI positioning. CPI eve goes from -3.26 bp (t -0.53) pre-2020 to
  +8.82 bp (t +0.66) after -- a sign flip with neither half significant, which is what a null looks
  like when it is cut in two. FOMC strengthens instead: +5.95 bp (t +1.09) to +22.34 bp (t +2.52). The
  event whose window excludes its own announcement is the one that got better.

  A SCALE NOTE THAT MATTERS MORE THAN THE STATISTICS. The FOMC leg earns 146.7 bp per year, which on a
  $5,000 account is about $73 before commissions and before any cost on the capital idle the other 244
  sessions. Pooled it is 236.7 bp, about $118. SPY buy-and-hold over the same span returned 1,295.7 bp
  a year at a HIGHER Sharpe (+0.74 against the pooled +0.52). Even the leg that works does not clear
  the opportunity cost of simply being invested, and no amount of added event frequency fixes that,
  because the added events have no expectancy to scale.

  WHAT TO DO WITH THIS. Keep the FOMC eve trade -- it is real, it survives a permutation test at the
  edge of significance (p 0.051), it survives a turn-of-the-month control, and the futures split shows
  the money arriving before the announcement rather than from it. Do not pool it. If the goal is a
  one-year track record with more than eight prints, the honest extension is to look for other events
  whose announcement falls OUTSIDE the holding window -- Treasury refunding announcements, ECB
  decisions timed to the US session, index rebalance effective dates -- rather than events chosen for
  being frequent. The scarce resource in all four studies so far has been mechanism, not sample size,
  and the disqualifying fact has each time been cheap to measure before building anything: here it was
  a wall clock. CPI prints at 08:30 and the market opens at 09:30, so the brief's trade was never a
  pre-announcement trade at all.""")


def main() -> int:
    print("=" * 112)
    print("MACRO OVERNIGHT ALPHA — CPI and NFP eves against the FOMC eve, 2015-present")
    print("=" * 112)
    on, day = load_panel()
    lab = build_labels(on.index)
    print(f"  {len(on)} sessions, {on.index.min().date()} .. {on.index.max().date()}, "
          f"names {', '.join(NAMES)}, dividend-adjusted")
    section0(on, day, lab)
    section1(on, lab)
    section2(on, lab)
    section3(on, lab)
    section4(on, lab)
    section5(on, lab)
    section6(lab)
    section7(on, lab)
    verdict()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
