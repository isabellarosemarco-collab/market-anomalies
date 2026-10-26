"""
backtest.py — does the forced passive flow around S&P 500 reconstitution pay, after costs, out of sample?

THE CONVENTION DETAIL THAT MOVES THE ANSWER MORE THAN ANY PARAMETER. S&P's language is "will replace X in
the S&P 500 effective prior to the opening of trading on Monday, September 21". The index composition
changes at the OPEN of the effective date, which means the passive funds must already own the stock by
then — so they buy in the closing auction of the LAST SESSION BEFORE the effective date. The brief
specifies exiting "at the close on the effective date", which is one session too late: it holds through
the auction and out the other side, collecting the post-inclusion reversal instead of the inclusion
demand. Both are tested here (`exit_offset=-1` for the real auction, `0` for the brief's spelling)
because the difference between them is a measurement of the reversal, not a matter of taste.

WHAT THE ENTRY RULE GIVES AWAY, AND WHY IT IS STILL THE RIGHT RULE. S&P announces after the close, so the
first tradeable price is the next open — and by then the stock has already gapped. Entering at that open
therefore forgoes the single largest piece of the reaction by construction. That is not a flaw in the
rule; it is the honest consequence of not being able to trade a press release that lands at 17:15. The
overnight gap is measured separately below precisely so the size of what is being forgone is visible
rather than assumed, and so nobody later "improves" the backtest by entering at the announcement close.

THREE WAYS THIS STUDY COULD LIE, ALL HANDLED EXPLICITLY.

1. DELETIONS ARE MOSTLY NOT DELETIONS. 73 of 185 removals in the sample happen because the company was
   acquired. Those names are pinned to a deal price and about to stop trading; shorting them is merger
   arbitrage, not index flow, and including them in a "short the deletions" book manufactures a return
   that has nothing to do with the thesis. Only market-cap removals are counted as tradeable deletions.

2. SURVIVORSHIP THROUGH THE DATA FEED. Delisted tickers return nothing from yfinance, so a naive run
   silently keeps only the names that still trade — which for deletions is precisely the ones that did
   not get acquired or go to zero. The fetch success rate is reported per book rather than swallowed, so
   the bias is visible.

3. TICKER REUSE. "P" was Pandora and is now Everpure; a ticker string is not a permanent identifier.
   Requiring bars to exist across the actual event window catches most of this, but it is a known
   residual risk in any ticker-keyed study and is reported, not waved away.

BETA IS ESTIMATED BEFORE THE EVENT, NOT ACROSS IT. The hedge ratio comes from 120 trading days ending 10
sessions before the announcement. Fitting beta on a window that contains the event itself would let the
event's own move set the hedge that is supposed to neutralise it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from index_rebal.changes import Change, fetch

BENCH = "SPY"
# RECYCLED SYMBOLS, AND WHY THIS IS A HARD-CODED LIST RATHER THAN A CHECK. "P" is Everpure, added
# 2026-09-21, and was Pandora Media until 2019. yfinance returns 2595 CONTINUOUS bars under "P" from 2016
# to 2026, $11.31 to $80.14, splicing the two companies together with no gap at all. A discontinuity guard
# was written for this and cannot fire, because there is no discontinuity to find — the vendor has already
# joined the series. Catching this class properly requires a point-in-time security master keyed on a
# permanent identifier (CUSIP/PERMNO), which no free source provides. So the one verified case is excluded
# by name and the limitation is stated rather than papered over with a check that does not work.
RECYCLED = {"P"}
# Plausibility filter on the announcement->effective lead. S&P's own practice is a handful of sessions;
# a 252-day or negative lead is a bad Wikipedia citation, not an exotic index event, and one mis-dated row
# with a huge window can dominate a 90-event sample.
MIN_LEAD, MAX_LEAD = 1, 30


@dataclass
class Trade:
    ticker: str
    side: int                  # +1 long addition, -1 short deletion
    announced: pd.Timestamp
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    gap_bp: float              # close(T_ann) -> open(T_ann+1): the piece you cannot have
    raw_bp: float              # entry open -> exit close, unhedged, signed by `side`
    bench_bp: float            # SPY over the identical window
    beta: float
    hedged_bp: float           # raw - beta * bench, signed
    days_held: int
    dollar_vol: float          # median daily $ volume over the holding window
    path: list[float]          # cumulative hedged bp from the entry open, by session held


def _px(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    out: dict[str, pd.DataFrame] = {}
    # Batch download: one request per ~40 tickers rather than one per name.
    for i in range(0, len(tickers), 40):
        chunk = tickers[i:i + 40]
        try:
            d = yf.download(chunk, start=start, end=end, interval="1d", auto_adjust=False,
                            progress=False, group_by="ticker", threads=True)
        except Exception:
            continue
        for t in chunk:
            try:
                sub = d[t].dropna(how="all")
            except (KeyError, TypeError):
                continue
            if len(sub) > 150:
                out[t] = sub
    return out


def build(changes: list[Change], side: str = "add", exit_offset: int = -1,
          px: dict | None = None) -> tuple[list[Trade], dict]:
    """side='add' -> long additions; side='del' -> short market-cap deletions.

    exit_offset: 0 exits at the close ON the effective date (the brief's spelling); -1 exits at the
    close of the last session BEFORE it, which is the session in which the indexers actually trade.
    """
    sel = []
    for c in changes:
        if c.announced is None or c.lead_days is None:
            continue
        if not (MIN_LEAD <= c.lead_days <= MAX_LEAD):
            continue
        if c.effective.year < 2018:
            continue
        t = c.add_ticker if side == "add" else c.del_ticker
        if not t or t in RECYCLED:
            continue
        if side == "del" and not c.deletion_is_mktcap:
            continue
        sel.append((t, c))

    stats = {"candidates": len(sel)}
    tickers = sorted({t for t, _ in sel})
    if px is None:
        px = _px(tickers + [BENCH], "2016-06-01", "2026-09-26")
    stats["tickers_wanted"] = len(tickers)
    stats["tickers_with_data"] = len([t for t in tickers if t in px])
    if BENCH not in px:
        raise RuntimeError("no benchmark data")
    spy = px[BENCH]
    sret = spy.Close.pct_change()

    sign = 1 if side == "add" else -1
    trades: list[Trade] = []
    for t, c in sel:
        if t not in px:
            continue
        b = px[t]
        idx = b.index
        ann = pd.Timestamp(c.announced)
        eff = pd.Timestamp(c.effective)
        # Entry: first session strictly after the announcement (S&P publishes after the close).
        nxt = idx[idx > ann]
        if len(nxt) == 0:
            continue
        entry = nxt[0]
        # Exit: the session whose close is the indexers' auction, i.e. the last session BEFORE effective.
        before = idx[idx < eff]
        if len(before) == 0:
            continue
        k = len(before) - 1 + (exit_offset + 1)
        if not (0 <= k < len(idx)):
            continue
        exit_ = idx[k]
        if exit_ <= entry:
            continue
        # Beta window: 120 sessions ending 10 sessions before the announcement.
        pre = idx[idx < ann]
        if len(pre) < 140:
            continue
        # LISTING CONTINUITY, WHICH CAUGHT A REAL CONTAMINATED ROW. Ticker "P" is Everpure, added
        # 2026-09-21 — but "P" was Pandora until 2019, so the 140 sessions of "pre-event history" used to
        # fit Everpure's beta were a different company's prices, separated by a multi-year gap that a
        # simple length check sails straight past. Requiring no gap longer than 15 sessions in the year
        # before the event is what distinguishes a continuously listed name from a recycled symbol.
        # A relisting gap is a DIFFERENT failure from the spliced-series one handled by RECYCLED above,
        # and this one is catchable: require the beta window to sit entirely after the last break.
        brk = np.flatnonzero(np.diff(idx.values) > np.timedelta64(45, "D"))
        if len(brk) and (len(pre) - (brk[-1] + 1)) < 140:
            stats.setdefault("relisted", []).append(t)
            continue
        w = pre[-130:-10]
        r_t = b.Close.pct_change().reindex(w).dropna()
        r_s = sret.reindex(w).dropna()
        j = r_t.index.intersection(r_s.index)
        if len(j) < 60:
            continue
        beta = float(np.cov(r_t.loc[j], r_s.loc[j])[0, 1] / max(np.var(r_s.loc[j]), 1e-12))
        beta = float(np.clip(beta, 0.0, 3.0))

        prev = idx[idx < entry]
        if len(prev) == 0:
            continue
        gap = 1e4 * (float(b.Open.loc[entry]) / float(b.Close.loc[prev[-1]]) - 1)
        raw = 1e4 * (float(b.Close.loc[exit_]) / float(b.Open.loc[entry]) - 1)
        try:
            bench = 1e4 * (float(spy.Close.loc[exit_]) / float(spy.Open.loc[entry]) - 1)
        except KeyError:
            continue
        hold = b.loc[entry:exit_]
        # Path from the ENTRY OPEN (not the prior close) so it measures only what the strategy can hold,
        # and hedged so a market rally inside the window is not mistaken for inclusion drift.
        sh = spy.loc[entry:exit_]
        path = []
        for dd in hold.index:
            if dd not in sh.index:
                break
            rs = float(b.Close.loc[dd]) / float(b.Open.loc[entry]) - 1
            rb = float(spy.Close.loc[dd]) / float(spy.Open.loc[entry]) - 1
            path.append(sign * 1e4 * (rs - beta * rb))
        trades.append(Trade(
            ticker=t, side=sign, announced=ann, entry_date=entry, exit_date=exit_,
            gap_bp=sign * gap, raw_bp=sign * raw, bench_bp=bench, beta=beta,
            # Long addition: short beta units of SPY. Short deletion: long beta units. Same expression.
            hedged_bp=sign * (raw - beta * bench),
            days_held=len(hold) - 1,
            dollar_vol=float((hold.Close * hold.Volume).median()), path=path))
    stats["trades"] = len(trades)
    return trades, stats


def summarise(trades: list[Trade], label: str, cost_bp: float = 0.0,
              borrow_bp_yr: float = 0.0) -> dict | None:
    """Per-trade stats. `cost_bp` is the ROUND-TRIP cost charged to every trade."""
    if not trades:
        print(f"  {label}: no trades")
        return None
    raw = np.array([t.raw_bp for t in trades])
    hed = np.array([t.hedged_bp for t in trades])
    gap = np.array([t.gap_bp for t in trades])
    held = np.array([t.days_held for t in trades])
    borrow = borrow_bp_yr * held / 252.0
    net_raw, net_hed = raw - cost_bp - borrow, hed - cost_bp - borrow

    def blk(x, nm):
        # Sharpe on a per-EVENT basis annualised by the number of events per year, not by sqrt(252):
        # these are ~20 discrete holdings a year, not a daily series, and using 252 would inflate it by
        # more than 3x. n_per_year comes from the actual sample span.
        span_yr = max((max(t.exit_date for t in trades) -
                       min(t.entry_date for t in trades)).days / 365.25, 1e-9)
        per_yr = len(trades) / span_yr
        sh = (x.mean() / x.std(ddof=1) * np.sqrt(per_yr)) if x.std(ddof=1) > 0 else float("nan")
        eq = np.cumsum(x)                      # equal-risk, non-compounded: bp added per event
        dd = float((eq - np.maximum.accumulate(eq)).min())
        print(f"    {nm:22} mean {x.mean():+7.1f} bp   median {np.median(x):+7.1f} bp   "
              f"win {100*(x>0).mean():4.0f}%   sd {x.std(ddof=1):6.1f}   "
              f"Sharpe {sh:5.2f}   maxDD {dd:+8.0f} bp   sum {eq[-1]:+8.0f} bp")
        return {"mean": x.mean(), "sharpe": sh, "win": (x > 0).mean(), "sum": eq[-1], "maxdd": dd}

    print(f"\n  {label}   n={len(trades)}   median hold {np.median(held):.0f} sessions   "
          f"cost charged {cost_bp:.0f} bp round trip"
          + (f" + borrow {borrow_bp_yr:.0f} bp/yr" if borrow_bp_yr else ""))
    print(f"    {'overnight gap (FORGONE)':22} mean {gap.mean():+7.1f} bp   "
          f"median {np.median(gap):+7.1f} bp   win {100*(gap>0).mean():4.0f}%")
    out = {"unhedged": blk(net_raw, "unhedged net"), "hedged": blk(net_hed, "beta-hedged net")}
    out["n"] = len(trades)
    out["gap_mean"] = gap.mean()
    print(f"    median daily $ volume of these names over the hold: "
          f"${np.median([t.dollar_vol for t in trades]):,.0f}")
    print(f"    mean beta {np.mean([t.beta for t in trades]):.2f}")
    return out


def shape(trades: list[Trade], label: str, min_len: int = 6) -> None:
    """Where in the window the hedged return accrues — the brief's question 2, answered directly.

    ON A BALANCED PANEL, WHICH IS NOT OPTIONAL HERE. The naive version of this table lets n shrink as
    the horizon lengthens, because events have different announcement-to-effective leads. That makes the
    later rows a DIFFERENT, self-selected sample — specifically the long-lead events, which are mostly
    the annual rebalance pre-announcements — and the cumulative mean appears to climb steeply purely
    because the sample changed underneath it. The first run of this table showed +330 bp at session 10
    against +92 bp at exit on the full sample, which is arithmetically impossible for one population and
    is the signature of exactly that artifact. Restricting to trades with at least `min_len` sessions
    holds the row composition fixed, so a rising line means drift rather than reshuffling.
    """
    keep = [t for t in trades if len(t.path) >= min_len]
    if not keep:
        print(f"\n  {label}: no trade holds {min_len}+ sessions")
        return
    print(f"\n  {label} — where the hedged return accrues, BALANCED panel "
          f"(only the {len(keep)} of {len(trades)} trades held {min_len}+ sessions, so n is constant)")
    gap = np.mean([t.gap_bp for t in keep])
    print(f"    session  0  (announcement overnight gap, BEFORE entry, forgone): {gap:+7.1f} bp")
    for d in range(min_len):
        vals = np.array([t.path[d] for t in keep])
        print(f"    session {d+1:2}  cum mean {vals.mean():+7.1f} bp   median {np.median(vals):+7.1f} bp"
              f"   share positive {100*(vals>0).mean():3.0f}%")


def robust(trades: list[Trade], label: str, cost_bp: float, borrow_bp_yr: float = 0.0,
           seed: int = 0) -> None:
    """Is the mean a structural effect or a handful of outliers? Three checks that disagree with each other
    when the answer is 'outliers'.

    t-STAT AND BOOTSTRAP. A mean of +80 bp on a standard deviation of 605 bp over 127 events is a t of
    1.5, which is nowhere near significant, and the bootstrap CI shows the same thing in units a trader
    reads. Reporting a Sharpe without this is how a coin-flip becomes a strategy.

    TRIMMED MEAN. If dropping the five best and five worst events collapses the mean toward zero, the
    "edge" is a few index events that happened to coincide with something else — most often a takeover
    rumour or an earnings report inside the window — and not the passive bid at all.

    PER-YEAR. A structural mandate does not switch off. If the return concentrates in one or two years,
    whatever is being measured is not the mandate.
    """
    x = np.array([t.hedged_bp for t in trades]) - cost_bp - borrow_bp_yr * np.array(
        [t.days_held for t in trades]) / 252.0
    n = len(x)
    t_stat = x.mean() / (x.std(ddof=1) / np.sqrt(n))
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(x, n, replace=True).mean() for _ in range(10000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\n  {label} — is it real?")
    print(f"    hedged net mean {x.mean():+7.1f} bp   t-stat {t_stat:5.2f}   "
          f"bootstrap 95% CI [{lo:+.0f}, {hi:+.0f}] bp   P(mean<=0) {100*(boot<=0).mean():4.1f}%")
    k = 5
    srt = np.sort(x)
    print(f"    drop the {k} best and {k} worst events: mean {srt[k:-k].mean():+7.1f} bp "
          f"(from {x.mean():+.1f})   best {srt[-1]:+.0f}  worst {srt[0]:+.0f}")
    print(f"    the {k} best events alone account for {100*srt[-k:].sum()/max(x.sum(), 1e-9):.0f}% "
          f"of the book's total {x.sum():+.0f} bp")
    print(f"    median {np.median(x):+7.1f} bp — read this against the mean; a mean far above the median")
    print(f"    on a 50% win rate is a lottery-ticket distribution, not a harvestable premium.")
    yrs: dict[int, list[float]] = {}
    for t, v in zip(trades, x):
        yrs.setdefault(t.entry_date.year, []).append(v)
    print("    per year:  " + "  ".join(
        f"{y}:{np.mean(v):+.0f}({len(v)})" for y, v in sorted(yrs.items())))
    pos = sum(1 for v in yrs.values() if np.mean(v) > 0)
    print(f"    positive in {pos} of {len(yrs)} years")

    # SPLIT SAMPLE, WHICH IS THE ONLY THING IN THIS FILE THAT RESEMBLES AN OUT-OF-SAMPLE TEST. Nothing
    # here is fitted, so there are no parameters to overfit — but "I found a positive mean in one pooled
    # sample" is not evidence of a persistent effect. Chronological halves are the honest check: if the
    # second half looks like the first, the flow is still there; if it does not, whatever was there has
    # been arbitraged, which is what happens to a public, pre-announced, calendar-dated flow.
    order = sorted(range(n), key=lambda i: trades[i].entry_date)
    h = n // 2
    f1, f2 = x[order[:h]], x[order[h:]]
    # EVENT-DATE CLUSTERING, WHICH IS THE STATISTIC THAT DECIDES THIS WHOLE STUDY. S&P does not announce
    # one change at a time; it announces batches, and the quarterly rebalance moves four or five names on
    # a single date. Those are NOT independent observations — they share a market regime, often a sector,
    # and sometimes a single theme (2026-03-09 added LITE, COHR and VRT, three AI-infrastructure names,
    # which between them contributed more than half the additions book's entire return). Treating 127
    # names as 127 draws overstates the sample by roughly a third and overstates the t-stat accordingly.
    # Averaging within each announcement date first is the correction, and it is not a small one.
    import collections
    by: dict = collections.defaultdict(list)
    for t, v in zip(trades, x):
        by[t.entry_date].append(v)
    dv = np.array([np.mean(by[d]) for d in sorted(by)])
    m = len(dv)
    td = dv.mean() / (dv.std(ddof=1) / np.sqrt(m)) if m > 1 else float("nan")
    bd = np.array([rng.choice(dv, m, replace=True).mean() for _ in range(10000)])
    print(f"    clustered by announcement DATE: {n} names fall on only {m} dates")
    print(f"      per-date mean {dv.mean():+7.1f} bp   median {np.median(dv):+7.1f} bp   "
          f"t-stat {td:5.2f}   dates positive {100*(dv>0).mean():3.0f}%")
    print(f"      per-date bootstrap 95% CI [{np.percentile(bd, 2.5):+.0f}, "
          f"{np.percentile(bd, 97.5):+.0f}] bp   P(mean<=0) {100*(bd <= 0).mean():4.1f}%")
    print(f"    FIRST half  ({trades[order[0]].entry_date.date()} .. "
          f"{trades[order[h-1]].entry_date.date()}, n={len(f1)}): mean {f1.mean():+7.1f} bp")
    print(f"    SECOND half ({trades[order[h]].entry_date.date()} .. "
          f"{trades[order[-1]].entry_date.date()}, n={len(f2)}): mean {f2.mean():+7.1f} bp")


def gap_stats(trades: list[Trade], label: str) -> None:
    """The announcement overnight gap on its own. Not tradeable — measured because its SIZE is the finding.

    This is the number the strategy spends its whole design avoiding. If the gap is large and reliable
    while the post-entry window is not, then the market's reaction to an index announcement is complete
    before the first tradeable print, and the "3 to 5 day inclusion drift" the brief is built on is the
    residual noise left over after the real move already happened.
    """
    g = np.array([t.gap_bp for t in trades])
    n = len(g)
    t_stat = g.mean() / (g.std(ddof=1) / np.sqrt(n))
    print(f"\n  {label} — the announcement gap itself (close before -> open after; NOT tradeable)")
    print(f"    mean {g.mean():+7.1f} bp   median {np.median(g):+7.1f} bp   "
          f"win {100*(g>0).mean():3.0f}%   t-stat {t_stat:5.2f}   n={n}")
    order = sorted(range(n), key=lambda i: trades[i].entry_date)
    h = n // 2
    print(f"    first half mean {g[order[:h]].mean():+7.1f} bp   "
          f"second half mean {g[order[h:]].mean():+7.1f} bp   "
          f"(is the reaction itself decaying as the trade gets more crowded?)")
