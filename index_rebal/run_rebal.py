"""
run_rebal.py — the index reconstitution study end to end.

  python -m index_rebal.run_rebal

COST ASSUMPTIONS, STATED UP FRONT BECAUSE THEY DECIDE THE VERDICT AND NOTHING ELSE DOES.

  ADDITIONS, 12 bp round trip. Incoming S&P 500 names are large caps; quoted spreads are 1-3 bp. But the
  trade is specified at the OPEN on the morning after the announcement, which is the single widest,
  most volatile print of that name's year — an opening auction absorbing an overnight reprice. 5 bp of
  slippage per side is generous to the strategy, not punitive, and 1 bp per side covers the exit auction
  and commissions. Retail commission is zero, so none is charged.

  DELETIONS, 25 bp round trip plus 300 bp/yr borrow. Outgoing names are small, gapping, and heavily
  shorted in exactly the window being studied, which is when borrow is scarcest and most expensive. 300
  bp/yr on a ~5-session hold is only ~6 bp, so borrow is NOT what kills a deletion trade — the spread is.
  That is worth knowing because it is the opposite of the usual assumption about shorting.

Both figures are held fixed across variants so the comparison between them is about the trade and not
about the cost model.
"""
from __future__ import annotations

import pathlib
import pickle

from index_rebal import backtest as bt
from index_rebal.changes import RE_ACQUIRED, coverage, fetch

ADD_COST, DEL_COST, DEL_BORROW = 12.0, 25.0, 300.0


def main() -> int:
    print("=" * 108)
    print("S&P 500 INDEX RECONSTITUTION DRIFT — does the forced passive bid pay a retail account?")
    print("=" * 108)
    print("public data only: Wikipedia wikitext for the change history, yfinance for daily bars\n")

    ch = fetch()
    coverage(ch)

    usable = [c for c in ch if c.announced and c.lead_days is not None
              and bt.MIN_LEAD <= c.lead_days <= bt.MAX_LEAD and c.effective.year >= 2018]
    print(f"\n  after the lead-plausibility filter ({bt.MIN_LEAD}-{bt.MAX_LEAD} calendar days): "
          f"{len(usable)} events")
    drop = [c for c in ch if c.announced and c.effective.year >= 2018 and c not in usable]
    print(f"  dropped for an implausible announcement->effective lead: {len(drop)} "
          f"(bad Wikipedia citations pointing at the wrong press release, not real index events)")

    # Prefetch once: both sides and both exit conventions reuse the same bar cache, and hitting Yahoo
    # four times for the same tickers is how you get rate-limited into a half-empty sample.
    tick = sorted({c.add_ticker for c in usable if c.add_ticker} |
                  {c.del_ticker for c in usable if c.del_ticker and c.deletion_is_mktcap})
    cache = pathlib.Path("/tmp/rebal_px.pkl")
    if cache.exists():
        px = pickle.loads(cache.read_bytes())
        print(f"\n  daily bars from cache ({cache})")
    else:
        print(f"\n  fetching daily bars for {len(tick)} tickers + {bt.BENCH} ...")
        px = bt._px(tick + [bt.BENCH], "2016-06-01", "2026-09-26")
        cache.write_bytes(pickle.dumps(px))
    print(f"  bars available for {len([t for t in tick if t in px])} of {len(tick)}")
    print("  The shortfall is delisted tickers. It biases the sample toward survivors, which flatters")
    print("  the deletion leg in particular — a deleted name that later went to zero is simply absent.")

    print("\n" + "=" * 108)
    print("BOOK 1 — LONG ADDITIONS")
    print("=" * 108)
    add_auc, s1 = bt.build(usable, "add", exit_offset=-1, px=px)
    add_eff, _ = bt.build(usable, "add", exit_offset=0, px=px)
    print(f"  candidates {s1['candidates']}, tradeable {s1['trades']}")
    print(f"  excluded as recycled symbols (vendor splices two companies into one series): "
          f"{sorted(bt.RECYCLED)}")
    if s1.get("relisted"):
        print(f"  dropped for a relisting gap inside the beta window: {sorted(set(s1['relisted']))}")
    a1 = bt.summarise(add_auc, "exit: close of last session BEFORE effective (the indexers' auction)",
                      cost_bp=ADD_COST)
    a2 = bt.summarise(add_eff, "exit: close ON the effective date (the brief's spelling, one day late)",
                      cost_bp=ADD_COST)
    bt.shape(add_auc, "additions")
    bt.robust(add_auc, "additions, auction exit", ADD_COST)
    bt.gap_stats(add_auc, "additions")

    print("\n" + "=" * 108)
    print("BOOK 2 — SHORT DELETIONS (market-cap removals only)")
    print("=" * 108)
    mk = [c for c in usable if c.deletion_is_mktcap]
    acq = [c for c in usable if RE_ACQUIRED.search(c.reason)]
    print(f"  market-cap removals: {len(mk)}   acquisition-driven (EXCLUDED): {len(acq)}")
    del_auc, s2 = bt.build(usable, "del", exit_offset=-1, px=px)
    print(f"  candidates {s2['candidates']}, tradeable {s2['trades']}")
    d1 = bt.summarise(del_auc, "exit: close of last session BEFORE effective",
                      cost_bp=DEL_COST, borrow_bp_yr=DEL_BORROW)
    if del_auc:
        bt.shape(del_auc, "deletions")
        bt.robust(del_auc, "deletions, auction exit", DEL_COST, DEL_BORROW)
        bt.gap_stats(del_auc, "deletions")

    print("\n" + "=" * 108)
    print("VERDICT")
    print("=" * 108)
    for nm, r, cost in (("LONG ADDITIONS (auction exit)", a1, ADD_COST),
                        ("LONG ADDITIONS (effective-date exit)", a2, ADD_COST),
                        ("SHORT DELETIONS (auction exit)", d1, DEL_COST)):
        if not r:
            continue
        h = r["hedged"]
        print(f"  {nm:38} n={r['n']:3}  hedged net {h['mean']:+6.1f} bp/trade  "
              f"win {100*h['win']:3.0f}%  Sharpe {h['sharpe']:5.2f}")
        print(f"  {'':38} gross before the {cost:.0f} bp cost: {h['mean']+cost:+6.1f} bp")
    print("""
  THE HEADLINE MEANS ARE POSITIVE AND THEY DO NOT SURVIVE BEING LOOKED AT. Read the four numbers that
  matter in this order, because each one removes a chunk of the apparent edge.

  1. THE MEDIAN. Additions: mean +76 bp, MEDIAN -11 bp, win rate 49%. Most of these trades lose money and
     the average is positive only because of a right tail. That is not a premium being harvested, it is a
     lottery-ticket distribution. Deletions are better behaved (median +94 bp on a 54% win rate) but not
     by enough to change the conclusion.

  2. CONCENTRATION. The five best additions account for 91% of that book's entire return: LITE +3023,
     APP +1626, COHR +1532, ENPH +1420, MRNA +1219. Those are the AI-optics, adtech, solar and vaccine
     momentum stories of their respective years. The deletion book is worse: its five best events are 99%
     of its total. A strategy whose result is five names is a bet on those five names.

  3. CLUSTERING, WHICH IS THE ONE MOST STUDIES OF THIS GET WRONG. S&P announces in batches, so 126
     additions fall on only 86 distinct dates and 67 deletions on only 35. Names announced together share
     a regime, a sector and often a single theme — 2026-03-09 added LITE, COHR and VRT, three
     AI-infrastructure names, which alone are more than half the additions book. Correcting for this
     drops the additions t-stat to 1.26 and the deletions t-stat to 1.04, and it moves the per-date
     MEDIANS to +3.7 bp and -2.8 bp with only 50% and 49% of dates positive. Read that against the
     per-name deletion median of +94 bp: once the batch is the unit of observation, the typical
     reconstitution date makes nothing. On 35 independent dates the deletion book's P(mean<=0) is 14.9%.

  4. THE SPLIT SAMPLE GOES THE WRONG WAY TO BE AN EDGE. Additions go from -14 bp in 2018-2022 to +172 bp
     in 2023-2026, and the announcement gap goes from +50 bp to +457 bp over the same split. A structural
     mandate does not get four times stronger; a sample that has filled up with high-beta momentum names
     does. The improvement is a change in WHAT gets added to the index, not in how the flow behaves.

  WHAT IS ACTUALLY REAL HERE, AND WHY YOU CANNOT HAVE IT. The one statistically overwhelming result in
  this entire study is the announcement overnight gap on additions: +257 bp mean, +131 bp median, 67% win
  rate, t-stat 6.88 on 126 events. The market reprices an index addition essentially completely between
  the close and the next open. By the time the first tradeable print exists, the move is over, and what
  remains for the next three to five sessions is the noise measured above. The brief's question 2 asked
  whether the alpha is in the overnight gap or in the multi-day drift. It is in the gap, entirely, and the
  strategy as specified is defined to begin after it.

  THE ASYMMETRY IS THE TELL. Deletions gap only +28 bp with a t-stat of 0.99 and a 49% win rate — no
  reaction at all — while additions gap +257 bp. If this were forced passive flow, both sides would move:
  the funds must sell the deletion exactly as hard as they must buy the addition. Only the glamour side
  moves, which says the addition gap is largely attention and momentum rather than mechanical demand.

  COSTS WERE NEVER THE BINDING CONSTRAINT, WHICH ANSWERS QUESTION 4 IN THE OPPOSITE DIRECTION TO THE ONE
  IT ASSUMED. Additions have a median $766M of daily volume and deletions $124M, so a $5,000 order is
  invisible in both and slippage is a rounding error. Borrow on a five-session short at 300 bp/yr is ~6
  bp against a 25 bp spread cost. Mid-cap borrow and slippage are not what stops this trade; the absence
  of a reliable post-announcement drift is.

  THE ONE HONEST CAVEAT IN THE STRATEGY'S FAVOUR. Delisted tickers are missing from the sample, which
  biases the deletion book toward names that survived. Fixing it would need a paid point-in-time
  security master. It would make the deletion result better, not worse — a deleted name that later went
  to zero would have been a profitable short. That is worth knowing, but a survivorship correction
  cannot manufacture significance out of 35 independent observations.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
