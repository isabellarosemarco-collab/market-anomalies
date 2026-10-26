"""
run_edgar_gate.py — leg 2 end to end: the SEC EDGAR 8-K breaking-news engine, gated before it is built.

  python -m event_alpha.run_edgar_gate

Three tests in the order that kills the idea fastest if it is dead. Same discipline that took leg 1 from
a multi-week build to a one-hour falsification.

  1. POPULATION  how many 8-Ks are both price-moving AND land while the stock trades?   (edgar_timing)
  2. LATENCY     how stale is a filing when a free poller first sees it?                (edgar_latency)
  3. REACTION    is the move still there afterwards, and can $5,000 transact in it?     (reaction)

WHAT THE ORIGINAL PLAN GOT BACKWARDS, STATED ONCE SO THE FILE TREE MAKES SENSE. The brief specified
`nlp_extractor.py` and `latency_bench.py` first: extract the outcome, then measure feed-to-execution in
milliseconds. Both presuppose that milliseconds are the binding constraint. They are not. The constraint
is measured below at roughly forty SECONDS, it sits inside the SEC's publication pipeline rather than in
any code path, and it is dwarfed in turn by the fact that the filings worth trading mostly do not arrive
while the market is open. A millisecond parser behind a forty-second queue is a forty-second-late trader,
and no amount of optimisation in `nlp_extractor.py` touches that.
"""
from __future__ import annotations

from event_alpha import edgar_latency as lat
from event_alpha import edgar_timing as tim
from event_alpha import reaction as rx


def verdict() -> None:
    print("\n" + "=" * 110)
    print("LEG 2 VERDICT")
    print("=" * 110)
    print("""
  Three measured constraints, each independently sufficient to cap the strategy below its goal:

  POPULATION   92 hard-material 8-Ks in 7 trading days; 13 of them accepted inside 09:30-16:00 ET.
               That is ~1 per day market-wide, and only ~5 of the 13 map to a listed common stock with
               usable 1-minute bars. Call it 130-250 opportunities a year BEFORE any filter on whether
               the filing is actually directional. 56.5% of hard filings go out after the close on
               purpose, which is not an accident of scheduling — it is issuers choosing to let an
               auction absorb the news rather than a scramble.

  LATENCY      Acceptance to first-visible on the public feed: median 39s, range 31-60s (n=14). The feed
               snapshot itself is only ~1s old when it arrives, so the delay is inside the SEC's
               pipeline, not in the poller. Polling faster re-downloads an identical document. The
               original premise was a 3-SECOND head start over retail; the real figure is forty seconds
               in the wrong direction, and the comparison is worse than that, because an issuer pushes
               its press release to Business Wire at the same instant it files. The wire reaches
               Bloomberg and every broker news panel with no queue at all. EDGAR is the slow mirror.

  REACTION     Volume in the acceptance minute is 0.8x a normal minute — below average. The market does
               not respond to these filings, so there is no reaction to be early to. The names that do
               file hard 8-Ks intraday have ~$2.9M average daily dollar volume and ~$46k in the minute
               you would be trading, making a $5,000 order 11% of that minute's flow. The subsequent
               moves are large in percentage terms (-153bp to +80bp at one minute) but their SIGN is
               unrelated to the item code, so they are volatility rather than edge.

  The honest limit of this evidence: n=5 on the reaction test is far too small to estimate an expected
  edge, and it is not being used to. The population and latency figures are what carry the verdict, and
  those rest on 1,254 filings and 14 timed arrivals respectively. The reaction table is there to show
  that the surviving candidates are illiquid microcaps and royalty trusts, which it does clearly.

  WHAT WOULD CHANGE THE ANSWER, since none of the above is about skill. The 40-second lag disappears on
  a paid direct feed and the population constraint disappears if you trade the OPEN rather than the
  filing: 88% of hard 8-Ks land outside the session, so the tradeable expression is an overnight or
  opening-auction position, not a millisecond race. That is a different strategy with a different edge —
  slower, capacity-constrained by nothing, and testable on daily data going back years instead of seven
  days. It is also the one this codebase is already set up to study.
""")


def main() -> int:
    print("=" * 110)
    print("SEC EDGAR 8-K ENGINE — gated before built")
    print("=" * 110)
    print("all endpoints public and unauthenticated; declared User-Agent throughout; no credentials\n")

    print("=" * 110)
    print("TEST 1 — POPULATION: are price-moving 8-Ks even filed while the market is open?")
    print("=" * 110)
    tim.report(tim.survey())

    print("\n" + "=" * 110)
    print("TEST 2 — LATENCY: how stale is a filing when you first see it?")
    print("=" * 110)
    lat.report(lat.probe(form="", minutes=5.0, every=3.0))

    print("\n" + "=" * 110)
    print("TEST 3 — REACTION: is the move still there, and can $5,000 transact in it?")
    print("=" * 110)
    rx.study()

    verdict()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
