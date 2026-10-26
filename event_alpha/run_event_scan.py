"""
run_event_scan.py — does prediction-market latency arbitrage exist at retail size?

  python -m event_alpha.run_event_scan

Runs the three tests in the order that kills the idea fastest if it is dead:

  1. THE GATE      is the contract open when its news lands?          (tradeability.py)
  2. CAPACITY      how many dollars sit within 5c of the touch?       (kalshi.measure_depth)
  3. VELOCITY      what does the exchange-wide tape actually look like?

Order matters. Benchmarking feed latency first — which is what the brief asked for — measures the one
variable that cannot matter until test 1 passes. This is the same inversion that made the odd-lot
scanner report a candidate it could never have traded: the exciting number got computed before the
disqualifying fact got checked.
"""
from __future__ import annotations

import collections
import datetime as dt

import pandas as pd

from event_alpha.kalshi import Kalshi, measure_depth


def gate(k) -> pd.DataFrame:
    from event_alpha import tradeability as tr
    print("=" * 112)
    print("TEST 1 — THE GATE: is the market open at the moment of its own release?")
    print("=" * 112)
    v = tr.audit_all(k)
    df = pd.DataFrame([x.row() for x in v])
    print(df.to_string(index=False))
    live = [x for x in v if x.tradeable_through_event]
    print(f"\n  series with ANY tradeable window through the release: {len(live)} of {len(v)}")
    if not live:
        print("  Every scheduled macro contract halts before its own print. There is no latency race to")
        print("  win here — the order cannot be submitted at all. Scheduled-macro fast-feed arb is dead")
        print("  for structural reasons, not competitive ones, and no amount of speed changes it.")
    return df


def capacity(k, n: int = 14) -> pd.DataFrame:
    from event_alpha import tradeability as tr
    print("\n" + "=" * 112)
    print("TEST 2 — CAPACITY: unscheduled-event markets DO stay open. How much money fits?")
    print("=" * 112)
    cands = tr.unscheduled_candidates(k)
    print(f"  unscheduled headline-driven markets found: {len(cands)}")
    rows = []
    for c in cands[:n]:
        try:
            yes, no = k.orderbook(c["ticker"])
        except Exception:
            continue
        dy = measure_depth(yes, c["ticker"], "yes")
        dn = measure_depth(no, c["ticker"], "no")
        rows.append({"event": c["event"], "days_out": c["days_out"],
                     "yes_touch": dy.touch, "yes_near$": round(dy.near),
                     "no_touch": dn.touch, "no_near$": round(dn.near),
                     "junk$": round(dy.junk + dn.junk)})
    if not rows:
        print("  no books retrievable.")
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("yes_near$", ascending=False)
    print(df.to_string(index=False))
    tot = df["yes_near$"].sum() + df["no_near$"].sum()
    print(f"\n  TOTAL real depth within 5c of touch across all {len(df)} markets: ${tot:,.0f}")
    print(f"  TOTAL penny-level junk resting in the same books:                  ${df['junk$'].sum():,.0f}")
    print("  The junk column is why these books look deep and are not. A 1-cent bid is a lottery")
    print("  ticket, not liquidity you can sell size into when the ruling drops.")
    return df


def velocity(k) -> None:
    print("\n" + "=" * 112)
    print("TEST 3 — VELOCITY: what is actually being traded on this exchange?")
    print("=" * 112)
    t = k.trades(limit=1000)
    if not t:
        print("  no tape.")
        return
    ts = sorted(x["created_time"] for x in t)
    a = dt.datetime.fromisoformat(ts[0].replace("Z", "+00:00"))
    b = dt.datetime.fromisoformat(ts[-1].replace("Z", "+00:00"))
    sec = max((b - a).total_seconds(), 1e-6)
    # Size field is `count_fp`, not `count`. Reading `count` raises KeyError on every row.
    notional = sum(float(x["count_fp"]) * float(x["yes_price_dollars"]) for x in t)
    print(f"  last {len(t)} trades spanned {sec:.1f}s  ->  {len(t)/sec:.0f} trades/sec exchange-wide")
    print(f"  notional across all of them: ${notional:,.0f}  (mean ${notional/len(t):.2f} per trade)")
    c = collections.Counter(x["ticker"].split("-")[0] for x in t)
    print(f"\n  where the volume actually is (series prefix, share of trades):")
    for pre, cnt in c.most_common(8):
        print(f"    {pre:28} {100*cnt/len(t):5.1f}%")
    print("\n  Read the mean trade size before anything else. This is a venue of ~$20 tickets, and the")
    print("  flow concentrates in 15-minute crypto price markets — high-frequency scalping, not news.")


def edgar_note() -> None:
    print("\n" + "=" * 112)
    print("LEG 2 — SEC EDGAR 8-K ENGINE: survives THIS gate, fails its own")
    print("=" * 112)
    print("  Equities have no pre-close mechanism, so unlike Kalshi macro there IS a window by")
    print("  construction. That is why leg 2 was worth measuring. It was measured; see")
    print("  `run_edgar_gate.py` for the full run. Summary of what came back:")
    print("    POPULATION  92 hard-material 8-Ks per 7 trading days, only 13 accepted inside")
    print("                09:30-16:00 ET. ~1/day market-wide; 56.5% go out after the close by design.")
    print("    LATENCY     acceptance -> first visible on the public feed is median 39s (31-60s, n=14),")
    print("                and the feed snapshot is 1s old on arrival, so the delay is the SEC's")
    print("                pipeline. The brief assumed a 3-second EDGE; it is 40 seconds of DEFICIT,")
    print("                and the issuer's own wire release beats EDGAR to Bloomberg regardless.")
    print("    REACTION    acceptance-minute volume is 0.8x normal — no response to be early to — in")
    print("                names with ~$2.9M ADV where $5,000 is 11% of the trading minute.")
    print("  Both legs of the brief died at the same place, and it was not speed: the venue structurally")
    print("  withholds the window (leg 1) or the population that has a window is too small and too")
    print("  illiquid to matter (leg 2). Neither failure is fixable with faster code, which is precisely")
    print("  why measuring the gate before building the pipeline was the whole point.")


def main() -> int:
    k = Kalshi()
    print("KALSHI / PREDICTION-MARKET LATENCY ARBITRAGE — falsification run")
    print("all endpoints public and unauthenticated; no API key exists in this codebase\n")
    gate(k)
    capacity(k)
    velocity(k)
    edgar_note()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
