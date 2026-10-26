"""
run_daily_scan.py — CLI orchestrator for the capacity-constrained special-situations book.

  python -m special_sits.run_daily_scan --days 120              # odd-lot tenders, last 120 days
  python -m special_sits.run_daily_scan --cef                   # CEF discount + activist screen
  python -m special_sits.run_daily_scan --days 365 --backtest    # how often does this even fire?

THE --backtest FLAG IS THE MOST IMPORTANT ONE. Every previous strategy in this project died on trade
FREQUENCY, not on edge size: the FOMC rule netted a real +14.7bp but fired 8 times a year, capping it
near 1.2%/yr. The same arithmetic governs this book. A 3% net return per tender is worthless if three
qualifying tenders exist per year, and wonderful if thirty do. So the scan reports the DEAL COUNT and
the implied annual return before it reports any individual opportunity, because that number decides
whether the whole idea is worth your attention.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from special_sits import cef_discount_mon, notify_sits, spread_calc, tender_parser  # noqa: E402
from special_sits.edgar_scanner import Edgar, group_by_deal                          # noqa: E402


def scan_tenders(days: int, fee: float, capital: float, quiet: bool) -> pd.DataFrame:
    e = Edgar()
    print(f"searching EDGAR full-text: SC TO-I filings mentioning 'odd lot', last {days} days ...")
    hits = e.search("odd lot", forms="SC TO-I", days=days)
    deals = group_by_deal(hits)
    print(f"  {len(hits)} matching documents across {len(deals)} distinct issuers\n")

    rows, details = [], []
    for (cik, ticker), group in deals.items():
        # Read up to 6 documents per deal and MERGE. The terms are split across exhibits: price in the
        # Offer to Purchase, odd-lot election in the Letter of Transmittal, and sometimes the denial in
        # a covering letter to brokers. Reading only the top-scoring document loses whichever fact
        # lives elsewhere — which is how the NFJ odd-lot denial was nearly missed.
        # 6 rather than 4, because relevance score does not track importance: for Arbutus the 268KB
        # Offer to Purchase — the only document carrying the Dutch auction price range — ranked 5th of
        # 8, behind three short transmittal exhibits. FTS scores phrase density and a long document
        # dilutes it, so the authoritative document is routinely not the top-scoring one.
        group = sorted(group, key=lambda h: (-h.score, h.file_date))
        parts, best = [], None
        for h in group[:6]:
            try:
                parts.append(tender_parser.parse(e.fetch_text(h)))
                best = best or h
            except Exception as ex:
                print(f"  ! fetch failed {h.company[:30]}: {ex}")
        if not parts:
            continue
        terms = tender_parser.merge(parts)
        best = best or group[0]

        # LIVE vs HISTORICAL pricing. An offer whose expiry has passed (or never parsed) on a filing
        # more than a fortnight old is a closed deal, and must be priced against the quote ON ITS
        # FILING DATE, not today's. Pricing WIX's $80-$92 range against a quote six months later
        # manufactured the only "candidate" the year-long scan produced; contemporaneously the stock
        # was $92.94, above the whole range, and the trade was a guaranteed loss.
        days_left = spread_calc._days_to(terms.expiry)
        historical = (days_left is None or days_left < 0) and \
                     (date.today() - date.fromisoformat(str(best.file_date))).days > 14
        asof = str(best.file_date) if historical else None
        px = spread_calc.quote(ticker, asof=asof) if ticker and ticker != "?" else None
        econ = spread_calc.evaluate(ticker or "?", best.company, px, terms,
                                    fee=fee, max_capital=capital) if px else None
        if econ is not None and historical:
            econ.risk_flags.insert(0, f"HISTORICAL: priced as of filing date {asof}, offer closed")
            if econ.verdict == "CANDIDATE":
                econ.verdict = "BACKTEST hit"
        if econ is None:
            note = ("non-traded interval fund / private BDC — no secondary market"
                    if terms.nontraded else "no listed ticker or quote unavailable")
            verdict = "EXCLUDE not traded" if terms.nontraded else "EXCLUDE no quote"
            rows.append({"ticker": ticker or "-", "company": best.company[:28],
                         "type": terms.kind, "odd_lot": terms.odd_lot, "mkt": None,
                         "tender_lo": terms.price_low, "tender_hi": terms.price_high,
                         "sh": None, "capital": None, "gross%": None, "NET%": None,
                         "net_$": None, "days": None, "ann%": None,
                         "verdict": verdict, "flags": note})
            continue
        r = econ.row()
        r["odd_lot"] = terms.odd_lot
        rows.append(r)
        details.append((econ, terms, best))

    df = pd.DataFrame(rows)
    if df.empty:
        print("no SC TO-I filings mentioning an odd-lot provision in this window.")
        return df

    order = {"CANDIDATE": 0, "BACKTEST hit": 1, "CAUTION wide spread": 2, "REVIEW odd-lot unclear": 2,
             "REVIEW NAV-linked": 3, "REVIEW manually": 4, "REVIEW parse suspect": 5}
    df["_o"] = df.verdict.map(lambda v: order.get(v, 6))
    df = df.sort_values(["_o", "NET%"], ascending=[True, False]).drop(columns="_o")
    cols = ["ticker", "company", "type", "odd_lot", "mkt", "tender_lo", "tender_hi", "sh",
            "capital", "gross%", "NET%", "net_$", "days", "ann%", "verdict", "flags"]
    df = df[[c for c in cols if c in df.columns]]

    print("=" * 150)
    print(f"ODD-LOT TENDER OFFERS — {len(df)} issuers, 99-share cap, ${fee:.0f} broker fee, "
          f"${capital:,.0f} budget")
    print("=" * 150)
    print(df.drop(columns=["flags"]).to_string(index=False, na_rep="-"))
    print("\nwhy each was excluded:")
    for _, r in df.iterrows():
        # r["flags"], never r.flags: `flags` is a reserved pandas attribute, so attribute access
        # returns the Series' own Flags object and prints "<Flags(allows_duplicate_labels=True)>"
        # instead of the reason. That silently blanked every exclusion reason in the table.
        if r["verdict"] != "CANDIDATE":
            print(f"  {str(r['ticker']):8} {r['verdict']:24} {r['flags']}")

    live = [d for d in details if d[0].verdict in ("CANDIDATE", "CAUTION wide spread")]
    if live and not quiet:
        print("\n" + "=" * 128)
        print("ACTIONABLE DETAIL")
        print("=" * 128)
        for econ, terms, hit in live:
            print(notify_sits.render_tender(econ, terms, hit))
            print()
        print(notify_sits.EXECUTION_STEPS)

    # Count live candidates and historical hits SEPARATELY, and label which is which. Lumping them
    # together is exactly how a look-ahead artifact gets reported as a live opportunity.
    live_mask = df.verdict == "CANDIDATE"
    hist_mask = df.verdict == "BACKTEST hit"
    hit_mask = live_mask | hist_mask
    n_ok, n_live, n_hist = int(hit_mask.sum()), int(live_mask.sum()), int(hist_mask.sum())
    per_yr = n_ok * 365 / days
    med = df.loc[hit_mask, "NET%"].median() if n_ok else 0.0
    print("=" * 128)
    print("FREQUENCY — the constraint that killed every previous strategy in this project")
    print("=" * 128)
    print(f"  qualifying deals in {days}d: {n_ok}   implied rate: {per_yr:.1f}/yr")
    print(f"    of which live now: {n_live}   historical (priced as of filing date): {n_hist}")
    if n_ok:
        print(f"  median net return per deal: {med:.2f}%")
        print(f"  if you caught EVERY one, sequentially, on ${capital:,.0f}: "
              f"{per_yr * med:.1f}%/yr  (${capital * per_yr * med / 100:,.0f})")
        print(f"  note: deals overlap in time, so capital can often work on several at once, which is")
        print(f"        the one way this book scales without needing more capital per position.")
    else:
        print("  nothing qualified. Widen --days or relax the filters, but do not relax them to get a")
        print("  hit — an empty screen is a real answer about this window.")
    df.to_csv("special_sits_tenders.csv", index=False)
    print("\nsaved -> special_sits_tenders.csv")
    return df


def scan_cefs(min_discount: float, max_z: float) -> pd.DataFrame:
    print(f"screening closed-end funds: discount <= {min_discount}%, "
          f"Z-score <= {max_z} OR hard catalyst ...")
    funds, stats = cef_discount_mon.screen(min_discount=min_discount, max_z=max_z)
    print(f"  universe {stats['universe']} funds   "
          f"activist 13D targets found on EDGAR: {stats['activist_13d_targets']}   "
          f"passed: {stats['passed']}\n")
    if not funds:
        print("nothing passed.")
        return pd.DataFrame()
    df = pd.DataFrame([f.row() for f in funds])
    print("=" * 128)
    print("CEF DISCOUNTS WITH A CATALYST  (z1y very negative = unusually wide FOR THIS FUND)")
    print("=" * 128)
    print(df.to_string(index=False, na_rep="-"))
    print("\n  'to_NAV%' is the return IF the discount closed fully. It usually does not. A tender at")
    print("  98% of NAV or an open-ending captures most of it; a board concession captures some; a")
    print("  losing proxy fight captures none. Treat it as the ceiling, not the estimate.")
    with_act = df[df.activist != "-"]
    print(f"\n  funds with a named activist on the register: {len(with_act)}"
          + (f" -> {', '.join(with_act.ticker)}" if len(with_act) else ""))
    df.to_csv("special_sits_cefs.csv", index=False)
    print("\nsaved -> special_sits_cefs.csv")
    return df


def main() -> int:
    p = argparse.ArgumentParser(description="Capacity-constrained special situations scanner")
    p.add_argument("--days", type=int, default=120, help="EDGAR lookback window")
    p.add_argument("--fee", type=float, default=spread_calc.DEFAULT_FEE,
                   help="broker voluntary corporate action fee")
    p.add_argument("--capital", type=float, default=5000.0, help="capital budget per position")
    p.add_argument("--cef", action="store_true", help="run the CEF discount screen instead")
    p.add_argument("--both", action="store_true", help="run both scans")
    p.add_argument("--min-discount", type=float, default=-12.0)
    p.add_argument("--max-z", type=float, default=-1.0)
    p.add_argument("--quiet", action="store_true", help="table only, no execution detail")
    p.add_argument("--webhook", default=None, help="override SITS_WEBHOOK_URL")
    a = p.parse_args()

    if a.cef:
        scan_cefs(a.min_discount, a.max_z)
        return 0
    df = scan_tenders(a.days, a.fee, a.capital, a.quiet)
    if a.both:
        print("\n")
        scan_cefs(a.min_discount, a.max_z)
    if a.webhook or notify_sits.WEBHOOK_ENV in __import__("os").environ:
        n = int((df.verdict == "CANDIDATE").sum()) if not df.empty else 0
        if n:
            notify_sits.post_webhook(f"{n} odd-lot tender candidate(s) open. "
                                     f"See special_sits_tenders.csv", a.webhook)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
