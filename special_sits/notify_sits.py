"""
notify_sits.py — terminal report and optional webhook, with the execution steps that actually matter.

WHY THE TRANSMITTAL INSTRUCTIONS ARE PART OF THE ALERT. This trade is not executed by an order router.
It is executed by ticking a box on a broker corporate-action form before a deadline, and the two ways to
lose money here are both operational, not analytical:

  1. MISSING THE DEADLINE. Brokers set their OWN cutoff, typically one to three business days BEFORE
     the offer's stated expiration, because they have to aggregate and forward elections to the
     depositary. The expiration date on the SEC filing is not your deadline. Yours is earlier and your
     broker does not advertise it.
  2. FAILING TO ELECT ODD-LOT PRIORITY. It is a separate checkbox from the tender itself. Tendering 99
     shares without claiming the odd-lot preference gets you prorated with the institutions — the
     entire edge, silently discarded, with the trade otherwise looking correct.

So every alert carries both, and the webhook payload is deliberately plain text: this needs to be
readable on a phone at 4pm on a deadline day.
"""
from __future__ import annotations

import json
import os
import urllib.request

WEBHOOK_ENV = "SITS_WEBHOOK_URL"


def render_tender(econ, terms, hit) -> str:
    L = []
    L.append(f"  {econ.ticker or '(no ticker)'}  {econ.company}")
    L.append(f"    {econ.verdict}   {econ.kind} offer")
    if terms.kind == "dutch":
        L.append(f"    range      ${terms.price_low} - ${terms.price_high} per share")
    elif terms.price_low:
        L.append(f"    offer      ${terms.price_low} per share")
    L.append(f"    market     ${econ.market_price:,.2f}   ->  buy {econ.shares} sh "
             f"= ${econ.capital:,.0f} capital")
    L.append(f"    economics  gross {100*econ.gross_spread:+.2f}%  "
             f"fee ${econ.fee:.0f}  NET {100*econ.net_spread:+.2f}% = ${econ.net_profit:+,.2f}")
    if econ.days_left is not None:
        L.append(f"    expires    {terms.expiry}  ({econ.days_left} days)"
                 + (f"   annualised {100*econ.annualised:.0f}%" if econ.annualised else ""))
    for f in econ.risk_flags:
        L.append(f"    ! {f}")
    L.append(f"    filing     {hit.filing_url}")
    if terms.odd_lot_quote:
        q = terms.odd_lot_quote[0]
        L.append(f'    odd-lot    "{q[:300]}"')
    return "\n".join(L)


EXECUTION_STEPS = """
EXECUTION CHECKLIST — the two failure modes here are operational, not analytical
  1. Confirm the odd-lot clause YOURSELF in the linked Offer to Purchase. Check three things the
     scanner cannot judge: that eligibility is by BENEFICIAL ownership (not holder-of-record as of a
     past date), that there is no cutoff date you already missed, and that it does not require you to
     have owned the shares before the offer was announced.
  2. Buy at most 99 shares. Your TOTAL holding in that issuer must be under 100 — if you already own
     any, you are ineligible. Buy in one order; partial fills that leave you at 100 disqualify you.
  3. Let it settle (T+1). Unsettled shares cannot be tendered.
  4. Call your broker or use the voluntary corporate action screen. TICK THE ODD-LOT PRIORITY BOX.
     It is separate from the tender election. Without it you are prorated with the institutions.
  5. On a Dutch auction, elect "at the price determined by the company" (sometimes labelled "shares
     tendered at any price"). Naming a specific price risks not being filled at all.
  6. Submit at least 2-3 BUSINESS DAYS before the stated expiration. Your broker's internal cutoff is
     earlier than the SEC filing's date and is not published.

RISKS, PLAINLY
  * The offer can be amended, extended, or terminated. Conditions are listed in the filing.
  * If it is cancelled you keep the shares, now without the bid support that tightened the spread.
    Position sizing should assume that outcome, not the good one.
  * Gains are short-term and taxed as ordinary income.
  * A wide spread is the market pricing deal-break risk. It is not free money the market overlooked.
"""


def post_webhook(text: str, url: str | None = None) -> bool:
    url = url or os.environ.get(WEBHOOK_ENV)
    if not url:
        return False
    body = json.dumps({"text": text}).encode()
    req = urllib.request.Request(url, data=body,
                                headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return 200 <= r.status < 300
    except Exception as e:
        print(f"  webhook failed: {e}")
        return False
