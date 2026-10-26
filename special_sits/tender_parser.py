"""
tender_parser.py — pull the economically decisive terms out of an Offer to Purchase.

THE BUG THIS FILE WAS REWRITTEN TO KILL. The first version set a boolean `has_odd_lot` by searching for
the phrase "odd lot". The first live candidate it produced, NFJ (Virtus Dividend, Interest & Premium
Strategy Fund), contains this sentence:

    "this tender offer will not have any special pro-ration provision for odd-lot tenders, which means
     that all odd-lot tenders (including shareholders who own fewer than 100 Shares) are subject to
     pro-ration."

The filing mentions odd lots specifically in order to DENY them priority, and keyword matching scored it
as a positive. Acting on that would mean buying 99 shares expecting full acceptance, getting prorated to
single digits instead, and paying the fee for the privilege. A false negative costs nothing here. A false
positive costs real money, so the default is now AMBIGUOUS and nothing is treated as granted unless
affirmative language is found with no negation attached to it.

THREE STATES, NOT A BOOLEAN:
    granted    — affirmative odd-lot priority language, no negation nearby
    denied     — explicit statement that odd lots get no priority / remain subject to proration
    ambiguous  — the phrase appears but intent is unclear. Requires a human read. NOT tradeable.

THE OTHER THING THE FIRST VERSION GOT WRONG: PRICE. Issuer self-tenders come in three pricing regimes,
and only two of them have a price you can know when you buy:

  * FIXED. "$44.00 per share." Knowable. Spread is computable today.
  * DUTCH AUCTION. A range, $X-$Y. Every holder names a price; the company sets one clearing price and
    pays it to everyone at or below it. Name a price above the clearing price and you are not filled at
    all — you still own the stock, now without the bid support. The retail-safe election is "at the
    price determined by the company", which guarantees a fill at an unknown price. So the only
    defensible valuation is against the BOTTOM of the range, never the midpoint. Nobody is paid a
    midpoint.
  * NAV-LINKED. Overwhelmingly common for closed-end funds: "98% of net asset value per share as
    determined at the close of trading on the Expiration Date." There is NO knowable price at trade
    time — NAV moves with the portfolio until the day it is struck. A scanner that reports a spread
    against today's NAV is reporting a number that does not exist. These are flagged, not priced.

WHAT ELSE IS EXCLUDED AS A CLASS. Non-traded interval funds and private BDCs (Ares Private Markets,
Fortress Private Lending, Crescent Private Credit, Franklin Lexington) file SC TO-I for their routine
quarterly repurchase offers. They have no ticker, no secondary market, and they repurchase AT NAV, so
there is no spread to capture and no way to buy in. They are detected and dropped rather than shown as
"no price data", which wrongly implies a data problem rather than a category error.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"

# Dutch auction ranges. The connector varies more than you would expect: Arbutus writes "not less than
# $5.00 and not more than $5.75 per Share", which the original "nor greater than" pattern missed
# entirely, leaving a live odd-lot auction classified as unpriceable.
RE_RANGE = [
    re.compile(r"not\s+less\s+than\s+\$\s?([\d,]+\.?\d*)\s+(?:nor|and\s+not)\s+(?:greater|more)\s+than\s+\$\s?([\d,]+\.?\d*)", re.I),
    re.compile(r"not\s+(?:greater|more)\s+than\s+\$\s?([\d,]+\.?\d*)\s+(?:nor|and\s+not)\s+less\s+than\s+\$\s?([\d,]+\.?\d*)", re.I),
    re.compile(r"price\s+range\s+of\s+\$\s?([\d,]+\.?\d*)\s*(?:to|[-–])\s*\$\s?([\d,]+\.?\d*)", re.I),
    re.compile(r"between\s+\$\s?([\d,]+\.?\d*)\s+and\s+\$\s?([\d,]+\.?\d*)\s+per\s+(?:share|unit)", re.I),
]
RE_FIXED = [
    re.compile(r"purchase\s+price\s+of\s+\$\s?([\d,]+\.?\d*)\s+per\s+(?:share|unit)", re.I),
    re.compile(r"at\s+a\s+price\s+of\s+\$\s?([\d,]+\.?\d*)\s+per\s+(?:share|unit)", re.I),
    re.compile(r"\$\s?([\d,]+\.?\d*)\s+per\s+(?:share|unit)\s*,?\s*net\s+to\s+the\s+(?:seller|tendering)", re.I),
    re.compile(r"cash\s+purchase\s+price\s+of\s+\$\s?([\d,]+\.?\d*)", re.I),
]
# NAV-linked pricing: the defining marker is a percentage OF net asset value, struck on a future date.
RE_NAV_LINKED = [
    re.compile(r"(\d{2}(?:\.\d+)?)\s*%\s+of\s+(?:the\s+)?net\s+asset\s+value", re.I),
    re.compile(r"net\s+asset\s+value\s+per\s+(?:share|unit)[^.]{0,120}?(?:determined|calculated)\s+(?:as\s+of|on)", re.I),
    re.compile(r"price\s+equal\s+to\s+(?:the\s+)?net\s+asset\s+value", re.I),
]
# Expiration. Deliberately permissive about the word between the time and the date: real filings write
# "expire at 5:00 p.m. Eastern Time October 5, 2026" with no "on", which the first version missed.
RE_EXPIRY = [
    re.compile(rf"expir\w*[^.]{{0,120}}?((?:{MONTHS})\s+\d{{1,2}},\s+20\d\d)", re.I),
    re.compile(rf"Expiration\s+Date[^.]{{0,120}}?((?:{MONTHS})\s+\d{{1,2}},\s+20\d\d)", re.I),
    re.compile(rf"until\s+[\d:]+\s*[apm.\s]{{2,12}}[^.]{{0,80}}?((?:{MONTHS})\s+\d{{1,2}},\s+20\d\d)", re.I),
]
RE_SHARES_SOUGHT = re.compile(
    r"(?:up\s+to|purchase)\s+([\d,]{5,15})\s+(?:shares|of\s+its\s+(?:outstanding\s+)?shares)", re.I)

ODD_LOT_CUES = ("odd lot", "odd-lot", "odd lots", "odd-lots",
                "fewer than 100 shares", "less than 100 shares", "fewer than one hundred")

# Language that GRANTS priority. All of these mean "you are not prorated".
GRANT_PAT = [
    r"odd[\s-]?lots?\s+(?:will|shall)\s+(?:be\s+)?(?:accepted|purchased)[^.]{0,80}?(?:before|prior\s+to|first)",
    r"(?:first\s+)?purchase\s+all\s+(?:shares|odd[\s-]?lots?)[^.]{0,120}?fewer\s+than\s+100",
    r"odd[\s-]?lots?[^.]{0,120}?(?:not\s+be\s+subject\s+to|exempt\s+from|without)\s+(?:any\s+)?pro[\s-]?ration",
    r"fewer\s+than\s+100\s+shares[^.]{0,140}?(?:not\s+be\s+subject\s+to|exempt\s+from|without)\s+(?:any\s+)?pro[\s-]?ration",
    r"odd[\s-]?lot\s+(?:priority|preference|provision\s+will\s+apply)",
    r"preference\s+(?:will\s+be\s+)?given\s+to[^.]{0,80}?odd[\s-]?lots?",
    r"accept\s+all\s+shares\s+(?:properly\s+)?tendered\s+by[^.]{0,100}?fewer\s+than\s+100",
]
# Language that DENIES priority. Checked first — a denial overrides everything.
DENY_PAT = [
    r"(?:will|shall|does)\s+not\s+have\s+any\s+(?:special\s+)?pro[\s-]?ration\s+provision\s+for\s+odd[\s-]?lots?",
    r"no\s+(?:special\s+)?(?:pro[\s-]?ration\s+)?(?:provision|priority|preference)[^.]{0,60}?odd[\s-]?lots?",
    r"odd[\s-]?lots?[^.]{0,140}?(?:are|will\s+be|remain)\s+subject\s+to\s+pro[\s-]?ration",
    r"fewer\s+than\s+100\s+shares\)?\s*(?:are|will\s+be)\s+subject\s+to\s+pro[\s-]?ration",
    r"(?:there\s+is|without)\s+no\s+odd[\s-]?lot\s+(?:priority|preference|provision)",
    r"all\s+odd[\s-]?lot\s+tenders[^.]{0,80}?subject\s+to\s+pro[\s-]?ration",
]
# Non-traded vehicles: routine NAV repurchase programmes, not tradeable events.
NONTRADED_PAT = [
    r"there\s+is\s+no\s+(?:established\s+)?(?:public\s+)?(?:secondary\s+)?(?:trading\s+)?market",
    r"shares\s+are\s+not\s+(?:currently\s+)?listed\s+on\s+any\s+(?:national\s+)?securities\s+exchange",
    r"quarterly\s+repurchase\s+offer",
    # "interval fund" must DESCRIBE THE REGISTRANT. The bare phrase matched NFJ, a NYSE-listed CEF,
    # because its standstill clause lists things an activist agrees not to pursue: "a merger, tender
    # or exchange offer, open-ending, conversion to an exchange traded fund or interval fund". An
    # incidental mention of the structure is not a claim to be it, and that false flag hid NFJ's real
    # disqualifier (the odd-lot denial) behind a wrong one.
    r"(?:Fund|Company|Trust|registrant)\s+is\s+(?:a|an)\s+[^.]{0,90}?interval\s+fund",
    r"registered\s+(?:as\s+)?(?:a|an)\s+[^.]{0,70}?interval\s+fund",
    r"continuously[\s-]offered[^.]{0,70}?(?:non[\s-]?traded|interval)",
    r"repurchase\s+offer\s+(?:is\s+)?(?:being\s+)?made\s+pursuant\s+to\s+Rule\s+23c-3",
]
CONDITION_CUES = {
    "financing condition": (r"financing\s+condition", r"obtain\s+(?:the\s+)?financing"),
    "minimum tender condition": (r"minimum\s+tender\s+condition", r"minimum\s+number\s+of\s+shares"),
    "regulatory condition": (r"regulatory\s+approval", r"antitrust", r"hart[\s-]?scott"),
    "may be terminated": (r"terminate\s+the\s+offer", r"withdraw\s+the\s+offer",
                          r"not\s+obligated\s+to\s+(?:accept|purchase)"),
}


def _money(s: str) -> float:
    return float(s.replace(",", ""))


def _snips(text: str, cues, limit: int = 3, width: int = 460) -> list[str]:
    low = text.lower()
    out, seen = [], set()
    for cue in cues:
        for m in re.finditer(re.escape(cue), low):
            a, b = max(0, m.start() - width // 3), min(len(text), m.end() + width)
            snip = text[a:b].strip()
            if snip[:70] in seen:
                continue
            seen.add(snip[:70])
            out.append(("..." if a else "") + snip + ("..." if b < len(text) else ""))
            if len(out) >= limit:
                return out
    return out


@dataclass
class Terms:
    kind: str = "unknown"                 # fixed | dutch | nav_linked | unknown
    price_low: float | None = None
    price_high: float | None = None
    nav_pct: float | None = None          # e.g. 98.0 for "98% of net asset value"
    expiry: str | None = None
    expiry_stale: bool = False            # parsed a date already in the past
    shares_sought: int | None = None
    odd_lot: str = "ambiguous"            # granted | denied | ambiguous
    odd_lot_quote: list[str] = field(default_factory=list)
    nontraded: bool = False
    conditions: list[str] = field(default_factory=list)

    @property
    def priceable(self) -> bool:
        """Can a spread be computed at all? NAV-linked offers have no knowable price when you buy."""
        return self.kind in ("fixed", "dutch") and self.price_low is not None

    @property
    def conservative_price(self) -> float | None:
        """The only price a buyer can count on: the fixed price, or the LOW end of a Dutch range."""
        return self.price_low if self.priceable else None

    @property
    def tradeable(self) -> bool:
        return (not self.nontraded) and self.odd_lot == "granted" and self.priceable


def _classify_odd_lot(text: str) -> tuple[str, list[str]]:
    low = text.lower()
    if not any(c in low for c in ODD_LOT_CUES):
        return "absent", []
    quotes = _snips(text, ODD_LOT_CUES)
    # Denial wins: a filing that both describes the concept and disclaims it is a denial.
    for p in DENY_PAT:
        if re.search(p, low):
            return "denied", quotes
    for p in GRANT_PAT:
        if re.search(p, low):
            return "granted", quotes
    return "ambiguous", quotes


def _pick_expiry(text: str) -> tuple[str | None, bool]:
    """Latest plausible expiration date found. Amendments extend offers, so latest wins.

    Returns (date_string, stale). Stale means every date found is already past, which usually means the
    document is an old amendment rather than the live offer — it must not be reported as an open deal.
    """
    found: list[date] = []
    for rx in RE_EXPIRY:
        for m in rx.finditer(text):
            for fmt in ("%B %d, %Y",):
                try:
                    found.append(datetime.strptime(m.group(1), fmt).date())
                except ValueError:
                    pass
    if not found:
        return None, False
    today = date.today()
    future = sorted(d for d in found if d >= today)
    if future:
        return future[0].strftime("%B %d, %Y"), False       # nearest upcoming deadline
    return max(found).strftime("%B %d, %Y"), True


def parse(text: str) -> Terms:
    t = Terms()
    low = text.lower()

    t.nontraded = any(re.search(p, low) for p in NONTRADED_PAT)

    for rx in RE_RANGE:
        m = rx.search(text)
        if m:
            a, b = _money(m.group(1)), _money(m.group(2))
            t.kind, t.price_low, t.price_high = "dutch", min(a, b), max(a, b)
            break
    if t.kind == "unknown":
        for rx in RE_NAV_LINKED:
            m = rx.search(text)
            if m:
                t.kind = "nav_linked"
                try:
                    t.nav_pct = float(m.group(1))
                except (IndexError, ValueError):
                    t.nav_pct = None
                break
    if t.kind == "unknown":
        for rx in RE_FIXED:
            m = rx.search(text)
            if m:
                t.kind = "fixed"
                t.price_low = t.price_high = _money(m.group(1))
                break

    t.expiry, t.expiry_stale = _pick_expiry(text)

    m = RE_SHARES_SOUGHT.search(text)
    if m:
        try:
            n = int(m.group(1).replace(",", ""))
            t.shares_sought = n if n >= 1000 else None
        except ValueError:
            pass

    t.odd_lot, t.odd_lot_quote = _classify_odd_lot(text)
    t.conditions = [name for name, pats in CONDITION_CUES.items()
                    if any(re.search(p, low) for p in pats)]
    return t


def merge(parts: list[Terms]) -> Terms:
    """Combine Terms parsed from several documents of the same deal.

    The terms are spread across exhibits: the Offer to Purchase has the price, the Letter of Transmittal
    has the odd-lot election, a covering letter may carry the denial. Taking only the highest-scoring
    document loses whichever fact lives elsewhere. Denials always propagate; grants only fill a gap.
    """
    out = Terms()
    for p in parts:
        if out.kind == "unknown" and p.kind != "unknown":
            out.kind, out.price_low, out.price_high, out.nav_pct = p.kind, p.price_low, p.price_high, p.nav_pct
        if out.expiry is None or (out.expiry_stale and not p.expiry_stale):
            if p.expiry:
                out.expiry, out.expiry_stale = p.expiry, p.expiry_stale
        out.shares_sought = out.shares_sought or p.shares_sought
        out.nontraded = out.nontraded or p.nontraded
        if p.odd_lot == "denied":
            out.odd_lot = "denied"
        elif p.odd_lot == "granted" and out.odd_lot != "denied":
            out.odd_lot = "granted"
        elif p.odd_lot == "ambiguous" and out.odd_lot not in ("denied", "granted"):
            out.odd_lot = "ambiguous"
        for q in p.odd_lot_quote:
            if q not in out.odd_lot_quote:
                out.odd_lot_quote.append(q)
        for c in p.conditions:
            if c not in out.conditions:
                out.conditions.append(c)
    if not any(p.odd_lot != "absent" for p in parts):
        out.odd_lot = "absent"
    return out
