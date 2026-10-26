"""
changes.py — the S&P 500 addition/deletion history, WITH announcement dates rather than just effective dates.

THE DATA PROBLEM THAT DEFINES THIS LEG, AND WHY THE OBVIOUS SOURCE IS THE WRONG ONE. The strategy is
specified on the announcement-to-effective window: buy the open after the announcement, sell into the
effective-date closing cross. So the backtest needs BOTH dates. Every convenient source gives only the
effective date. The Wikipedia changes table's own column header says "Effective Date" in as many words,
and the constituent list's "Date added" is the same quantity. Building the study on that column would
silently test a completely different trade — one that enters days AFTER the announcement it is supposed
to be front-running — and it would look fine while doing it.

WHERE THE ANNOUNCEMENT DATE ACTUALLY LIVES. Each row of that table cites the S&P Dow Jones press release
that announced the change, and those URLs are date-stamped by S&P in the path:

    https://www.spglobal.com/spdji/en/documents/indexnews/announcements/20260204-1482028/1482028_day5dvax6.pdf
                                                                       ^^^^^^^^
    -> announced 2026-02-04, effective 2026-02-09

That 8-digit prefix is S&P's own publication date for the announcement document, which is the cleanest
machine-readable announcement date available without scraping 400 PDFs. The `|date=` field inside the
citation is a secondary source for the same thing, used as a fallback; where both exist they agree to
within a day, and the URL prefix is preferred because it is S&P's stamp rather than an editor's.

TWO THINGS ABOUT THIS SOURCE THAT ARE NOT OPTIONAL TO KNOW.

1. THE TABLE MOVED. It was split out of "List of S&P 500 companies" on 2026-08-11 into "Historical
   components of the S&P 500". The old article's wikitext no longer contains it at all, so code written
   against the old title silently parses zero changes rather than failing.

2. THE ANNOUNCEMENT IS AFTER THE CLOSE, USUALLY. S&P publishes index announcements after the US market
   close. Entering at the open of the NEXT session is therefore both the tradeable choice and free of
   look-ahead. If a given release actually went out in the morning, entering the following open merely
   misses part of the move — an error in the conservative direction, which is the right direction for an
   error to run in.

SURVIVORSHIP: the table records both sides of every change ("AAA replaces BBB"), including companies that
were added and later removed, so it is not a list of current members and does not suffer the survivorship
bias that the constituent table's "Date added" column would introduce.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime

import requests

UA = "BellaMarco-Research/1.0 (bellamarco.research@gmail.com)"
SOURCE_TITLE = "Historical components of the S&P 500"
RAW = "https://en.wikipedia.org/w/index.php"

RE_ANN_URL = re.compile(r"spglobal\.com/spdji/[^\s|}\]]*?/announcements/(\d{8})-")
# Second URL shape: newer rows cite the press-release archive instead of the PDF, and stamp the date
# differently — press.spglobal.com/2026-08-13-Reddit-Set-to-Join... . Same quantity, different spelling.
RE_ANN_URL2 = re.compile(r"press\.spglobal\.com/(\d{4})-(\d{2})-(\d{2})-")
RE_CITE_DATE = re.compile(r"\|\s*date\s*=\s*([^|}\n]+)")
RE_ACCESS_DATE = re.compile(r"\|\s*access-date\s*=\s*([^|}\n]+)")
RE_REF_NAMED = re.compile(r'<ref\s+name\s*=\s*"?([^">/]+)"?\s*/>')
RE_REF_DEF = re.compile(r'<ref\s+name\s*=\s*"?([^">/]+)"?\s*>(.*?)</ref>', re.S)
RE_REF_ANY = re.compile(r"<ref[^>]*>(.*?)</ref>", re.S)
RE_LINK = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")

# Removal reasons. "Market capitalization change" is a genuine index-flow deletion: the company keeps
# trading and passive funds must sell it. An acquisition-driven removal is NOT index flow in any
# tradeable sense — the stock is pinned to a deal price and about to stop existing. Conflating the two is
# the single biggest way to fabricate alpha in a deletion study, so the reason is parsed and kept.
RE_ACQUIRED = re.compile(r"acquir|merg|bought|take[- ]?over|purchase|bankrupt|spun off|"
                         r"split|delist|privat|combin", re.I)
RE_MKTCAP = re.compile(r"market cap", re.I)


def _norm_date(s: str) -> date | None:
    s = s.strip().strip("[]")
    for fmt in ("%B %d, %Y", "%Y-%m-%d", "%d %B %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


@dataclass
class Change:
    effective: date
    announced: date | None
    add_ticker: str | None
    add_name: str
    del_ticker: str | None
    del_name: str
    reason: str
    ann_source: str          # "url" | "cite-date" | "access-date" | "" — provenance matters, keep it

    @property
    def lead_days(self) -> int | None:
        if self.announced is None:
            return None
        return (self.effective - self.announced).days

    @property
    def deletion_is_mktcap(self) -> bool:
        """True only for genuine index-flow deletions (company keeps trading)."""
        return bool(RE_MKTCAP.search(self.reason)) and not RE_ACQUIRED.search(self.reason)


def _wikitext(title: str) -> str:
    r = requests.get(RAW, params={"title": title, "action": "raw"},
                     headers={"User-Agent": UA}, timeout=60)
    r.raise_for_status()
    return r.text


def fetch(title: str = SOURCE_TITLE) -> list[Change]:
    w = _wikitext(title)
    if 'id="changes"' not in w:
        raise RuntimeError(f"no changes table in '{title}' — the article was restructured again; "
                           f"find its new home before trusting an empty result")

    # Named refs are defined once and reused ("<ref name=':4' />"), so resolve the map first or every
    # reuse loses its announcement date and the sample quietly halves.
    refs = {n: body for n, body in RE_REF_DEF.findall(w)}

    table = w.split('id="changes"', 1)[1]
    table = table.split("\n|}", 1)[0]

    out: list[Change] = []
    for block in table.split("\n|-")[1:]:
        # Rows appear in two styles in this table: inline (|a || b || c) and one-cell-per-line. Normalise.
        body = block.strip()
        if not body or body.startswith("!"):
            continue
        cells = []
        for line in body.split("\n"):
            line = line.strip()
            if not line.startswith("|"):
                continue
            # Cells are prefixed with either "|" or "||" depending on which editor touched the row last,
            # and are sometimes inline-separated by "||" as well. lstrip both, then split, or the leading
            # pipe ends up inside the date string and every row fails to parse as a date — which is
            # exactly how this returned 0 changes on the first run instead of raising.
            cells.extend(p.strip() for p in line.lstrip("|").split("||"))
        if len(cells) < 6:
            continue
        eff = _norm_date(re.sub(r"<[^>]+>|\{\{[^}]*\}\}", "", cells[0]))
        if eff is None:
            continue

        reason_raw = cells[5]
        # Search the WHOLE row for citations, not just the reason cell: the table gained a separate
        # "Refs" column, so refs now live in cells[6] on recent rows and inside cells[5] on older ones.
        # Scanning the row is indifferent to which, and to the next column the editors add.
        ref_text = " ".join(RE_REF_ANY.findall(body))
        for n in RE_REF_NAMED.findall(body):
            ref_text += " " + refs.get(n, "")

        ann, src = None, ""
        m = RE_ANN_URL.search(ref_text)
        if m:
            ann, src = _norm_date(f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:]}"), "url"
        if ann is None:
            m2 = RE_ANN_URL2.search(ref_text)
            if m2:
                ann, src = _norm_date("-".join(m2.groups())), "press-url"
        if ann is None:
            for rx, tag in ((RE_CITE_DATE, "cite-date"), (RE_ACCESS_DATE, "access-date")):
                mm = rx.search(ref_text)
                if mm:
                    ann = _norm_date(mm.group(1))
                    if ann:
                        src = tag
                        break

        def tick(c: str) -> str | None:
            c = re.sub(r"<[^>]+>", "", c).strip()
            c = RE_LINK.sub(r"\1", c).strip()
            return c if re.fullmatch(r"[A-Z][A-Z.\-]{0,6}", c) else None

        def nm(c: str) -> str:
            c = re.sub(r"<[^>]+>", "", c).strip()
            return RE_LINK.sub(r"\1", c).strip()

        out.append(Change(
            effective=eff, announced=ann,
            add_ticker=tick(cells[1]), add_name=nm(cells[2]),
            del_ticker=tick(cells[3]), del_name=nm(cells[4]),
            reason=re.sub(r"<ref.*?</ref>|<ref[^>]*/>", "", reason_raw, flags=re.S).strip(),
            ann_source=src))
    return out


def coverage(ch: list[Change], since: int = 2018) -> None:
    import collections
    sel = [c for c in ch if c.effective.year >= since]
    print(f"  changes parsed: {len(ch)} total, {len(sel)} with effective date >= {since}")
    print(f"  date range: {min(c.effective for c in ch)} .. {max(c.effective for c in ch)}")
    withann = [c for c in sel if c.announced]
    print(f"  announcement date resolved: {len(withann)} of {len(sel)} "
          f"({100*len(withann)/max(len(sel),1):.0f}%)")
    print(f"  provenance: {dict(collections.Counter(c.ann_source for c in sel))}")
    leads = sorted(c.lead_days for c in withann if c.lead_days is not None)
    if leads:
        import statistics
        print(f"  announcement -> effective LEAD in calendar days: "
              f"median {statistics.median(leads):.0f}, "
              f"p10 {leads[int(.1*len(leads))]}, p90 {leads[int(.9*len(leads))]}, "
              f"min {leads[0]}, max {leads[-1]}")
        bad = [l for l in leads if l < 0]
        if bad:
            print(f"  !! {len(bad)} rows have announcement AFTER effective date — bad citations, dropped")
    byyr = collections.Counter(c.effective.year for c in sel)
    print(f"  changes per year: " + ", ".join(f"{y}:{n}" for y, n in sorted(byyr.items())))
    radd = collections.Counter("mktcap" if c.deletion_is_mktcap else
                               ("acquired" if RE_ACQUIRED.search(c.reason) else "other")
                               for c in sel)
    print(f"  removal reason mix: {dict(radd)}")
    print("  Only 'mktcap' removals are tradeable deletions; an acquired name is pinned to a deal price")
    print("  and about to stop trading, which is merger arb wearing an index-flow costume.")


if __name__ == "__main__":
    print("=" * 100)
    print("S&P 500 INDEX CHANGES — announcement and effective dates")
    print("=" * 100)
    ch = fetch()
    coverage(ch)
