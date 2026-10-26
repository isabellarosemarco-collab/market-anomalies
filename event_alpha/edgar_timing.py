"""
edgar_timing.py — the same gate applied to the 8-K engine: is the stock tradeable when the filing lands?

WHY THIS TEST COMES BEFORE ANY NLP. The 8-K plan is to parse a filing in milliseconds and trade the
reaction. That presumes the filing arrives while the stock is continuously tradeable. Issuers know
exactly what they are doing with release timing: material news is deliberately filed after the close or
before the open so that the market absorbs it through an auction rather than a scramble. If most 8-Ks
land outside 09:30-16:00 ET, the price discovery happens in the opening cross at a single print that
everyone gets, and a faster parser buys you nothing. That is the equity-market analogue of Kalshi
halting its macro contracts a minute before the BLS print, and it is checkable in an hour.

HOW THE TIMESTAMP IS OBTAINED WITHOUT DOWNLOADING TERABYTES. The daily index gives accession numbers but
only a FILING DATE, which is a calendar day and useless for this question. The acceptance timestamp to
the second lives in the SGML header of the dissemination .txt, on line three, as
`<ACCEPTANCE-DATETIME>YYYYMMDDHHMMSS` in US/Eastern. Those .txt files are often tens of megabytes, so
each one is fetched with an HTTP Range request for the first 600 bytes: a full survey costs a few hundred
KB instead of gigabytes. Note the quarterly bulk form.idx on S3 returns 403 from this network, but the
DAILY index under /Archives/edgar/daily-index/ serves fine — a distinction worth remembering.

FILED-AS-OF-DATE IS NOT ACCEPTANCE DATE, AND THE GAP IS THE POINT. The sample below routinely shows
acceptance at 21:46 on the 23rd for a filing dated the 24th. Anything accepted after 17:30 ET is dated
the next business day. Reasoning from the index date alone would put that filing in the wrong session
entirely.
"""
from __future__ import annotations

import collections
import re
import time
from dataclasses import dataclass, field as dcfield
from datetime import date, datetime, timedelta

import requests

UA = "BellaMarco-Research/1.0 (bellamarco.research@gmail.com)"
PAUSE = 0.15                     # under the SEC's published 10 req/s ceiling
DAILY = "https://www.sec.gov/Archives/edgar/daily-index"
ARCH = "https://www.sec.gov/Archives/"

RE_ACC = re.compile(r"<ACCEPTANCE-DATETIME>(\d{14})")
RE_ITEM = re.compile(r"^ITEM INFORMATION:\s*(.+?)\s*$", re.M)

# Historical SGML headers carry the item as PROSE, not as the "Item 2.02" number the live Atom feed uses,
# so materiality has to be keyword-matched off the description.
#
# HARD vs SOFT IS NOT A NICETY, IT IS THE DIFFERENCE BETWEEN 90% AND 8%. A first pass that counted Item
# 7.01 (Reg FD) and Item 8.01 (Other Events) as material declared 89.7% of all 8-Ks price-moving, which is
# absurd on its face: those two are deliberate catch-alls. 8.01 is where an issuer puts anything it wants
# on the record — a conference appearance, a dividend declaration, a routine legal update. 7.01 is the
# same for fair-disclosure housekeeping. Items 1.01 and 5.02 are genuinely ambiguous rather than empty: a
# "material definitive agreement" can be a transformative contract or an amended office lease, and an
# officer departure can be a fired CEO or a retiring divisional VP. None of the four can be traded on the
# item code alone — they need the document read and judged, which is exactly the part no regex does
# reliably and the part the "fast NLP extractor" in the original plan was quietly assumed to solve. Only
# HARD items are self-describing enough that the code itself is the signal.
HARD_PHRASES = {
    "results of operations": "EARNINGS",
    "bankruptcy": "bankruptcy",
    "completion of acquisition": "M&A completed",
    "non-reliance": "restatement",
    "delisting": "delisting",
    "material impairment": "impairment",
    "costs associated with exit": "restructuring",
}
SOFT_PHRASES = {
    "material definitive agreement": "material agreement",
    "regulation fd": "Reg FD",
    "other events": "other events",
    "departure of directors": "officer departure",
}


def classify(items: list[str], phrases: dict[str, str] | None = None) -> list[str]:
    """Tags carried by a filing; empty list means nothing in the requested class."""
    tags = []
    for it in items:
        low = it.lower()
        for phrase, tag in (phrases if phrases is not None else HARD_PHRASES).items():
            if phrase in low and tag not in tags:
                tags.append(tag)
    return tags

# Regular-session boundaries in ET. Pre/post windows exist but are thin and spread-heavy, which is the
# whole reason a fill there is not comparable to a fill in the continuous session.
OPEN_M, CLOSE_M = 9 * 60 + 30, 16 * 60


@dataclass
class Filing:
    form: str
    company: str
    cik: str
    filed: str
    path: str
    accepted: datetime | None = None
    items: list[str] = dcfield(default_factory=list)

    @property
    def tags(self) -> list[str]:
        return classify(self.items)

    @property
    def soft_tags(self) -> list[str]:
        return classify(self.items, SOFT_PHRASES)

    @property
    def session(self) -> str:
        if self.accepted is None:
            return "unknown"
        t = self.accepted
        if t.weekday() >= 5:
            return "weekend"
        m = t.hour * 60 + t.minute
        if m < 4 * 60:
            return "overnight"
        if m < OPEN_M:
            return "pre-market"
        if m < CLOSE_M:
            return "INTRADAY"
        if m < 20 * 60:
            return "after-hours"
        return "overnight"


class EdgarTiming:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA, "Accept-Encoding": "gzip, deflate"})

    def _get(self, url: str, tries: int = 4, **kw):
        last = None
        for attempt in range(tries):
            time.sleep(PAUSE)
            try:
                r = self.s.get(url, timeout=40, **kw)
                if r.status_code in (429, 500, 502, 503, 504) and attempt < tries - 1:
                    time.sleep((3.0 if r.status_code == 429 else 1.0) * (2 ** attempt))
                    continue
                r.raise_for_status()
                return r
            except requests.exceptions.RequestException as e:
                last = e
                if attempt == tries - 1:
                    raise
                time.sleep(2 ** attempt)
        raise last if last else RuntimeError("unreachable")

    def daily_index(self, d: date, forms: tuple[str, ...] = ("8-K",)) -> list[Filing]:
        """Parse one day's form index. Returns only the requested form types (exact match)."""
        q = (d.month - 1) // 3 + 1
        url = f"{DAILY}/{d.year}/QTR{q}/form.{d:%Y%m%d}.idx"
        try:
            txt = self._get(url).text
        except requests.exceptions.HTTPError:
            return []                      # weekends and holidays have no index
        out = []
        for line in txt.splitlines():
            # Fixed-width columns; the form type is the leading token and must match exactly so that
            # "8-K/A" and "8-K12B" do not get swept in with plain "8-K".
            if not line.strip() or line.startswith(("Form Type", "-", " ")) and not line[:12].strip():
                continue
            parts = line.split()
            if not parts or parts[0] not in forms:
                continue
            m = re.match(r"^(\S+)\s+(.+?)\s+(\d{4,10})\s+(\d{8})\s+(\S+)$", line.strip())
            if not m:
                continue
            out.append(Filing(form=m.group(1), company=m.group(2).strip(),
                              cik=m.group(3), filed=m.group(4), path=m.group(5)))
        return out

    def header(self, f: Filing) -> None:
        """Fill in acceptance time AND item information from the SGML header, in one Range request.

        2500 bytes rather than 600: the acceptance timestamp is on line three, but the ITEM INFORMATION
        lines sit below the accession block and a filer with many items pushes them past 600. 2500 covers
        every header observed and still means a 300-filing survey moves under a megabyte.
        """
        try:
            r = self._get(ARCH + f.path, headers={"Range": "bytes=0-2500"})
        except requests.exceptions.HTTPError:
            return
        m = RE_ACC.search(r.text)
        if m:
            f.accepted = datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
        f.items = RE_ITEM.findall(r.text)


def survey(days: int = 5, per_day: int = 60) -> list[Filing]:
    """Sample 8-K acceptance times across the last `days` business days."""
    e = EdgarTiming()
    out: list[Filing] = []
    d = date.today()
    checked = 0
    while checked < days and len(out) < days * per_day:
        if d.weekday() < 5:
            fl = e.daily_index(d)
            if fl:
                checked += 1
                print(f"  {d}: {len(fl)} 8-K filings in the daily index, "
                      f"sampling {min(per_day, len(fl))}")
                step = max(1, len(fl) // per_day)
                for f in fl[::step][:per_day]:
                    e.header(f)
                    out.append(f)
        d -= timedelta(days=1)
        if (date.today() - d).days > 20:
            break
    return out


def report(fl: list[Filing]) -> None:
    got = [f for f in fl if f.accepted]
    print(f"\n  8-K filings sampled: {len(fl)}   acceptance timestamp resolved: {len(got)}")
    if not got:
        return
    c = collections.Counter(f.session for f in got)
    print(f"\n  WHEN 8-Ks ARE ACTUALLY ACCEPTED (US/Eastern):")
    order = ["INTRADAY", "pre-market", "after-hours", "overnight", "weekend", "unknown"]
    for s in order:
        if c.get(s):
            bar = "#" * int(40 * c[s] / len(got))
            print(f"    {s:12} {c[s]:4} ({100*c[s]/len(got):5.1f}%)  {bar}")
    intra = 100 * c.get("INTRADAY", 0) / len(got)
    print(f"\n  share landing in the continuous session (09:30-16:00 ET): {intra:.1f}%")

    # HALF-HOUR buckets, not hourly. An hourly histogram straddles both boundaries that matter: the
    # 09:00 hour is half pre-open and half open, and the 16:00 hour is entirely after the close but
    # looks adjacent to the session. Bucketing at :00/:30 makes every bucket unambiguously one side.
    hrs = collections.Counter((f.accepted.hour, 0 if f.accepted.minute < 30 else 30) for f in got)
    print(f"\n  by half-hour (only buckets wholly inside 09:30-16:00 are the session):")
    mx = max(hrs.values())
    for (h, m) in sorted(hrs):
        start = h * 60 + m
        mark = " <- SESSION" if OPEN_M <= start and start + 30 <= CLOSE_M else ""
        print(f"    {h:02d}:{m:02d}  {hrs[(h, m)]:4}  {'#' * int(50*hrs[(h, m)]/mx)}{mark}")

    # THE CROSSTAB THAT DECIDES THE LEG. 14% of 8-Ks landing intraday is not zero, so the next question
    # is whether THOSE are the ones a price responds to. Two ways for the leg to die here: the intraday
    # slice is routine housekeeping, or the material slice deliberately avoids the session. Either way
    # the tradeable population is the intersection, not the 14%.
    hard = [f for f in got if f.tags]
    soft = [f for f in got if not f.tags and f.soft_tags]
    routine = [f for f in got if not f.tags and not f.soft_tags]
    hard_intra = [f for f in hard if f.session == "INTRADAY"]
    print(f"\n  MATERIALITY x SESSION:")
    print(f"    HARD  (item code alone is the signal)     {len(hard):4} of {len(got)} "
          f"({100*len(hard)/len(got):5.1f}%)")
    print(f"    SOFT  (needs the document read to judge)  {len(soft):4} of {len(got)} "
          f"({100*len(soft)/len(got):5.1f}%)")
    print(f"    routine housekeeping                      {len(routine):4} of {len(got)} "
          f"({100*len(routine)/len(got):5.1f}%)")
    print(f"\n    HARD filings accepted inside 09:30-16:00 ET:   {len(hard_intra)} of {len(hard)}")
    print(f"    TRADEABLE POPULATION (hard AND intraday):      {len(hard_intra)} of {len(got)} "
          f"= {100*len(hard_intra)/len(got):.1f}% of all 8-Ks, "
          f"~{len(hard_intra)/5:.1f} per day market-wide")
    if hard:
        print(f"\n    hard filings by session:")
        cm = collections.Counter(f.session for f in hard)
        for s in order:
            if cm.get(s):
                print(f"      {s:12} {cm[s]:4} ({100*cm[s]/len(hard):5.1f}%)")
    for name, phr in (("HARD", HARD_PHRASES), ("SOFT", SOFT_PHRASES)):
        tagc = collections.Counter(t for f in got for t in classify(f.items, phr))
        if not tagc:
            continue
        print(f"\n    {name} items, and how many of each land intraday:")
        for t, n in tagc.most_common():
            ni = sum(1 for f in got if t in classify(f.items, phr) and f.session == "INTRADAY")
            print(f"      {t:20} {n:4} total, {ni:3} intraday ({100*ni/n:.0f}%)")

    # The filed-date vs acceptance-date gap: proof that the index date cannot be used for sessioning.
    shifted = sum(1 for f in got if f.accepted.strftime("%Y%m%d") != f.filed)
    print(f"\n  filings whose ACCEPTANCE date differs from the index FILED date: "
          f"{shifted} of {len(got)} ({100*shifted/len(got):.1f}%)")
    print("  Those are the after-17:30 acceptances rolled to the next business day. Sessioning from")
    print("  the index date alone would place every one of them in the wrong trading session.")


if __name__ == "__main__":
    print("=" * 100)
    print("8-K ACCEPTANCE-TIME SURVEY — is the stock tradeable when the filing lands?")
    print("=" * 100)
    report(survey())
