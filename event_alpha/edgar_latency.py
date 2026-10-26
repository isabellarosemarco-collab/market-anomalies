"""
edgar_latency.py — THE DISSEMINATION LAG. How stale is a filing by the time a free poller can see it?

WHY THIS IS THE ONLY LATENCY NUMBER THAT MATTERS, AND WHY NO AMOUNT OF CODE SHRINKS IT. The 8-K thesis
is "parse the filing in milliseconds and trade before humans react". That silently assumes the clock
starts when the filing is ACCEPTED. It does not. It starts when the filing becomes VISIBLE to you, and
the gap between those two instants is set by the SEC's publication pipeline, not by your poller. A 1ms
parser sitting behind a 10-minute feed delay is a 10-minute-late trader.

Measured directly: poll the public "latest filings" Atom feed, and for each accession record
    lag = (wall clock when I FIRST saw it) - (its acceptance timestamp)
Acceptance time comes from the feed's own per-entry <updated> element, so both halves are the SEC's own
numbers and the measurement does not depend on my clock being aligned with theirs beyond wall time.

THAT <updated> == ACCEPTANCE IS AN ASSUMPTION, SO IT WAS CHECKED RATHER THAN ASSERTED. The entire lag
number collapses if the feed's per-entry <updated> is a PUBLICATION time instead of an acceptance time:
the measurement would then be wall-clock minus publication, which is near zero by construction and would
report a 40-second pipeline as instantaneous. Verified against the authoritative source by pulling six
consecutive feed entries and comparing each one's <updated> to the <ACCEPTANCE-DATETIME> in its own SGML
header. All six matched to the second (e.g. feed 2026-09-25T10:28:46 vs sgml 20260925102846). The
timestamps are the same quantity, so the lag below is genuine acceptance-to-visible.

A SECOND, INDEPENDENT STALENESS TERM WORTH SEPARATING. The feed document carries its own <updated>
header — the moment the SEC built the page. If that header is already minutes behind wall clock when the
response arrives, the response is a cached snapshot, and polling faster only re-fetches the same stale
snapshot. So this script reports two deltas: feed-build staleness, and total acceptance-to-visible lag.
Only the second is the trading constraint, but the first tells you whether faster polling could help at
all. If the feed is cached, the answer is no, and the fix is a paid direct feed rather than more code.

Rate discipline: the SEC publishes a 10 req/s ceiling and enforces it with IP bans applied to the whole
household, not the script. One request every few seconds is far under it and still fine-grained enough,
because the quantity being measured turns out to be minutes.
"""
from __future__ import annotations

import collections
import html
import re
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import requests

UA = "BellaMarco-Research/1.0 (bellamarco.research@gmail.com)"
FEED = ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type={form}"
        "&company=&dateb=&owner=include&count=100&output=atom")

RE_ENTRY = re.compile(r"<entry>(.*?)</entry>", re.S)
RE_ACCNO = re.compile(r"accession-number=([\d-]+)")
RE_UPD = re.compile(r"<updated>([^<]+)</updated>")
RE_TITLE = re.compile(r"<title>([^<]+)</title>")
RE_ITEM = re.compile(r"Item ([\d.]+):")

# The 8-K items that actually move a price. Everything else is governance housekeeping that files on a
# schedule and trades on nothing.
MATERIAL = {
    "1.03": "bankruptcy",
    "2.02": "EARNINGS",
    "2.01": "M&A completed",
    "1.01": "material agreement",
    "4.02": "restatement / non-reliance",
    "3.01": "delisting notice",
    "5.02": "officer departure",
    "7.01": "Reg FD",
    "8.01": "other events",
}


@dataclass
class Seen:
    accno: str
    title: str
    accepted: datetime
    first_seen: float
    items: list[str] = field(default_factory=list)

    @property
    def lag_s(self) -> float:
        return self.first_seen - self.accepted.timestamp()

    @property
    def material(self) -> bool:
        return any(i in MATERIAL for i in self.items)


def probe(form: str = "8-K", minutes: float = 6.0, every: float = 4.0) -> list[Seen]:
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept-Encoding": "gzip, deflate"})
    url = FEED.format(form=form)
    seen: dict[str, Seen] = {}
    build_stale: list[float] = []
    t_end = time.time() + minutes * 60

    while time.time() < t_end:
        try:
            r = s.get(url, timeout=30)
            now = time.time()
            r.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"  ! {e}")
            time.sleep(every * 2)
            continue

        body = r.text
        # Feed-build staleness: the header <updated> is the FIRST one in the document, before any entry.
        head = RE_UPD.search(body.split("<entry>")[0])
        if head:
            built = datetime.fromisoformat(head.group(1))
            build_stale.append(now - built.timestamp())

        fresh = 0
        for block in RE_ENTRY.findall(body):
            a = RE_ACCNO.search(block)
            u = RE_UPD.search(block)
            if not a or not u:
                continue
            if a.group(1) in seen:
                continue
            t = RE_TITLE.search(block)
            seen[a.group(1)] = Seen(
                accno=a.group(1),
                title=html.unescape(t.group(1)) if t else "",
                accepted=datetime.fromisoformat(u.group(1)),
                first_seen=now,
                items=RE_ITEM.findall(html.unescape(block)),
            )
            fresh += 1
        print(f"  poll @ {datetime.now().strftime('%H:%M:%S')}  "
              f"feed built {build_stale[-1]:5.0f}s ago  new accessions: {fresh:3d}  "
              f"tracked: {len(seen)}")
        time.sleep(every)

    if build_stale:
        print(f"\n  FEED-BUILD STALENESS: median {statistics.median(build_stale):.0f}s, "
              f"min {min(build_stale):.0f}s, max {max(build_stale):.0f}s")
        print("  That is how old the SEC's snapshot already is when it reaches you. Polling faster than")
        print("  this interval re-downloads the identical snapshot and buys exactly zero milliseconds.")
    return list(seen.values())


def report(rows: list[Seen]) -> None:
    # Only filings FIRST SEEN after the probe started have a meaningful lag; the initial poll returns a
    # backlog whose lag is an artifact of when the script happened to launch. Drop the first batch.
    if not rows:
        print("  nothing observed.")
        return
    t0 = min(r.first_seen for r in rows)
    live = [r for r in rows if r.first_seen > t0 + 1]
    print(f"\n  accessions observed: {len(rows)}   appeared DURING the probe (usable): {len(live)}")
    if not live:
        print("\n  No new filing appeared during the window, which is itself the answer at this hour:")
        print("  the arrival rate is low enough that there is no continuous stream to race.")
        return

    lags = sorted(r.lag_s for r in live)
    print(f"\n  ACCEPTANCE -> FIRST VISIBLE (seconds):")
    print(f"    min {lags[0]:7.0f}   median {statistics.median(lags):7.0f}   "
          f"mean {statistics.mean(lags):7.0f}   max {lags[-1]:7.0f}")
    print(f"    in minutes:  median {statistics.median(lags)/60:.1f} min, worst {lags[-1]/60:.1f} min")

    mat = [r for r in live if r.material]
    print(f"\n  of those, carrying a price-moving item: {len(mat)} of {len(live)}")
    for r in sorted(mat, key=lambda x: x.lag_s)[:12]:
        tags = ",".join(MATERIAL[i] for i in r.items if i in MATERIAL)
        print(f"    {r.lag_s/60:6.1f} min  {r.title[:56]:56}  {tags}")

    c = collections.Counter(i for r in live for i in r.items)
    print(f"\n  item mix across everything observed:")
    for i, n in c.most_common(10):
        lbl = MATERIAL.get(i, "routine")
        print(f"    Item {i:5} {n:4}  {lbl}")


if __name__ == "__main__":
    import sys

    # DEFAULT TO ALL FORMS, NOT 8-K, AND THE REASON IS A MEASUREMENT FAILURE WORTH RECORDING. A first run
    # filtered to type=8-K and observed ZERO arrivals in six minutes at 10:16 ET — not because the feed is
    # slow but because intraday 8-Ks arrive roughly once every fifteen minutes market-wide. With no
    # arrivals there is no lag to measure, and the probe reports nothing while looking like it ran fine.
    # The dissemination pipeline is form-agnostic: a 4 is published by the same machinery as an 8-K. So
    # measure the lag on the whole firehose, where arrivals are continuous, and apply the result to 8-Ks.
    form = sys.argv[1] if len(sys.argv) > 1 else ""
    mins = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    print("=" * 100)
    print("EDGAR DISSEMINATION LAG — acceptance timestamp vs first moment a free poller can see it")
    print(f"form filter: {form or 'ALL FORMS (the firehose; the pipeline does not care which form)'}")
    print("=" * 100)
    report(probe(form=form, minutes=mins, every=3.0))
