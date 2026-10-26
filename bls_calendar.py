"""
bls_calendar.py — real CPI and Employment Situation release dates, 2015-present, with release TIMES.

WHY THIS FILE EXISTS INSTEAD OF A CALENDAR RULE. The brief describes CPI as "mid-month
Wednesdays/Thursdays" and NFP as "first Friday of the month". Both are approximations, and for an
event study the approximation is the whole ballgame: a date that is wrong by one day moves a real
event night into the control group AND puts an ordinary night into the event group, which biases the
test toward the null twice over. NFP is the worse offender -- the Employment Situation is released on
the third Friday following the conclusion of the reference week, which is usually the first Friday of
the month but lands on the SECOND Friday several times a decade. CPI has no rule at all; it is
whatever day BLS picks, usually between the 10th and the 15th.

HOW THE DATES ARE OBTAINED, GIVEN THAT bls.gov BLOCKS PROGRAMMATIC ACCESS. bls.gov returns 403 to any
request without a browser fingerprint (Akamai, not robots.txt -- a spoofed User-Agent does not help).
The side door is that BLS's own per-release schedule pages are archived by the Wayback Machine, which
serves them freely: each snapshot of

    bls.gov/schedule/news_release/cpi.htm      (and .../empsit.htm)

carries a table of [Reference Month, Release Date, Release Time] covering roughly fourteen months
forward from the snapshot date. Every year from 2008 to 2026 has multiple snapshots, so taking a few
per year and unioning them reconstructs the full schedule from the authoritative source rather than
from a rule or from memory. This is BLS's own published page; the archive is only the transport.

WHAT THE RELEASE TIME IS FOR. It is not decoration. The study depends on whether a release lands
before the equity open (08:30 ET, so inside the overnight window) or during the session, and a
handful of releases have moved -- the autumn 2025 shutdown pushed several CPI prints off schedule.
Any release not at 08:30 is reported rather than silently pooled.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

UA = {"User-Agent": "BellaMarco-Research/1.0 (bellamarco.research@gmail.com)"}
CACHE = Path("data_cache")
CDX = ("https://web.archive.org/cdx/search/cdx?url=bls.gov/schedule/news_release/{page}"
       "&output=json&fl=timestamp&filter=statuscode:200&collapse=timestamp:6&limit=4000")
SNAP = "https://web.archive.org/web/{ts}id_/https://www.bls.gov/schedule/news_release/{page}"
PAGES = {"CPI": "cpi.htm", "NFP": "empsit.htm"}
# BLS writes "Sept." on some pages and "Sep." on others; pandas accepts neither consistently.
MONTH = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
ROW = re.compile(r"^([A-Z][a-z]+\.?)\s+(\d{1,2}),\s*((?:19|20)\d\d)$")


def _parse_date(txt: str) -> pd.Timestamp | None:
    m = ROW.match(txt.strip().replace("Sept.", "Sep."))
    if not m:
        return None
    mon = MONTH.get(m.group(1)[:3].lower())
    if not mon:
        return None
    try:
        return pd.Timestamp(year=int(m.group(3)), month=mon, day=int(m.group(2)))
    except ValueError:
        return None


def _get(url: str, tries: int = 3) -> str:
    """Fetch with a short, bounded backoff (3s, 10s, 30s).

    The archive rate-limits after a few hundred requests. An unbounded backoff turns that into an
    apparent hang with no output, so the retry budget here is deliberately small and the caller is
    expected to be re-runnable instead: every snapshot is cached individually by
    `_snapshot_rows`, so running the scrape again picks up exactly where it left off.
    """
    import time
    last = None
    for a in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=60)
            if r.status_code == 200:
                return r.text
            last = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last = type(e).__name__
        time.sleep((3, 10, 30)[min(a, 2)])
    raise RuntimeError(f"{last} after {tries} tries: {url}")


def _snapshots(page: str, years: range, per_year: int = 3) -> list[str]:
    """Pick up to `per_year` archived snapshots per calendar year.

    Each page covers ~14 months forward, so two well-spread snapshots per year would suffice; three
    is cheap insurance against a snapshot that happens to have captured a partially-rendered page.
    """
    # The snapshot index is cached too: it is one request per page, but it is the request that
    # aborts the whole scrape when the archive throttles, and it never changes materially.
    idx_cache = CACHE / f"bls_cdx_{page.replace('.', '_')}.json"
    if idx_cache.exists():
        rows = json.loads(idx_cache.read_text())
    else:
        rows = json.loads(_get(CDX.format(page=page)))[1:]
        idx_cache.write_text(json.dumps(rows))
    by_year: dict[str, list[str]] = {}
    for (ts,) in rows:
        by_year.setdefault(ts[:4], []).append(ts)
    out: list[str] = []
    for y in years:
        got = by_year.get(str(y), [])
        if not got:
            continue
        step = max(1, len(got) // per_year)
        out += got[::step][:per_year]
    return out


SNAP_STORE = CACHE / "bls_snapshots.json"


def _snapshot_rows(page: str, ts: str, store: dict, verbose: bool) -> dict[str, list]:
    """Parsed [ref_month -> (date, time)] for one archived snapshot, cached on disk.

    Caching per snapshot rather than per scrape is what makes this tractable against a rate-limiting
    archive: a run that gets throttled halfway still keeps everything it fetched, so re-running
    converges on full coverage instead of starting over.
    """
    key = f"{page}|{ts}"
    if key in store:
        return store[key]
    try:
        import time
        time.sleep(1.0)      # ~130 requests total; pacing them is what keeps the archive answering
        html = _get(SNAP.format(ts=ts, page=page))
    except RuntimeError as e:
        if verbose:
            print(f"      skip {key}: {e}")
        return {}
    soup = BeautifulSoup(re.sub(r"<!--.*?-->", "", html, flags=re.S), "html.parser")
    rows: dict[str, list] = {}
    for tr in soup.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        # The real rows are exactly [reference month, release date, release time]. The page also
        # contains one giant cell holding the whole layout; requiring 3 cells discards it.
        if len(cells) != 3:
            continue
        d = _parse_date(cells[1])
        ref = cells[0].strip()
        if d is None or not re.match(r"^[A-Z][a-z]+\.?\s+(19|20)\d\d$", ref):
            continue
        rows[ref] = [d.strftime("%Y-%m-%d"), cells[2].strip()]
    store[key] = rows
    SNAP_STORE.write_text(json.dumps(store))
    return rows


def _scrape_page(page: str, years: range, per_year: int = 6, verbose: bool = True) -> dict[str, list]:
    """Map each REFERENCE MONTH to the release date given by the LATEST snapshot that mentions it.

    Keying on reference month rather than unioning release dates is what makes a rescheduled release
    come out right. BLS publishes a schedule months ahead and then amends it: the autumn 2025
    shutdown moved several CPI prints, and a naive union of every date ever printed on the page keeps
    BOTH the original date and the replacement. One of those two is a day on which nothing happened,
    and putting it in the event group is exactly the misdating this file exists to avoid. Since each
    reference month has exactly one real release, taking the most recently archived answer per
    reference month resolves every amendment in favour of the final schedule.
    """
    store = json.loads(SNAP_STORE.read_text()) if SNAP_STORE.exists() else {}
    snaps = _snapshots(page, years, per_year)
    if verbose:
        have = sum(1 for ts in snaps if f"{page}|{ts}" in store)
        print(f"    {page}: {len(snaps)} snapshots selected, {have} already cached")
    found: dict[str, list] = {}
    for ts in snaps:
        for ref, (d, tm) in _snapshot_rows(page, ts, store, verbose).items():
            prior = found.get(ref)
            if prior is None or ts > prior[2]:
                found[ref] = [d, tm, ts]
    return found


def release_dates(start: str = "2015-01-01", end: str = "2026-12-31",
                  refresh: bool = False) -> dict[str, pd.DataFrame]:
    """{"CPI": df, "NFP": df} with columns date, time, off_schedule. Cached to data_cache/."""
    CACHE.mkdir(exist_ok=True)
    cache = CACHE / "bls_release_dates.json"
    if cache.exists() and not refresh:
        raw = json.loads(cache.read_text())
    else:
        years = range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1)
        raw = {k: _scrape_page(p, years) for k, p in PAGES.items()}
        cache.write_text(json.dumps(raw, indent=1, sort_keys=True))

    out = {}
    for k, byref in raw.items():
        # Tolerate both cache shapes: the current one keys on reference month and stores
        # [date, time, snapshot], an older one keyed on date and stored just the time string.
        if byref and isinstance(next(iter(byref.values())), str):
            byref = {d: [d, t, ""] for d, t in byref.items()}
        rows = [(v[0], v[1], ref) for ref, v in byref.items()]
        df = pd.DataFrame(rows, columns=["date", "time", "ref_month"])
        df["date"] = pd.to_datetime(df.date)
        df = (df[(df.date >= start) & (df.date <= end)]
              .sort_values("date").drop_duplicates("date").reset_index(drop=True))
        # 08:30 AM is the standard slot and the only one that sits inside the overnight window.
        df["off_schedule"] = ~df.time.str.replace(" ", "").str.upper().str.startswith("08:30")
        out[k] = df
    return out


def summary(rel: dict[str, pd.DataFrame]) -> None:
    print("  BLS RELEASE DATES, scraped from BLS's own schedule pages via the Wayback Machine:")
    for k, df in rel.items():
        per_year = df.groupby(df.date.dt.year).size()
        gaps = per_year[per_year != 12]
        print(f"    {k:4} n={len(df):4}  {df.date.min().date()} .. {df.date.max().date()}  "
              f"off-08:30 slots: {int(df.off_schedule.sum())}")
        if len(gaps):
            print(f"         years without 12 releases: {dict(gaps)}")
        if df.off_schedule.any():
            for _, r in df[df.off_schedule].iterrows():
                print(f"         {r.date.date()} at {r.time}")
    # The weekday distribution is the cheap check that the scrape found real dates and not noise.
    for k, df in rel.items():
        wd = df.date.dt.day_name().value_counts()
        print(f"    {k:4} weekday mix: {dict(wd)}")
    nfp = rel["NFP"].date
    first_fri = nfp.dt.day <= 7
    print(f"    NFP on the FIRST Friday: {int(first_fri.sum())} of {len(nfp)} "
          f"({100 * first_fri.mean():.0f}%) — the brief's rule would misdate the other "
          f"{int((~first_fri).sum())}.")
