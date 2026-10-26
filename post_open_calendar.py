"""
post_open_calendar.py — release dates for the events that share the FOMC decision's structural DNA.

THE STRUCTURAL RULE THIS FILE SERVES. The brief's trade buys at 15:59 and sells at 09:31. An event
qualifies as a pre-announcement drift candidate only if its release happens AFTER 09:31, so that the
holding window carries the pre-event uncertainty and none of the reaction. That makes the release CLOCK
a hard inclusion criterion, not a footnote, and it is the reason each date list below is paired with a
time rather than just a date.

WHERE EACH LIST COMES FROM, AND WHY NOT FROM A RULE.
  * FOMC MINUTES (14:00 ET). Scraped from federalreserve.gov, which prints the actual release date as
    "(Released February 18, 2026)" next to each meeting's minutes link. The rule "three weeks after the
    decision" is close but not exact -- it slips for holidays -- and 2020 has ten minutes releases
    rather than eight because the emergency March meetings produced their own. Using the Fed's own
    published release date avoids inventing either problem.
  * ISM MANUFACTURING (10:00) and ISM SERVICES (10:00). These genuinely are a rule: the first and third
    BUSINESS day of the month. ISM does not publish a free machine-readable archive, so the rule is
    used -- but business day is computed against the US FEDERAL holiday calendar, not the trading
    calendar, because those differ (Good Friday is a market holiday and not a federal one) and an
    off-by-one here would misdate the event exactly the way the NFP first-Friday rule did. The rule is
    then validated against intraday bars in the study itself rather than trusted.
  * HUMPHREY-HAWKINS TESTIMONY (10:00). Scraped from the Fed's per-year testimony listings, matching
    "Semiannual Monetary Policy Report to the Congress". Only the FIRST of the two days is kept: the
    prepared text is released that morning and the second chamber hears the same statement, so the
    news is on day one.
  * TREASURY QUARTERLY REFUNDING. Not scraped from Treasury's refunding page, which is
    JavaScript-rendered and returns no dates to a plain request. Taken instead from TreasuryDirect's
    auction API, using the fact that the QRA *is* the announcement of the quarterly 3y/10y/30y
    refunding package: the announcementDate of each new 30-Year Bond issue is the refunding statement
    date. That yields 48 dates, exactly four a year, every one a Wednesday -- which is the check that
    the identification is right.

    AND IT IS THE ONE SIBLING THAT FAILS THE STRUCTURAL TEST. The Quarterly Refunding Statement is
    released at 08:30 ET, before the 09:30 open, so the 15:59 -> 09:31 window HOLDS THROUGH IT. It has
    the same defect as CPI and NFP and it is carried here only as a negative control, never pooled.
    What does qualify is the companion release two days earlier: Treasury publishes its marketable
    borrowing estimates on the Monday of refunding week at 15:00 ET, which is after the 09:31 exit, so
    the eve-of-Monday trade is clean. Both are returned, labelled and kept apart.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import requests

UA = {"User-Agent": "BellaMarco-Research/1.0 (bellamarco.research@gmail.com)"}
CACHE = Path("data_cache")
FED = "https://www.federalreserve.gov"
RELEASED = re.compile(r"\(Released\s+([A-Z][a-z]+)\s+(\d{1,2}),\s*(\d{4})\)")
# The testimony listing puts the date in a <time> cell immediately before the event's link text.
TESTIMONY = re.compile(
    r"<time>(\d{1,2}/\d{1,2}/\d{4})</time>.{0,600}?Semiannual Monetary Policy Report", re.S)
TD_API = ("https://www.treasurydirect.gov/TA_WS/securities/search?format=json&type=Bond"
          "&dateFieldName=issueDate&startDate={a}&endDate={b}&pagesize=800")

# Release times, ET. These are the inclusion criterion, so they are stated once, here.
TIMES = {"MINUTES": "14:00", "ISM_MFG": "10:00", "ISM_SVC": "10:00",
         "HUMPHREY": "10:00", "QRA_ESTIMATES": "15:00", "QRA_STATEMENT": "08:30"}
EXIT = "09:31"          # the brief's exit; anything released before this is inside the window


def _get(url: str, tries: int = 3) -> str:
    import time
    last = None
    for a in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=60)
            if r.status_code == 200:
                return r.text
            last = f"HTTP {r.status_code}"
            if r.status_code == 404:
                break              # a page that does not exist will not exist on the third try either
        except requests.RequestException as e:
            last = type(e).__name__
        time.sleep((2, 6, 15)[min(a, 2)])
    raise RuntimeError(f"{last} after {tries} tries: {url}")


def minutes_dates(years=range(2015, 2027), refresh: bool = False) -> pd.DatetimeIndex:
    """Actual FOMC minutes release dates, from the Fed's own "(Released ...)" annotations.

    The current calendar page carries every meeting from 2021 forward; earlier years live on
    fomchistorical{y}.htm. Both are parsed and unioned, so a year appearing on both pages contributes
    the same dates twice and the set collapses it.
    """
    CACHE.mkdir(exist_ok=True)
    cache = CACHE / "fomc_minutes_dates.json"
    if cache.exists() and not refresh:
        return pd.DatetimeIndex(json.loads(cache.read_text()))
    found: set[str] = set()
    pages = [f"{FED}/monetarypolicy/fomccalendars.htm"]
    pages += [f"{FED}/monetarypolicy/fomchistorical{y}.htm" for y in years]
    for u in pages:
        try:
            html = _get(u)
        except RuntimeError:
            continue                     # a missing historical page is expected for recent years
        for mon, day, yr in RELEASED.findall(html):
            try:
                found.add(pd.Timestamp(f"{mon} {day}, {yr}").strftime("%Y-%m-%d"))
            except ValueError:
                continue
    out = sorted(found)
    cache.write_text(json.dumps(out))
    return pd.DatetimeIndex(out)


def humphrey_dates(years=range(2015, 2027), refresh: bool = False) -> pd.DatetimeIndex:
    """First-day dates of the Chair's Semiannual Monetary Policy Report testimony."""
    CACHE.mkdir(exist_ok=True)
    cache = CACHE / "humphrey_hawkins_dates.json"
    if cache.exists() and not refresh:
        return pd.DatetimeIndex(json.loads(cache.read_text()))
    found: set[str] = set()
    for y in years:
        # The Fed renamed these listings partway through the sample: 2017 onward lives under
        # /newsevents/testimony/{y}-testimony.htm, 2016 and earlier at /newsevents/{y}testimony.htm.
        # All three known patterns are tried for every year rather than hardcoding the changeover.
        # Note that a missing year returns 200 with a "page not found" body, so absence shows up as
        # zero regex matches rather than as an HTTP error -- which the union below handles silently.
        for u in (f"{FED}/newsevents/testimony/{y}-testimony.htm",
                  f"{FED}/newsevents/{y}testimony.htm",
                  f"{FED}/newsevents/{y}-testimony.htm"):
            try:
                html = _get(u)
            except RuntimeError:
                continue
            for d in TESTIMONY.findall(html):
                found.add(pd.Timestamp(d).strftime("%Y-%m-%d"))
    out = sorted(found)
    cache.write_text(json.dumps(out))
    return pd.DatetimeIndex(out)


def qra_dates(start="2015-01-01", end="2026-12-31", refresh: bool = False) -> pd.DatetimeIndex:
    """Quarterly refunding statement dates = announcement dates of each new 30-Year Bond issue.

    Reopenings are labelled "29-Year 11-Month" and similar, so filtering securityTerm to exactly
    "30-Year" isolates the four quarterly refunding packages a year. The returned dates are all
    Wednesdays, which is the sanity check that this identification is the right one.
    """
    CACHE.mkdir(exist_ok=True)
    cache = CACHE / "qra_dates.json"
    if cache.exists() and not refresh:
        out = json.loads(cache.read_text())
    else:
        d = json.loads(_get(TD_API.format(a=start, b=end)))
        out = sorted({x["announcementDate"][:10] for x in d
                      if x.get("securityTerm") == "30-Year"})
        cache.write_text(json.dumps(out))
    idx = pd.DatetimeIndex(out)
    return idx[(idx >= start) & (idx <= end)]


def _business_days(start: str, end: str) -> pd.DatetimeIndex:
    """Business days on the US FEDERAL calendar, which is what ISM schedules against."""
    from pandas.tseries.holiday import USFederalHolidayCalendar
    from pandas.tseries.offsets import CustomBusinessDay
    return pd.date_range(start, end, freq=CustomBusinessDay(calendar=USFederalHolidayCalendar()))


def ism_dates(start="2015-01-01", end="2026-12-31") -> dict[str, pd.DatetimeIndex]:
    """{"ISM_MFG": 1st business day of each month, "ISM_SVC": 3rd business day}.

    This is the one calendar here generated from a rule rather than read from the issuer, because ISM
    publishes no free archive. The rule itself is stable and well documented; the risk is the
    definition of "business day", which is why the federal calendar is used explicitly above rather
    than the trading calendar. The study validates these dates against intraday bars before using them.
    """
    bd = _business_days(start, end)
    g = pd.Series(bd, index=bd).groupby([bd.year, bd.month])
    mfg = g.nth(0)
    svc = g.nth(2)
    return {"ISM_MFG": pd.DatetimeIndex(pd.Series(mfg).values),
            "ISM_SVC": pd.DatetimeIndex(pd.Series(svc).values)}


def all_events(start="2015-01-01", end="2026-12-31",
               refresh: bool = False) -> dict[str, pd.DatetimeIndex]:
    """Every candidate event, keyed by name. Times live in TIMES; the study enforces the clock."""
    ev = dict(ism_dates(start, end))
    ev["MINUTES"] = minutes_dates(refresh=refresh)
    ev["HUMPHREY"] = humphrey_dates(refresh=refresh)
    q = qra_dates(start, end, refresh=refresh)
    ev["QRA_STATEMENT"] = q
    # Treasury's borrowing estimates come out at 15:00 on the Monday of refunding week, i.e. two
    # calendar days before the Wednesday statement. Monday is a trading day except on holidays, and
    # a date that is not in the session index is dropped downstream rather than shifted.
    ev["QRA_ESTIMATES"] = pd.DatetimeIndex(q - pd.Timedelta(days=2))
    return {k: v[(v >= start) & (v <= end)] for k, v in ev.items()}


def summary(ev: dict[str, pd.DatetimeIndex]) -> None:
    print("  CANDIDATE EVENT CALENDARS, with the release clock that decides whether each qualifies:")
    print(f"  {'event':16} {'time ET':>8} {'clean?':>7} {'n':>5}  {'span':23} per-year  weekday mix")
    for k, d in ev.items():
        t = TIMES[k]
        clean = "YES" if t > EXIT else "NO"
        py = len(d) / max(1e-9, (d.max() - d.min()).days / 365.25) if len(d) > 1 else 0
        wd = d.day_name().value_counts()
        mix = ", ".join(f"{a[:3]} {b}" for a, b in list(wd.items())[:3]) if len(wd) else ""
        print(f"  {k:16} {t:>8} {clean:>7} {len(d):5}  {d.min().date()}..{d.max().date()}  "
              f"{py:7.1f}   {mix}")
    print(f"\n  'clean' means the release lands after the {EXIT} exit, so the 15:59->09:31 window")
    print("  contains the pre-event uncertainty and none of the reaction. QRA_STATEMENT is the one")
    print("  sibling that fails: 08:30 is inside the window, exactly like CPI and NFP.")
