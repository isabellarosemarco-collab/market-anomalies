"""
event_calendar.py — macro event dates, sourced rather than remembered where possible.

FOMC dates are scraped from federalreserve.gov (authoritative, including the historical pages for
2018-2020) and cached. NFP and OpEx are pure calendar rules, so they are generated. CPI is the
problem case: bls.gov returns 403 to programmatic requests, so those dates are a hardcoded table and
are therefore the one input here that could be wrong.

That is why `validate()` exists. A hardcoded date list that is off by a day produces a silently
null result rather than an error, so every calendar is checked against its own expected market
signature before being used:

  * CPI and NFP print at 08:30 ET, before the open, so a correct date lands elevated ABSOLUTE
    OVERNIGHT return on the release morning.
  * FOMC decisions print at 14:00 ET, during the session, so a correct date lands elevated absolute
    INTRADAY return on the decision day and should show nothing unusual overnight.

If a calendar fails its own signature test, the dates are wrong and any result built on them is
meaningless. Reported, not assumed.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path("data_cache")
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]
# The Fed writes both "January" and "Jan/Feb", so match on the 3-letter stem.
MONTHS = {m[:3].lower(): i + 1 for i, m in enumerate(MONTH_NAMES)}

# Meetings that produce no 2pm rate decision. A notation vote is an administrative ballot taken
# without a meeting; unscheduled meetings are real but are not a tradable ex-ante calendar, and
# including them would leak hindsight into an "eve of a known event" filter.
NOT_A_DECISION = re.compile(r"notation|unscheduled|conference call", re.I)

HIST_FIRST_YEAR = 1996          # earliest fomchistorical<year>.htm page that exists

# CPI release dates (data for the prior month), 08:30 ET. Hardcoded because bls.gov blocks scraping.
# Lower confidence for late 2025 onward: the autumn 2025 shutdown disrupted the BLS schedule, so
# validate() carries real weight for those years.
CPI_DATES = """
2018-01-12 2018-02-14 2018-03-13 2018-04-11 2018-05-10 2018-06-12 2018-07-12 2018-08-10
2018-09-13 2018-10-11 2018-11-14 2018-12-12
2019-01-11 2019-02-13 2019-03-12 2019-04-10 2019-05-10 2019-06-12 2019-07-11 2019-08-13
2019-09-12 2019-10-10 2019-11-13 2019-12-11
2020-01-14 2020-02-13 2020-03-11 2020-04-10 2020-05-12 2020-06-10 2020-07-14 2020-08-12
2020-09-11 2020-10-13 2020-11-12 2020-12-10
2021-01-13 2021-02-10 2021-03-10 2021-04-13 2021-05-12 2021-06-10 2021-07-13 2021-08-11
2021-09-14 2021-10-13 2021-11-10 2021-12-10
2022-01-12 2022-02-10 2022-03-10 2022-04-12 2022-05-11 2022-06-10 2022-07-13 2022-08-10
2022-09-13 2022-10-13 2022-11-10 2022-12-13
2023-01-12 2023-02-14 2023-03-14 2023-04-12 2023-05-10 2023-06-13 2023-07-12 2023-08-10
2023-09-13 2023-10-12 2023-11-14 2023-12-12
2024-01-11 2024-02-13 2024-03-12 2024-04-10 2024-05-15 2024-06-12 2024-07-11 2024-08-14
2024-09-11 2024-10-10 2024-11-13 2024-12-11
2025-01-15 2025-02-12 2025-03-12 2025-04-10 2025-05-13 2025-06-11 2025-07-15 2025-08-12
2025-09-11 2025-10-24 2025-11-13 2025-12-18
2026-01-13 2026-02-11 2026-03-11 2026-04-10 2026-05-12 2026-06-10 2026-07-14 2026-08-12
2026-09-11
""".split()


# ------------------------------------------------------------------------------- FOMC

def _resolve(mon_txt: str, day_txt: str, year: int) -> pd.Timestamp | None:
    """Turn a Fed meeting label into its decision date — the LAST day of the meeting.

    Handles every shape the two page layouts use: "January" + "27-28", "Jan/Feb" + "31-1",
    "April/May" + "30-1", and "January" + "31-February 1". The decision day is the last day number
    and it belongs to the last month named; a Dec/Jan straddle rolls the year forward.
    """
    toks = re.findall(r"[A-Za-z]+", f"{mon_txt} {day_txt}")
    months = [MONTHS[t[:3].lower()] for t in toks if t[:3].lower() in MONTHS]
    nums = re.findall(r"\d+", day_txt)
    if not months or not nums:
        return None
    month, day = months[-1], int(nums[-1])
    if months[0] == 12 and month == 1:
        year += 1
    try:
        return pd.Timestamp(year=year, month=month, day=day)
    except ValueError:
        return None


def fomc_dates(refresh: bool = False) -> pd.DatetimeIndex:
    """Scheduled FOMC decision days (the LAST day of each meeting, when the statement prints).

    Two layouts. The current calendar page marks up each meeting with `fomc-meeting__month` /
    `fomc-meeting__date` spans and covers 2021 onward; the per-year historical pages carry the
    meeting only in a panel heading ("April/May 30-1 Meeting - 2019") and cover 2018-2020.
    """
    CACHE.mkdir(exist_ok=True)
    cache = CACHE / "fomc_dates.json"
    if cache.exists() and not refresh:
        return pd.DatetimeIndex(json.loads(cache.read_text()))

    import requests
    from bs4 import BeautifulSoup

    hdr = {"User-Agent": "Mozilla/5.0 (quant research)"}
    found: set[pd.Timestamp] = set()

    soup = BeautifulSoup(requests.get(
        "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
        timeout=30, headers=hdr).text, "html.parser")
    for panel in soup.find_all(class_=re.compile(r"panel-default")):
        head = panel.find(class_=re.compile(r"panel-heading"))
        ym = re.search(r"\b(20\d\d)\b", head.get_text(" ", strip=True) if head else "")
        if not ym:
            continue
        for mt in panel.find_all(class_="fomc-meeting"):
            mon_el = mt.find(class_="fomc-meeting__month")
            day_el = mt.find(class_="fomc-meeting__date")
            if not (mon_el and day_el):
                continue
            day_txt = day_el.get_text(" ", strip=True)
            if NOT_A_DECISION.search(day_txt):
                continue
            ts = _resolve(mon_el.get_text(" ", strip=True), day_txt.replace("*", ""),
                          int(ym.group(1)))
            if ts is not None:
                found.add(ts)

    # Pre-2021: headings of the form "<Month[/Month]> <d[-d]> Meeting - <year>". Pages exist back
    # to 1996, which matters because it is the only source of a pre-2018 holdout for a
    # calendar-based filter — there is no other way to test an FOMC signal out of sample.
    # The day group must admit letters: month-straddling meetings are written "July 31-August 1".
    pat = re.compile(
        r"^([A-Za-z]+(?:/[A-Za-z]+)?)\s+([\d][\dA-Za-z\s-]*?)\s+Meeting\s*-\s*((?:19|20)\d\d)\s*$")
    for y in range(HIST_FIRST_YEAR, 2021):
        soup = BeautifulSoup(requests.get(
            f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm",
            timeout=30, headers=hdr).text, "html.parser")
        for h in soup.find_all(class_=re.compile(r"panel-heading")):
            txt = h.get_text(" ", strip=True)
            if NOT_A_DECISION.search(txt):
                continue
            m = pat.match(txt)
            if m:
                ts = _resolve(m.group(1), m.group(2), int(m.group(3)))
                if ts is not None:
                    found.add(ts)

    # Some older pages split a two-day meeting into one heading per day ("September 15 Meeting" and
    # "September 16 Meeting" are the single Sep 15-16 2003 meeting). Scheduled meetings are never on
    # consecutive days, so any adjacent pair is one meeting and only the second day is a decision.
    days = sorted(found)
    days = [d for i, d in enumerate(days)
            if i == len(days) - 1 or (days[i + 1] - d).days > 1]
    out = [d.strftime("%Y-%m-%d") for d in days]
    cache.write_text(json.dumps(out))
    return pd.DatetimeIndex(out)


# ------------------------------------------------------------- pure calendar rules

def _nth_weekday(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=month, day=1)
    shift = (weekday - d.dayofweek) % 7
    return d + pd.Timedelta(days=shift + 7 * (n - 1))


def nfp_dates(start: str, end: str) -> pd.DatetimeIndex:
    """Non-Farm Payrolls: first Friday of the month, 08:30 ET. BLS occasionally shifts this when the
    first Friday falls in the first day or two of the month; the resulting handful of misdated
    entries out of ~105 is accepted rather than guessed at, and validate() will show it as noise."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    out = [_nth_weekday(y, m, 4, 1)
           for y in range(s.year, e.year + 1) for m in range(1, 13)]
    return pd.DatetimeIndex([d for d in out if s <= d <= e])


def opex_dates(start: str, end: str) -> pd.DatetimeIndex:
    """Monthly equity option expiration: third Friday."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    out = [_nth_weekday(y, m, 4, 3)
           for y in range(s.year, e.year + 1) for m in range(1, 13)]
    return pd.DatetimeIndex([d for d in out if s <= d <= e])


def cpi_dates(start: str, end: str) -> pd.DatetimeIndex:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    d = pd.DatetimeIndex(sorted(CPI_DATES))
    return d[(d >= s) & (d <= e)]


# ------------------------------------------------------------------------- validation

def _t(a: pd.Series, b: pd.Series) -> float:
    """Welch t-statistic, group a vs group b."""
    if len(a) < 5 or len(b) < 5:
        return float("nan")
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    return float((a.mean() - b.mean()) / np.sqrt(va + vb))


def validate(r: pd.DataFrame, cals: dict[str, pd.DatetimeIndex]) -> pd.DataFrame:
    """Does each calendar show the market signature its release time implies?

    `session` says which leg should light up: 08:30 releases move the overnight gap, 14:00 decisions
    move the intraday session. A ratio near 1.0 with t near 0 means the dates are wrong (or the
    event does not move the market), and either way nothing should be built on them.
    """
    expect = {"FOMC": "r_day", "CPI": "r_night", "NFP": "r_night", "OPEX": "r_day"}
    rows = []
    for name, dates in cals.items():
        col = expect.get(name, "r_night")
        hit = r.index.isin(dates)
        if hit.sum() < 5:
            rows.append({"calendar": name, "n_matched": int(hit.sum())})
            continue
        for leg in ("r_night", "r_day"):
            a, b = r.loc[hit, leg].abs(), r.loc[~hit, leg].abs()
            rows.append({"calendar": name, "session": leg, "expected": leg == col,
                         "n_matched": int(hit.sum()),
                         "mean_abs_bp": 1e4 * a.mean(), "baseline_abs_bp": 1e4 * b.mean(),
                         "ratio": a.mean() / b.mean(), "t_welch": _t(a, b)})
    return pd.DataFrame(rows)


def build(start: str, end: str) -> dict[str, pd.DatetimeIndex]:
    f = fomc_dates()
    f = f[(f >= pd.Timestamp(start)) & (f <= pd.Timestamp(end))]
    return {"FOMC": f, "CPI": cpi_dates(start, end),
            "NFP": nfp_dates(start, end), "OPEX": opex_dates(start, end)}


if __name__ == "__main__":
    cals = build("2018-01-01", "2026-12-31")
    for k, v in cals.items():
        print(f"{k:5} n={len(v):4}  {v[0].date()} .. {v[-1].date()}")
    print("\nFOMC per year:", dict(pd.Series(cals['FOMC']).dt.year.value_counts().sort_index()))
