"""
FOMC decision dates, scraped from federalreserve.gov and cached.

This is a port of the parser validated during the research phase. It produces 255 scheduled decision
dates for 1996-2027 with exactly 8 per year, the sole exception being 2020's 7 — correct, because the
scheduled March 2020 meeting was cancelled and replaced by an unscheduled emergency cut.

Four things this parser has to get right, all of which silently produced wrong dates before being
fixed. They are listed because the failure mode of a calendar bug is a null result, not an error:

  1. The decision date is the LAST day of a multi-day meeting, when the statement prints at 14:00.
  2. Month abbreviations and straddles: "January" + "27-28", "Jan/Feb" + "31-1", "April/May" + "30-1",
     and "July 31-August 1" all appear. The last month named pairs with the last day number.
  3. Notation votes, conference calls and unscheduled meetings are NOT decision days and must be
     excluded — an unscheduled meeting is also not something a forward-looking calendar can trade.
  4. Older pages split a two-day meeting into one heading per day ("September 15 Meeting" and
     "September 16 Meeting" are one meeting). Scheduled meetings are never on consecutive days, so
     adjacent pairs collapse to the later date.

FOMC EVE is the trading day immediately BEFORE a decision date. The backtested trade buys that day's
close and sells the decision day's open, so it holds only the pre-decision positioning drift and
takes no 14:00 announcement risk. "Trading day" must come from the broker's calendar, not from a
weekday check, so `is_eve()` takes an `is_trading_day` callable rather than guessing about holidays.
"""
from __future__ import annotations

import json
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Iterable

CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
HISTORICAL_URL = "https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"
HIST_FIRST_YEAR = 1996          # earliest fomchistorical<year>.htm page that exists
USER_AGENT = "Mozilla/5.0 (fed-drift-suite; scheduled-calendar-fetch)"

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_CACHE = DATA_DIR / "fomc_dates.json"

MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]
MONTHS = {m[:3].lower(): i + 1 for i, m in enumerate(MONTH_NAMES)}
NOT_A_DECISION = re.compile(r"notation|unscheduled|conference call", re.I)
HIST_HEADING = re.compile(
    r"^([A-Za-z]+(?:/[A-Za-z]+)?)\s+([\d][\dA-Za-z\s-]*?)\s+Meeting\s*-\s*((?:19|20)\d\d)\s*$")


class CalendarError(RuntimeError):
    pass


def _resolve(mon_txt: str, day_txt: str, year: int) -> date | None:
    """Turn a Fed meeting label into its decision date — the last day of the meeting."""
    toks = re.findall(r"[A-Za-z]+", f"{mon_txt} {day_txt}")
    months = [MONTHS[t[:3].lower()] for t in toks if t[:3].lower() in MONTHS]
    nums = re.findall(r"\d+", day_txt)
    if not months or not nums:
        return None
    month, day = months[-1], int(nums[-1])
    if months[0] == 12 and month == 1:          # Dec/Jan straddle rolls the year
        year += 1
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _dedupe_adjacent(days: Iterable[date]) -> list[date]:
    d = sorted(set(days))
    return [x for i, x in enumerate(d) if i == len(d) - 1 or (d[i + 1] - x).days > 1]


def scrape(first_year: int = HIST_FIRST_YEAR) -> list[date]:
    """Fetch decision dates from the Fed. Network-bound; callers should prefer `FomcCalendar`."""
    import requests
    from bs4 import BeautifulSoup

    hdr = {"User-Agent": USER_AGENT}
    found: set[date] = set()

    def soup_of(url: str):
        resp = requests.get(url, timeout=30, headers=hdr)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")

    # Current layout (2021 onward): explicit month/date spans, year from the enclosing panel.
    soup = soup_of(CALENDAR_URL)
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
            if ts:
                found.add(ts)

    # Historical layout (pre-2021): the meeting lives only in the panel heading.
    for year in range(first_year, 2021):
        try:
            soup = soup_of(HISTORICAL_URL.format(year=year))
        except Exception:
            continue                                  # a missing year is not fatal
        for h in soup.find_all(class_=re.compile(r"panel-heading")):
            txt = h.get_text(" ", strip=True)
            if NOT_A_DECISION.search(txt):
                continue
            m = HIST_HEADING.match(txt)
            if m:
                ts = _resolve(m.group(1), m.group(2), int(m.group(3)))
                if ts:
                    found.add(ts)

    if not found:
        raise CalendarError("scraped zero FOMC dates — the page layout has probably changed")
    return _dedupe_adjacent(found)


class FomcCalendar:
    """Cached FOMC decision dates with FOMC-eve lookup.

    The cache is the normal path: these dates are published years ahead and change rarely, so the
    engine should not depend on federalreserve.gov being reachable at 15:45. `refresh_if_thin()`
    re-scrapes only when the cache is running out of future dates.
    """

    def __init__(self, cache_path: Path = DEFAULT_CACHE, auto_fetch: bool = True):
        self.cache_path = Path(cache_path)
        self.auto_fetch = auto_fetch
        self._dates: list[date] | None = None

    # ------------------------------------------------------------------ cache plumbing
    def _read_cache(self) -> list[date] | None:
        if not self.cache_path.exists():
            return None
        try:
            raw = json.loads(self.cache_path.read_text())
            return [date.fromisoformat(s) for s in raw]
        except (ValueError, TypeError, json.JSONDecodeError):
            return None

    def _write_cache(self, days: list[date]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps([d.isoformat() for d in days], indent=0))

    def dates(self) -> list[date]:
        if self._dates is None:
            cached = self._read_cache()
            if cached:
                self._dates = cached
            elif self.auto_fetch:
                self._dates = scrape()
                self._write_cache(self._dates)
            else:
                raise CalendarError(
                    f"no cached dates at {self.cache_path} and auto_fetch is off")
        return self._dates

    def refresh(self) -> list[date]:
        self._dates = scrape()
        self._write_cache(self._dates)
        return self._dates

    def refresh_if_thin(self, today: date, min_future: int = 4) -> bool:
        """Re-scrape when fewer than `min_future` decision dates remain ahead of `today`."""
        try:
            remaining = [d for d in self.dates() if d >= today]
        except CalendarError:
            remaining = []
        if len(remaining) >= min_future:
            return False
        self.refresh()
        return True

    # --------------------------------------------------------------------- queries
    def is_decision_day(self, day: date) -> bool:
        return day in set(self.dates())

    def next_decision(self, today: date) -> date | None:
        return next((d for d in self.dates() if d >= today), None)

    def eve_of(self, decision: date, is_trading_day: Callable[[date], bool]) -> date | None:
        """The trading day immediately before a decision date.

        Walks backwards past weekends and holidays. Bounded at 10 days so a broken
        `is_trading_day` cannot spin forever.
        """
        d = decision - timedelta(days=1)
        for _ in range(10):
            if is_trading_day(d):
                return d
            d -= timedelta(days=1)
        return None

    def is_eve(self, day: date, is_trading_day: Callable[[date], bool]) -> tuple[bool, date | None]:
        """Is `day` the trading session immediately before an FOMC decision?

        Returns (is_eve, decision_date). Checks the next few decision dates rather than only the
        first, so nothing depends on `day` itself being a trading day.
        """
        if not is_trading_day(day):
            return False, None
        for decision in self.dates():
            if decision < day:
                continue
            if decision - day > timedelta(days=10):
                break
            if self.eve_of(decision, is_trading_day) == day:
                return True, decision
        return False, None

    def upcoming(self, today: date, n: int = 4,
                 is_trading_day: Callable[[date], bool] | None = None) -> list[dict]:
        out = []
        for d in self.dates():
            if d < today or len(out) >= n:
                continue
            eve = self.eve_of(d, is_trading_day) if is_trading_day else None
            out.append({"decision": d, "eve": eve, "days_away": (d - today).days})
        return out

    def past_decisions(self, before: date | None = None) -> list[date]:
        """Every decision date strictly before `before` — the backtest universe."""
        cut = before or date.today()
        return [d for d in self.dates() if d < cut]

    def coverage(self) -> dict:
        """Cache summary, including per-year counts. Eight a year, except 2020's seven."""
        ds = self.dates()
        per_year: dict[int, int] = {}
        for d in ds:
            per_year[d.year] = per_year.get(d.year, 0) + 1
        return {"count": len(ds), "first": ds[0], "last": ds[-1], "per_year": per_year}
