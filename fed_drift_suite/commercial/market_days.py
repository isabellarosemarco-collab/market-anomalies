"""
NYSE trading-day calendar, computed from rules rather than fetched.

The personal layer gets trading days from Alpaca, which is authoritative. The API has no broker
connection, so it needs its own answer — and it genuinely needs one: `eve_of()` walks backwards from
a decision date to the previous *trading* session, and a plain weekday check would put the entry on a
closed exchange whenever a holiday lands next to a meeting. Since FOMC decisions are always Tuesday
or Wednesday, the eve is always Monday or Tuesday, which is exactly where the Monday holidays are.

Rules-based, so it works for future years without a data refresh. Ad-hoc closures (state funerals,
Sandy) are not modelled — they are unpredictable by nature, and a signal API cannot be the place that
knows about them. `/v1/status` reports the calendar it used so a client can see the assumption.
"""
from __future__ import annotations

import functools
from datetime import date, timedelta

JUNETEENTH_FROM = 2022          # first year the NYSE observed it


def easter(year: int) -> date:
    """Anonymous Gregorian algorithm. Needed only to locate Good Friday."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """n-th `weekday` (Mon=0) of a month; n=-1 means the last one."""
    if n > 0:
        d = date(year, month, 1)
        d += timedelta(days=(weekday - d.weekday()) % 7)
        return d + timedelta(weeks=n - 1)
    nxt = date(year + (month == 12), (month % 12) + 1, 1)
    d = nxt - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d: date) -> date | None:
    """Weekend rule: Saturday holidays move to Friday, Sunday holidays to Monday.

    Returns None for a January 1st falling on a Saturday — the NYSE does not observe it on
    December 31st of the prior year.
    """
    if d.weekday() == 5:
        return None if (d.month, d.day) == (1, 1) else d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


@functools.lru_cache(maxsize=64)
def holidays(year: int) -> frozenset[date]:
    fixed = [date(year, 1, 1), date(year, 7, 4), date(year, 12, 25)]
    if year >= JUNETEENTH_FROM:
        fixed.append(date(year, 6, 19))

    days = {d for d in (_observed(f) for f in fixed) if d is not None}
    days |= {
        _nth_weekday(year, 1, 0, 3),            # MLK Jr Day
        _nth_weekday(year, 2, 0, 3),            # Washington's Birthday
        _nth_weekday(year, 5, 0, -1),           # Memorial Day
        _nth_weekday(year, 9, 0, 1),            # Labor Day
        _nth_weekday(year, 11, 3, 4),           # Thanksgiving
        easter(year) - timedelta(days=2),       # Good Friday
    }
    return frozenset(days)


def is_trading_day(day: date) -> bool:
    return day.weekday() < 5 and day not in holidays(day.year)
