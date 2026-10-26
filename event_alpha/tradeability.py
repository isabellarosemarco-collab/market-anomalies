"""
tradeability.py — THE GATE. Is the contract still open at the moment its own news lands?

THIS IS THE CHEAPEST QUESTION IN THE WHOLE PROJECT AND IT INVALIDATES THE MOST EXPENSIVE PLAN. The
fast-feed thesis is: parse the BLS release in milliseconds, hit a book still priced at 50/50, watch it
reprice to 0.99 as humans catch up 10-30 seconds later. Every word of that is about SPEED, and speed is
irrelevant if the market is closed. So before benchmarking a single millisecond of feed latency, check
the exchange's own `close_time` against the release clock.

WHAT THE AUDIT FOUND, AND IT IS UNANIMOUS. Every scheduled US macro series on Kalshi closes BEFORE its
own print:

    CPI YoY      closes 08:29 ET   BLS releases 08:30 ET   ->  60s before
    CPI Index    closes 08:29 ET                           ->  60s before
    Payrolls     closes 08:29 ET   BLS releases 08:30 ET   ->  60s before
    Unemployment closes 08:29 ET                           ->  60s before
    Core CPI     closes 08:25 ET                           ->  5min before
    Core PCE     closes 08:25 ET   BEA releases 08:30 ET   ->  5min before
    Fed funds    closes 13:55 ET   FOMC announces 14:00 ET ->  5min before

47/47, 43/43, 32/32, 28/28, 22/22, 22/22, 10/10 markets. Not one exception, not one second of overlap.

So there is no latency window on scheduled macro. Not a narrow one, not a contested one — none. The
trade requires submitting an order into a halted market. This is a structural property of a
CFTC-regulated venue whose own contract terms bar trading by anyone holding material non-public
information on the underlying: pre-closing is how the exchange enforces that, and it is not going to be
optimised away by a faster scraper.

WHERE THE THESIS IS NOT DEAD. Pre-closing only works for SCHEDULED events, because the exchange has to
know when to halt. For unscheduled news — a court ruling on an unknown date, a resignation, a strike —
the market must stay open, and `close_time` sits months out. Those are the only candidates. The gate
sorts them for you; `depth.py` then asks whether enough money fits, which is the second thing that kills
latency plans and the reason the answer still is not free.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

# Official release clock in US/Eastern for the series we care about. Sourced from the publishing
# agency's schedule, not inferred from Kalshi. Times are local ET and DST-aware via zoneinfo.
RELEASE_ET = {
    "KXCPIYOY": (8, 30, "BLS CPI"),
    "KXCPICORE": (8, 30, "BLS core CPI"),
    "KXCPINDEX": (8, 30, "BLS CPI-U index"),
    "KXCPICOREYOY": (8, 30, "BLS core CPI YoY"),
    "KXPAYROLLS": (8, 30, "BLS nonfarm payrolls"),
    "KXUSNFP": (8, 30, "BLS nonfarm payrolls"),
    "KXU3": (8, 30, "BLS unemployment rate"),
    "KXUE": (8, 30, "BLS unemployment rate"),
    "KXJOBLESS": (8, 30, "DOL initial claims"),
    "KXJOBLESSCLAIMS": (8, 30, "DOL initial claims"),
    "KXCONTCLAIMS": (8, 30, "DOL continuing claims"),
    "KXPCECORE": (8, 30, "BEA core PCE"),
    "KXPCEHEAD": (8, 30, "BEA headline PCE"),
    "KXFED": (14, 0, "FOMC statement"),
    "KXEFFR": (14, 0, "FOMC / EFFR"),
}


@dataclass
class Verdict:
    series: str
    n: int
    close_et: str
    release_et: str
    lead_seconds: float | None      # how long BEFORE the release the market halts; >0 means no window
    tradeable_through_event: bool
    note: str

    def row(self) -> dict:
        return {
            "series": self.series, "n": self.n,
            "closes_ET": self.close_et, "release_ET": self.release_et,
            "halts_before_by": ("-" if self.lead_seconds is None
                                else f"{self.lead_seconds/60:.0f} min"),
            "window?": "YES" if self.tradeable_through_event else "NONE",
            "note": self.note,
        }


def audit_series(k, series_ticker: str, limit: int = 200) -> Verdict | None:
    """Compare every settled market's close_time against the official release clock."""
    mk = k.markets(series_ticker=series_ticker, status="settled", limit=limit)
    mk = [m for m in mk if m.get("close_time")]
    if not mk:
        return None

    closes = []
    for m in mk:
        t = datetime.fromisoformat(m["close_time"].replace("Z", "+00:00")).astimezone(ET)
        closes.append(t)
    # The clock time is the invariant; the dates differ. Report the modal ET wall-clock.
    hhmm = {}
    for t in closes:
        hhmm[t.strftime("%H:%M:%S")] = hhmm.get(t.strftime("%H:%M:%S"), 0) + 1
    modal = max(hhmm, key=hhmm.get)

    spec = RELEASE_ET.get(series_ticker)
    if not spec:
        # No known schedule. If close_time is far out, the market rides through whatever happens.
        far = sum(1 for t in closes if (t - datetime.now(ET)).days > 30)
        return Verdict(series_ticker, len(mk), modal, "unscheduled", None,
                       far > 0, "no published release clock; unscheduled-event candidate")

    hh, mm, src = spec
    # Measure the lead on the modal close against the release on the SAME calendar day, so DST is
    # handled by construction rather than by an offset constant.
    ref = closes[0]
    rel = ref.replace(hour=hh, minute=mm, second=0, microsecond=0)
    lead = (rel - ref).total_seconds()
    if len(set(hhmm)) > 1:
        note = f"{src}; close times not uniform: {hhmm}"
    else:
        note = f"{src}; unanimous across {len(mk)} markets"
    return Verdict(series_ticker, len(mk), modal, f"{hh:02d}:{mm:02d}:00",
                   lead, lead <= 0, note)


def audit_all(k, series: list[str] | None = None) -> list[Verdict]:
    out = []
    for s in (series or list(RELEASE_ET)):
        try:
            v = audit_series(k, s)
        except Exception as e:
            print(f"  ! {s}: {e}")
            continue
        if v:
            out.append(v)
    return out


def unscheduled_candidates(k, pattern: str = r"(court|scotus|ruling|resign|fired|out as|indict|"
                                             r"nominee|confirm|strike|ceasefire|shutdown|tariff)",
                           min_days_out: int = 14) -> list[dict]:
    """Open markets whose close_time is far enough out that they ride THROUGH breaking news.

    These are the only structurally valid latency candidates on the venue, because the exchange cannot
    pre-halt an event whose timing it does not know.
    """
    import re
    rx = re.compile(pattern, re.I)
    now = datetime.now(ET)
    out = []
    for e in k.events(status="open", limit=200):
        title = e.get("title") or ""
        if not rx.search(title):
            continue
        for m in k.markets(event_ticker=e["event_ticker"], limit=3)[:2]:
            ct = m.get("close_time")
            if not ct:
                continue
            t = datetime.fromisoformat(ct.replace("Z", "+00:00")).astimezone(ET)
            if (t - now) > timedelta(days=min_days_out):
                out.append({"event": title[:52], "ticker": m["ticker"],
                            "closes": t.date().isoformat(), "days_out": (t - now).days})
    return out
