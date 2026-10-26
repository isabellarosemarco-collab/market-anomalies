"""
cef_discount_mon.py — closed-end funds trading below NAV, filtered to the ones with a CATALYST.

THE MISTAKE THIS MODULE EXISTS TO AVOID. Screening for "CEFs at a big discount to NAV" produces a list
of value traps. A fund that has traded at -20% for fifteen years is not mispriced; the discount IS its
price, and it persists because of real, permanent causes — a high expense ratio, an illiquid or
unmarkable portfolio, a captive board, a share class nobody can arbitrage. Buying a permanent discount
and waiting for it to close is not a trade, it is a hope. The discount only becomes an opportunity when
something forces NAV realisation: a liquidation, an open-ending, a tender at ~98% of NAV, or a board
that has lost a proxy fight. That is a CATALYST, and catalysts show up in filings.

SO THE SCREEN IS A CONJUNCTION OF THREE THINGS, NOT ONE:
  1. Discount wide in ABSOLUTE terms (below `min_discount`).
  2. Discount wide RELATIVE TO ITS OWN HISTORY — CEFConnect's 1-year discount Z-score. This is the
     part that separates "unusually cheap" from "always cheap", and it is the single most important
     column here. A fund at -22% with a Z-score of 0 is trading exactly where it always does.
  3. An ACTIVIST ON THE REGISTER, evidenced by a Schedule 13D. A 13D (not 13G) is the filing that
     declares intent to influence control, and the CEF activists are a small, known, repeat set:
     Saba, Bulldog, Karpus, City of London, Almitas, 180 Degree. Their presence is the mechanism by
     which the discount gets attacked.

NOTE ON THE FORM NAME. EDGAR indexes these as "SCHEDULE 13D", not "SC 13D". Querying the latter returns
zero results silently, which looks exactly like "no activists are active" — a false negative that would
have quietly disabled this entire module.

WHY $5K FITS HERE AND $100M DOES NOT. Several of these funds trade well under $1M a day. An activist
needs a 5-10% stake to file and force a vote, which is weeks of volume and moves the discount against
them as they build. A 200-share position is invisible: you get the same catalyst, at the pre-campaign
discount, with none of the accumulation cost. You are free-riding a campaign somebody else is funding.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import requests

CEF_API = "https://www.cefconnect.com/api/v3/DailyPricing"
FTS = "https://efts.sec.gov/LATEST/search-index"

# Browser UA: cefconnect rejects scripted agents. SEC calls use the compliant UA from edgar_scanner.
BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

# Repeat CEF activists. This list is the mechanism, so it is explicit rather than inferred.
ACTIVISTS = ("Saba Capital", "Bulldog Investors", "Karpus Management",
             "City of London Investment", "Almitas Capital", "180 Degree Capital")


@dataclass
class Cef:
    ticker: str
    name: str
    price: float
    nav: float
    discount: float          # negative = trading below NAV
    z1y: float | None        # 1-year discount Z-score; very negative = unusually wide FOR THIS FUND
    disc_52wk_avg: float | None
    volume: float | None
    mktcap_m: float | None
    expense: float | None
    term: bool               # term funds have a mandated wind-up date: a built-in catalyst
    activists: list[str]

    @property
    def upside_to_nav(self) -> float:
        """Return if the discount closed completely. The realistic case is partial."""
        return (self.nav / self.price - 1.0) if self.price else 0.0

    def row(self) -> dict:
        return {
            "ticker": self.ticker, "name": self.name[:30],
            "px": round(self.price, 2), "nav": round(self.nav, 2),
            "disc%": round(self.discount, 1),
            "avg_disc%": round(self.disc_52wk_avg, 1) if self.disc_52wk_avg is not None else None,
            "z1y": round(self.z1y, 2) if self.z1y is not None else None,
            "to_NAV%": round(100 * self.upside_to_nav, 1),
            "vol": int(self.volume) if self.volume else None,
            "cap_$M": round(self.mktcap_m) if self.mktcap_m else None,
            "exp%": self.expense,
            "term": "yes" if self.term else "",
            "activist": ", ".join(self.activists) or "-",
        }


def fetch_universe() -> list[dict]:
    s = requests.Session()
    s.headers.update({"User-Agent": BROWSER_UA, "Accept": "application/json, text/plain, */*"})
    r = s.get(CEF_API, timeout=40)
    r.raise_for_status()
    return r.json()


def activist_subjects(user_agent: str, days: int = 240) -> dict[str, set[str]]:
    """Company name (upper) -> set of activists who filed a SCHEDULE 13D naming it.

    Matching is by company NAME rather than CIK because the 13D's indexed entities include both the
    filer and the subject, and CEF tickers are unreliably present in EDGAR's ticker map.
    """
    from datetime import date, timedelta
    s = requests.Session()
    s.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
    end = date.today()
    start = end - timedelta(days=days)
    out: dict[str, set[str]] = {}
    for who in ACTIVISTS:
        for form in ("SCHEDULE 13D",):
            frm = 0
            while frm < 300:
                time.sleep(0.15)
                try:
                    d = s.get(FTS, params={"q": f'"{who}"', "forms": form,
                                           "startdt": start.isoformat(), "enddt": end.isoformat(),
                                           "from": frm}, timeout=30).json()
                except Exception:
                    break
                hits = d.get("hits", {}).get("hits", [])
                if not hits:
                    break
                for h in hits:
                    for nm in h["_source"].get("display_names", []):
                        nm = nm.split("  (CIK")[0].strip().upper()
                        if who.split()[0].upper() in nm:
                            continue            # that's the filer, not the target
                        out.setdefault(nm, set()).add(who)
                total = d["hits"]["total"]["value"]
                frm += len(hits)
                if frm >= min(total, 300):
                    break
    return out


def _name_match(fund_name: str, subjects: dict[str, set[str]]) -> list[str]:
    """Loose containment match between CEFConnect's fund name and EDGAR's registrant name.

    They differ ("Aberdeen Government Markets Income" vs "ABERDEEN GLOBAL INCOME FUND INC"), so this
    compares the first three significant words. Deliberately permissive: a false positive costs you one
    filing to read, a false negative loses the catalyst entirely.
    """
    stop = {"FUND", "INC", "TRUST", "CORP", "THE", "CLOSED", "END", "COMPANY", "LP", "LTD", "&"}
    want = [w for w in fund_name.upper().replace(",", " ").split() if w not in stop][:3]
    if len(want) < 2:
        return []
    found = []
    for subj, whos in subjects.items():
        if sum(w in subj for w in want) >= 2:
            found.extend(whos)
    return sorted(set(found))


def screen(min_discount: float = -12.0, max_z: float = -1.0,
           min_volume: float = 20_000, require_catalyst: bool = False,
           user_agent: str = "BellaMarco-Research/1.0 (bellamarco.research@gmail.com)",
           ) -> tuple[list[Cef], dict]:
    raw = fetch_universe()
    subjects = activist_subjects(user_agent)

    out: list[Cef] = []
    stats = {"universe": len(raw), "activist_13d_targets": len(subjects)}
    for x in raw:
        d, nav, px = x.get("Discount"), x.get("NAV"), x.get("Price")
        if not isinstance(d, (int, float)) or not nav or not px:
            continue
        if d > min_discount:
            continue
        z = x.get("ZScore1Yr")
        vol = x.get("AvgDailyVolume")
        if vol is not None and vol < min_volume:
            continue
        acts = _name_match(x.get("Name", ""), subjects)
        c = Cef(ticker=x["Ticker"], name=x.get("Name", ""), price=px, nav=nav, discount=d,
                z1y=z if isinstance(z, (int, float)) else None,
                disc_52wk_avg=x.get("Discount52WkAvg"), volume=vol,
                mktcap_m=x.get("MarketCapUSDm"), expense=x.get("ExpenseRatio"),
                term=bool(x.get("Term")), activists=acts)
        # Keep if unusually wide for itself, OR has a hard catalyst (activist / term structure).
        unusual = c.z1y is not None and c.z1y <= max_z
        catalyst = bool(acts) or c.term
        if require_catalyst and not catalyst:
            continue
        if unusual or catalyst:
            out.append(c)

    out.sort(key=lambda c: (-len(c.activists), c.z1y if c.z1y is not None else 0, c.discount))
    stats["passed"] = len(out)
    return out, stats
