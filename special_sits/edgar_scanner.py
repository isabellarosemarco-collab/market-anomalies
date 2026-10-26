"""
edgar_scanner.py — find issuer self-tender offers on EDGAR that mention an odd-lot provision.

WHY FULL-TEXT SEARCH AND NOT THE BULK INDEX. The obvious approach is the quarterly form.idx files,
which list every filing by type. Those are served from an S3 bucket that returns 403 AccessDenied from
this network, so they are not an option here. EDGAR's full-text search API (efts.sec.gov) works, and it
is the better tool anyway: it searches inside document TEXT, so it returns the exact exhibit containing
the phrase "odd lot" rather than every tender offer ever filed. That turns the parsing problem from
"download 40 exhibits per deal and hunt" into "fetch the one file EDGAR already told us matches".

  root_forms="SC TO-I" captures amendments (SC TO-I/A) as well as originals, which matters because the
  final Dutch-auction clearing price is announced in an amendment, not the original offer.

COVERAGE LIMIT, STATED UP FRONT. Full-text search only indexes filings from 2001 onward and caps at
10,000 hits per query. Neither binds here: the realistic query window is months, not decades.

RATE LIMIT. SEC's published policy is 10 requests/second and a declared User-Agent with contact info.
This module sleeps between requests and sets that header. Do not remove either — the penalty is an IP
ban, and it is applied to the whole household, not the script.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date, timedelta

import requests

FTS = "https://efts.sec.gov/LATEST/search-index"
ARCHIVE = "https://www.sec.gov/Archives/edgar/data"
TICKER_MAP = "https://www.sec.gov/files/company_tickers.json"

# SEC requires a real contact address. Change this to yours before running at any volume.
USER_AGENT = "BellaMarco-Research/1.0 (bellamarco.research@gmail.com)"
PAUSE = 0.15            # ~7 req/s, under the published 10/s ceiling


@dataclass
class Hit:
    """One EDGAR document that matched the search, with enough metadata to fetch it."""
    adsh: str                       # accession number, e.g. 0001104659-26-104051
    cik: str
    company: str
    form: str                       # SC TO-I or SC TO-I/A
    file_date: str
    filename: str
    file_description: str
    score: float
    ticker: str | None = None
    text: str = field(default="", repr=False)

    @property
    def doc_url(self) -> str:
        return f"{ARCHIVE}/{int(self.cik)}/{self.adsh.replace('-', '')}/{self.filename}"

    @property
    def filing_url(self) -> str:
        nd = self.adsh.replace("-", "")
        return f"{ARCHIVE}/{int(self.cik)}/{nd}/{self.adsh}-index.htm"


class Edgar:
    def __init__(self, user_agent: str = USER_AGENT):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self._tickers: dict[str, str] | None = None

    # ---------------------------------------------------------------- low level
    def _get(self, url: str, tries: int = 4, **kw):
        """GET with backoff on transient server errors.

        EDGAR full-text search returns sporadic 500s that succeed on an immediate retry — the same
        365-day query that failed once returned 200 five times in a row moments later. Without a retry
        a single blip kills a twelve-minute scan at document one, which reads as "the window is empty"
        rather than "the server hiccuped". 429 is also retried, and more patiently: that one is the
        rate limiter, and the penalty for ignoring it is an IP ban on the whole household.
        """
        last = None
        for attempt in range(tries):
            time.sleep(PAUSE)
            try:
                r = self.s.get(url, timeout=30, **kw)
                if r.status_code in (429, 500, 502, 503, 504) and attempt < tries - 1:
                    wait = (3.0 if r.status_code == 429 else 1.0) * (2 ** attempt)
                    print(f"  ~ http {r.status_code}, retry {attempt+1}/{tries-1} in {wait:.0f}s")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                return r
            except requests.exceptions.RequestException as e:
                last = e
                if attempt == tries - 1:
                    raise
                time.sleep(2 ** attempt)
        raise last if last else RuntimeError("unreachable")

    def tickers(self) -> dict[str, str]:
        """CIK (as int-string) -> ticker. One 800KB fetch, cached for the process."""
        if self._tickers is None:
            d = self._get(TICKER_MAP).json()
            self._tickers = {str(v["cik_str"]): v["ticker"] for v in d.values()}
        return self._tickers

    # ---------------------------------------------------------------- search
    def search(self, phrase: str, forms: str = "SC TO-I",
               days: int = 120, limit: int = 400) -> list[Hit]:
        """Every document of `forms` filed in the last `days` whose text contains `phrase`.

        `phrase` is quoted, so this is an exact-phrase search rather than a bag of words. "odd lot"
        unquoted would match any document containing both common words and drown the signal.
        """
        end = date.today()
        start = end - timedelta(days=days)
        out: list[Hit] = []
        frm = 0
        while frm < limit:
            d = self._get(FTS, params={
                "q": f'"{phrase}"', "forms": forms,
                "startdt": start.isoformat(), "enddt": end.isoformat(),
                "from": frm,
            }).json()
            hits = d.get("hits", {}).get("hits", [])
            if not hits:
                break
            tk = self.tickers()
            for h in hits:
                src = h["_source"]
                cik = src["ciks"][0]
                name = (src.get("display_names") or ["?"])[0]
                out.append(Hit(
                    adsh=src["adsh"], cik=cik,
                    company=name.split("  (CIK")[0].strip(),
                    form=src.get("form", ""), file_date=src.get("file_date", ""),
                    filename=h["_id"].split(":", 1)[1],
                    file_description=src.get("file_description", "") or "",
                    score=h.get("_score", 0.0),
                    ticker=tk.get(str(int(cik))),
                ))
            total = d["hits"]["total"]["value"]
            frm += len(hits)
            if frm >= min(total, limit):
                break
        return out

    def fetch_text(self, hit: Hit) -> str:
        """Raw document text with HTML stripped. Cached onto the Hit."""
        if hit.text:
            return hit.text
        from bs4 import BeautifulSoup
        raw = self._get(hit.doc_url).text
        soup = BeautifulSoup(raw, "lxml")
        # Collapse whitespace: the parsers below use regexes with \s+ and tables emit ragged runs.
        hit.text = " ".join(soup.get_text(" ").split())
        return hit.text


def group_by_deal(hits: list[Hit]) -> dict[tuple[str, str], list[Hit]]:
    """Collapse document hits into deals, keyed by (cik, ticker).

    One tender offer generates many matching documents: the Schedule TO itself, the Offer to Purchase,
    the Letter of Transmittal, and one amendment per extension or price change. Reporting each as a
    separate opportunity would triple-count. Grouping by issuer is right because a company runs at most
    one self-tender at a time in practice.
    """
    deals: dict[tuple[str, str], list[Hit]] = {}
    for h in hits:
        deals.setdefault((h.cik, h.ticker or "?"), []).append(h)
    for v in deals.values():
        v.sort(key=lambda h: (h.file_date, h.score), reverse=True)
    return deals
