"""
kalshi.py — read-only public Kalshi client. No API keys, no order placement, by design.

WHY THERE IS NO `place_order` IN THIS FILE. The brief asked for a client with "market quotes & order
placement". Quotes are here; order placement is not, and that is deliberate on two grounds. First,
sequencing: an execution path built before the edge is measured is the odd-lot pipeline all over again —
beautiful plumbing aimed at nothing. Second, blast radius: order placement needs a key that can move
money, and the measurement work that decides whether to proceed needs no key at all. Everything below
runs unauthenticated against public endpoints. If this ever goes live, the key belongs in a gitignored
.env, scoped trade-only with withdrawals disabled, and it still does not belong in this module.

THE ONE THING THAT MATTERS ABOUT THIS API, AND IT IS A FIELD NAME. Kalshi returns order books under
`orderbook_fp` (fixed-point), NOT `orderbook`. Reading the wrong key returns an empty dict rather than
an error, so every market silently looks like it has zero depth and a capacity study concludes the
venue is empty. It is not empty; you were reading the wrong field. Same class of bug as pandas' reserved
`.flags` attribute in the tender scanner: a lookup that fails soft and lies.

Base URL note: `trading-api.kalshi.com` now returns 401 with a migration notice. The live host is
`api.elections.kalshi.com`, which serves ALL series, not just elections, despite the name.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import requests

BASE = "https://api.elections.kalshi.com/trade-api/v2"
UA = "BellaMarco-Research/1.0 (bellamarco.research@gmail.com)"
PAUSE = 0.12                     # be a good citizen; the public endpoints are a courtesy
MAX_CANDLES = 5000               # hard server limit per candlestick request


class Kalshi:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA, "Accept": "application/json"})

    def _get(self, path: str, tries: int = 4, **params):
        last = None
        for attempt in range(tries):
            time.sleep(PAUSE)
            try:
                r = self.s.get(f"{BASE}{path}", params=params or None, timeout=40)
                if r.status_code in (429, 500, 502, 503, 504) and attempt < tries - 1:
                    time.sleep((3.0 if r.status_code == 429 else 1.0) * (2 ** attempt))
                    continue
                r.raise_for_status()
                return r.json()
            except requests.exceptions.RequestException as e:
                last = e
                if attempt == tries - 1:
                    raise
                time.sleep(2 ** attempt)
        raise last if last else RuntimeError("unreachable")

    # ------------------------------------------------------------------ discovery
    def series(self, category: str) -> list[dict]:
        return self._get("/series", category=category).get("series", [])

    def markets(self, *, series_ticker: str | None = None, event_ticker: str | None = None,
                status: str | None = None, limit: int = 200) -> list[dict]:
        p = {"limit": min(limit, 1000)}
        if series_ticker:
            p["series_ticker"] = series_ticker
        if event_ticker:
            p["event_ticker"] = event_ticker
        if status:
            p["status"] = status
        return self._get("/markets", **p).get("markets", [])

    def market(self, ticker: str) -> dict:
        return self._get(f"/markets/{ticker}").get("market", {})

    def events(self, *, status: str = "open", limit: int = 200) -> list[dict]:
        return self._get("/events", status=status, limit=min(limit, 200)).get("events", [])

    # ------------------------------------------------------------------ microstructure
    def orderbook(self, ticker: str) -> tuple[list, list]:
        """(yes_levels, no_levels) as [(price_dollars, contracts), ...], best price first.

        Reads `orderbook_fp`. See the module docstring for why that spelling is load-bearing.
        """
        d = self._get(f"/markets/{ticker}/orderbook")
        ob = d.get("orderbook_fp") or d.get("orderbook") or {}
        f = lambda rows: [(float(p), float(q)) for p, q in (rows or [])]
        return f(ob.get("yes_dollars")), f(ob.get("no_dollars"))

    def trades(self, *, ticker: str | None = None, limit: int = 1000) -> list[dict]:
        """Recent trade tape. Note the size field is `count_fp`, not `count`."""
        p = {"limit": min(limit, 1000)}
        if ticker:
            p["ticker"] = ticker
        return self._get("/markets/trades", **p).get("trades", [])

    def candles(self, series_ticker: str, ticker: str, start_ts: int, end_ts: int,
                period_interval: int = 1) -> list[dict]:
        """OHLC of yes_bid / yes_ask plus volume and open interest per period.

        `period_interval` is in MINUTES (1, 60, 1440). The server caps a request at 5000 candles, so a
        1-minute pull covers at most ~3.5 days; asking for a month at 1-minute returns a 400 rather than
        truncating, which is the honest behaviour but means the caller must chunk.
        """
        span = (end_ts - start_ts) / 60.0 / period_interval
        if span > MAX_CANDLES:
            raise ValueError(f"{span:.0f} candles requested, server max {MAX_CANDLES}; chunk the range")
        return self._get(f"/series/{series_ticker}/markets/{ticker}/candlesticks",
                         start_ts=start_ts, end_ts=end_ts,
                         period_interval=period_interval).get("candlesticks", [])


@dataclass
class Depth:
    """Dollar capacity on one side of a book, split by distance from the touch."""
    ticker: str
    side: str
    touch: float | None
    near: float          # $ available within NEAR_CENTS of the touch
    total: float         # $ available at any price
    junk: float          # $ resting at 1c-2c: lottery orders, not real liquidity
    levels: int

    def row(self) -> dict:
        return {"ticker": self.ticker[:40], "side": self.side,
                "touch": self.touch, "near$": round(self.near),
                "total$": round(self.total), "junk$": round(self.junk), "lvls": self.levels}


NEAR_CENTS = 0.05
JUNK_MAX = 0.02


def measure_depth(levels: list[tuple[float, float]], ticker: str, side: str) -> Depth:
    """Convert a raw book side into honest capacity.

    WHY `junk` IS BROKEN OUT SEPARATELY AND WHY IT IS THE WHOLE STORY ON THIS VENUE. Kalshi books are
    stuffed with orders resting at 1 cent — free lottery tickets on a tail outcome, placed by people who
    will never be filled at a price that matters. Summing a book naively makes a market with $200,000
    of 1-cent bids look institutionally deep when the actual tradeable size near the touch is a few
    hundred dollars. A latency strategy needs to BUY at a price well below fair value the instant news
    breaks, so the only number that counts is size available within a few cents of the touch. Penny
    depth is not capacity; it is noise wearing capacity's clothes.
    """
    real = [(p, q) for p, q in levels if p > JUNK_MAX]
    touch = max((p for p, _ in real), default=None)
    near = sum(p * q for p, q in real if touch is not None and p >= touch - NEAR_CENTS)
    return Depth(ticker=ticker, side=side, touch=touch,
                 near=near, total=sum(p * q for p, q in real),
                 junk=sum(p * q for p, q in levels if p <= JUNK_MAX), levels=len(real))
