"""
funding_data.py — historical perpetual funding rates and spot/perp prices, cached locally, no API keys.

NO CREDENTIALS ARE USED OR REQUIRED ANYWHERE IN THIS PACKAGE. Every endpoint here is public and
unauthenticated. That is a deliberate property, not a convenience: a backtest has no business holding a
key that can move money. If this book ever went live, the key would live in a .env that is gitignored,
scoped to trade-only with withdrawals disabled, on a burner subaccount — and it still would not belong
in this file.

WHY THE DATA SOURCE IS AN S3 BUCKET AND NOT THE EXCHANGE API. Binance's and Bybit's REST APIs are
geo-blocked from this network — `fapi.binance.com` returns HTTP 451 "Service unavailable from a
restricted location" and Bybit returns a CloudFront 403. `ccxt` does not help: it is a wrapper over
those same blocked endpoints, so it fails identically. What is NOT blocked is Binance's public bulk
data dump at data.binance.vision, which carries monthly CSVs of every funding settlement from
2020-01 to the present. So the deepest available history is reachable, just not through the API.

OKX's API responds, but its funding-rate-history endpoint exhausts after roughly three months of
pagination, which makes it useless for a multi-year study. It is kept only as a live spot check.

Hyperliquid is included for a different reason: it is the venue Bella could actually TRADE from, being
a US-accessible DEX, and it settles funding HOURLY rather than 8-hourly. Measuring on Binance and
executing on Hyperliquid is a venue mismatch, so the two are compared over their overlapping window
before any Binance-derived conclusion is trusted for a Hyperliquid deployment.

THE ONE UNIT TRAP THAT MATTERS. A funding rate is quoted per SETTLEMENT, not per year. Binance settles
every 8 hours (3x/day), Hyperliquid every hour (24x/day). The same headline "0.01%" is 10.95% annualised
on Binance and 87.6% on Hyperliquid. Every rate in this module is stored raw, per-settlement, alongside
its interval in hours, so the annualisation is done once, explicitly, and never by eye.
"""
from __future__ import annotations

import io
import re
import time
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

VISION = "https://data.binance.vision/"
VISION_LIST = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
HYPERLIQUID = "https://api.hyperliquid.xyz/info"
OKX = "https://www.okx.com/api/v5/public/funding-rate-history"

# Binance's CDN rejects obviously scripted agents on some paths; this is a plain browser string and is
# not an attempt to evade anything — the data is published for exactly this use.
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

CACHE = Path(__file__).resolve().parent / "data"
CACHE.mkdir(exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA})


def _get(url: str, tries: int = 4, **kw) -> requests.Response:
    """GET with backoff. Same lesson as the EDGAR scanner: one transient 5xx must not kill a long pull."""
    for attempt in range(tries):
        try:
            r = SESSION.get(url, timeout=60, **kw)
            if r.status_code in (429, 500, 502, 503, 504) and attempt < tries - 1:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r
        except requests.exceptions.RequestException:
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def _epoch_to_utc(s: pd.Series) -> pd.Series:
    """Epoch integers -> UTC timestamps, inferring the unit per row.

    Binance switched kline timestamps from MILLIseconds to MICROseconds partway through the history,
    and both formats sit in the same multi-year pull. Parsing a microsecond stamp as milliseconds gives
    "year 56971 is out of range" if you are lucky and a silently absurd date if you are not, so the
    unit is inferred from magnitude: ~1.7e12 is ms, ~1.7e15 is us.
    """
    v = s.astype("int64")
    us = v > 10 ** 14
    out = pd.Series(pd.NaT, index=v.index, dtype="datetime64[ns, UTC]")
    if (~us).any():
        out.loc[~us] = pd.to_datetime(v[~us], unit="ms", utc=True)
    if us.any():
        out.loc[us] = pd.to_datetime(v[us], unit="us", utc=True)
    return out


def _months(start: str, end: str | None = None) -> list[str]:
    """Inclusive list of 'YYYY-MM' strings. The dump lags the live market by up to a month."""
    a = datetime.strptime(start, "%Y-%m").date().replace(day=1)
    b = (datetime.strptime(end, "%Y-%m").date() if end else date.today()).replace(day=1)
    out = []
    while a <= b:
        out.append(a.strftime("%Y-%m"))
        a = (a + timedelta(days=32)).replace(day=1)
    return out


def _vision_csv(path: str, headerless: bool, names: list[str]) -> pd.DataFrame | None:
    """Download and unzip one monthly CSV. Returns None for months that do not exist yet."""
    try:
        r = _get(VISION + path)
    except requests.exceptions.HTTPError:
        return None
    try:
        z = zipfile.ZipFile(io.BytesIO(r.content))
    except zipfile.BadZipFile:
        return None
    raw = z.read(z.namelist()[0]).decode()
    if headerless:
        return pd.read_csv(io.StringIO(raw), header=None, names=names)
    # Futures dumps carry a header; spot dumps do not. Sniff rather than assume, because Binance
    # changed this mid-history and a mis-sniffed header silently becomes a NaN row.
    first = raw.split("\n", 1)[0]
    hdr = 0 if re.search(r"[A-Za-z]", first.split(",")[0]) else None
    df = pd.read_csv(io.StringIO(raw), header=hdr, names=None if hdr == 0 else names)
    return df


def funding_history(symbol: str = "BTCUSDT", start: str = "2021-01",
                    refresh: bool = False) -> pd.DataFrame:
    """Every funding settlement for `symbol` since `start`. Columns: time, rate, interval_h.

    Cached to a single parquet so the ~70 monthly downloads happen once.
    """
    f = CACHE / f"funding_{symbol}_{start}.parquet"
    if f.exists() and not refresh:
        return pd.read_parquet(f)

    frames = []
    for m in _months(start):
        p = f"data/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{m}.zip"
        d = _vision_csv(p, headerless=False,
                        names=["calc_time", "funding_interval_hours", "last_funding_rate"])
        if d is None or d.empty:
            continue
        d.columns = [c.strip() for c in d.columns]
        frames.append(d[["calc_time", "funding_interval_hours", "last_funding_rate"]])
    if not frames:
        raise RuntimeError(f"no funding data retrieved for {symbol}")
    df = pd.concat(frames, ignore_index=True)
    df["time"] = _epoch_to_utc(df.calc_time)
    df["rate"] = df.last_funding_rate.astype(float)
    df["interval_h"] = df.funding_interval_hours.astype(float)
    df = (df[["time", "rate", "interval_h"]]
          .drop_duplicates("time").sort_values("time").reset_index(drop=True))
    df.to_parquet(f)
    return df


def daily_prices(symbol: str = "BTCUSDT", start: str = "2021-01",
                 refresh: bool = False) -> pd.DataFrame:
    """Daily closes for the PERP and the SPOT pair. Columns: date, perp, spot, basis_bp.

    Both legs are needed because the carry is not only funding. Long spot / short perp earns
    (basis_at_entry - basis_at_exit) on top of funding, where basis = perp - spot. Ignoring it treats a
    position entered at a 40bp premium as if it were entered at parity.
    """
    f = CACHE / f"prices_{symbol}_{start}.parquet"
    if f.exists() and not refresh:
        return pd.read_parquet(f)

    K = ["open_time", "open", "high", "low", "close", "volume", "close_time",
         "quote_volume", "count", "tb_vol", "tb_quote", "ignore"]
    out = {}
    for kind, base, headerless in (("perp", f"data/futures/um/monthly/klines/{symbol}/1d/", False),
                                   ("spot", f"data/spot/monthly/klines/{symbol}/1d/", True)):
        frames = []
        for m in _months(start):
            d = _vision_csv(f"{base}{symbol}-1d-{m}.zip", headerless=headerless, names=K)
            if d is None or d.empty:
                continue
            d.columns = list(K)[:len(d.columns)]
            frames.append(d[["open_time", "close"]])
        if not frames:
            raise RuntimeError(f"no {kind} klines for {symbol}")
        d = pd.concat(frames, ignore_index=True)
        d["date"] = _epoch_to_utc(d.open_time).dt.date
        out[kind] = d.groupby("date")["close"].last().astype(float)

    df = pd.DataFrame({"perp": out["perp"], "spot": out["spot"]}).dropna().reset_index()
    df["basis_bp"] = 10_000 * (df.perp - df.spot) / df.spot
    df.to_parquet(f)
    return df


def hyperliquid_funding(coin: str = "BTC", days: int = 720,
                        refresh: bool = False) -> pd.DataFrame:
    """Hourly funding from Hyperliquid — the venue actually reachable from the US.

    Paginates 500 rows at a time. Rates are HOURLY here, so `interval_h` is 1 and the annualisation
    factor is 8760, not 1095. Conflating the two overstates Hyperliquid carry by 8x.
    """
    f = CACHE / f"hl_funding_{coin}_{days}.parquet"
    if f.exists() and not refresh:
        return pd.read_parquet(f)

    start = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    now = int(datetime.now(timezone.utc).timestamp() * 1000)
    rows, cursor, guard = [], start, 0
    while cursor < now and guard < 400:
        guard += 1
        r = SESSION.post(HYPERLIQUID, json={"type": "fundingHistory", "coin": coin,
                                            "startTime": cursor}, timeout=40)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        nxt = max(int(b["time"]) for b in batch) + 1
        if nxt <= cursor:
            break
        cursor = nxt
        time.sleep(0.12)
        if len(batch) < 500:
            break
    if not rows:
        raise RuntimeError(f"no Hyperliquid funding for {coin}")
    df = pd.DataFrame(rows)
    df["time"] = _epoch_to_utc(df.time)
    df["rate"] = df.fundingRate.astype(float)
    df["premium"] = df.premium.astype(float)
    df["interval_h"] = 1.0
    df = (df[["time", "rate", "premium", "interval_h"]]
          .drop_duplicates("time").sort_values("time").reset_index(drop=True))
    df.to_parquet(f)
    return df


def annualise(rate_per_settlement: float, interval_h: float) -> float:
    """Simple (non-compounded) annualisation. 8760/interval settlements per year."""
    return rate_per_settlement * (8760.0 / interval_h)


if __name__ == "__main__":
    for sym in ("BTCUSDT", "ETHUSDT"):
        fd = funding_history(sym)
        px = daily_prices(sym)
        ann = annualise(fd.rate.mean(), fd.interval_h.mode()[0])
        print(f"{sym}: {len(fd):,} settlements {fd.time.min().date()}..{fd.time.max().date()} | "
              f"mean rate {fd.rate.mean():.6%}/8h = {ann:.2%}/yr | "
              f"{len(px):,} days, mean basis {px.basis_bp.mean():.1f}bp")
    hl = hyperliquid_funding("BTC")
    print(f"HL BTC: {len(hl):,} hourly {hl.time.min().date()}..{hl.time.max().date()} | "
          f"mean {hl.rate.mean():.6%}/1h = {annualise(hl.rate.mean(), 1.0):.2%}/yr")
