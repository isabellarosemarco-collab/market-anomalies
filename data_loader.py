"""
data_loader.py — intraday OHLCV acquisition and hygiene.

Provider note: yfinance serves 15m bars for the trailing 60 calendar days only, and
1h bars for the trailing 730. A 6-month in-sample window therefore cannot be built
from 15m data via this provider; `interval="60m"` is the only setting that supports
walk-forward folds. Alpaca is wired in as an optional provider because its IEX feed
reaches back years at minute resolution; it activates only when APCA keys are present.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

UNIVERSE = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]

CACHE = Path(__file__).parent / "data_cache"
CACHE.mkdir(exist_ok=True)

ET = "America/New_York"
RTH_OPEN, RTH_CLOSE = "09:30", "16:00"

# Provider ceilings, in calendar days, keyed by yfinance interval string.
YF_MAX_DAYS = {"1m": 7, "5m": 60, "15m": 60, "30m": 60, "60m": 730, "1h": 730, "1d": 10000}


def _rth_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep regular-session bars and drop the 16:00 stub yfinance sometimes emits."""
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df.tz_convert(ET)
    df = df.between_time(RTH_OPEN, RTH_CLOSE, inclusive="left")
    return df[df.index.dayofweek < 5]


def _clean(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Drop unusable bars. Bars are never forward-filled: a synthetic bar would let the
    backtester transact at a price that never traded."""
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].copy()
    df = df[~df.index.duplicated(keep="first")].sort_index()

    n0 = len(df)
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    df = df[df["volume"].fillna(0) > 0]

    # Integrity: a bar whose high/low does not bracket open/close is corrupt.
    bad = (df["high"] < df[["open", "close"]].max(axis=1)) | (
        df["low"] > df[["open", "close"]].min(axis=1)
    )
    df = df[~bad]

    df["symbol"] = symbol
    dropped = n0 - len(df)
    if dropped:
        print(f"    {symbol}: dropped {dropped} unusable bar(s)")
    return df


def _fetch_yf(symbol: str, interval: str, period: str) -> pd.DataFrame:
    import yfinance as yf

    # auto_adjust applies split/dividend adjustment to OHLC, normalising corporate actions.
    df = yf.download(
        symbol, period=period, interval=interval,
        progress=False, auto_adjust=True, prepost=False,
    )
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _fetch_alpaca(symbol: str, interval: str, days: int) -> pd.DataFrame:
    """Optional deeper-history provider. Requires APCA_API_KEY_ID / APCA_API_SECRET_KEY."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    unit, n = (TimeFrameUnit.Minute, int(interval.rstrip("m"))) if interval.endswith("m") \
        else (TimeFrameUnit.Hour, 1)
    client = StockHistoricalDataClient(
        os.environ["APCA_API_KEY_ID"], os.environ["APCA_API_SECRET_KEY"]
    )
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(n, unit),
        start=pd.Timestamp.utcnow() - pd.Timedelta(days=days),
    )
    df = client.get_stock_bars(req).df
    if df.empty:
        return df
    return df.reset_index(level=0, drop=True)


def load_universe(
    symbols: list[str] | None = None,
    interval: str = "60m",
    period: str = "730d",
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """Return {symbol: cleaned RTH OHLCV frame}, tz-aware in US/Eastern."""
    symbols = symbols or UNIVERSE
    tag = f"{interval}_{period}"
    provider = "alpaca" if os.getenv("APCA_API_KEY_ID") else "yfinance"

    cap = YF_MAX_DAYS.get(interval)
    if provider == "yfinance" and cap and period.endswith("d"):
        want = int(period.rstrip("d"))
        if want > cap:
            print(f"  ! {interval} capped at {cap}d by provider (asked {want}d)")

    print(f"  provider={provider}  interval={interval}  period={period}")
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        f = CACHE / f"{sym}_{tag}.parquet"
        if use_cache and f.exists():
            out[sym] = pd.read_parquet(f)
            continue
        try:
            if provider == "alpaca":
                days = int(period.rstrip("d")) if period.endswith("d") else 730
                raw = _fetch_alpaca(sym, interval, days)
            else:
                raw = _fetch_yf(sym, interval, period)
        except Exception as e:
            print(f"    {sym}: FETCH FAILED {type(e).__name__}: {str(e)[:110]}")
            continue
        if raw is None or raw.empty:
            print(f"    {sym}: no data returned")
            continue
        df = _clean(_rth_only(raw), sym)
        if df.empty:
            print(f"    {sym}: empty after cleaning")
            continue
        df.to_parquet(f)
        out[sym] = df
    return out


SECTORS = ["XLE", "XLF", "XLK", "XLI", "XLP", "XLU", "XLV", "XLY", "XLC"]
BENCH, CASH = "SPY", "BIL"

# Sub-industry and thematic ETFs, chosen for dispersion rather than coverage: the point is that
# semis and clean energy can diverge 60 points in six months, which nine broad sectors cannot do.
# Inception dates matter. Most of these list 2005-2011, so a 2014 start gives a full lookback
# immediately. Two do not: HACK lists 2014-11 and JETS 2015-04. They are kept rather than dropped
# because eligibility is evaluated per formation date (a NaN momentum score is simply ineligible),
# so including them adds names over time without back-filling history that never existed.
INDUSTRY = [
    # technology
    "SMH", "SOXX", "XSD", "IGV", "FDN", "HACK",
    # health care
    "XBI", "IBB", "XPH", "IHE", "IHI", "IHF",
    # financials
    "KRE", "KBE", "KIE", "KCE",
    # consumer
    "ITB", "XHB", "XRT", "PEJ", "PBJ",
    # industrials and transport
    "IYT", "JETS", "ITA", "PPA",
    # materials, metals, resources
    "XME", "SLX", "GDX", "GDXJ", "MOO", "WOOD", "PHO", "URA",
    # energy
    "OIH", "XOP", "FCG",
    # clean energy
    "TAN", "ICLN", "PBW",
    # real estate
    "VNQ", "IYR",
]


def load_daily(
    symbols: list[str],
    start: str = "2014-01-01",
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """Daily adjusted OHLCV. Deliberately bypasses the intraday RTH filter, which would
    drop every daily bar (they timestamp at midnight, outside 09:30-16:00).

    Listing dates differ across this universe (XLC begins 2018-06, XLRE 2015-10). Frames are
    returned at their true start and never back-filled, so the strategy layer can restrict
    each rebalance to names that actually traded on that date.
    """
    import yfinance as yf

    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        f = CACHE / f"{sym}_1d_{start}.parquet"
        if use_cache and f.exists():
            out[sym] = pd.read_parquet(f)
            continue
        try:
            df = yf.download(sym, start=start, interval="1d",
                             progress=False, auto_adjust=True)
        except Exception as e:
            print(f"    {sym}: FETCH FAILED {type(e).__name__}: {str(e)[:100]}")
            continue
        if df is None or df.empty:
            print(f"    {sym}: no data")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        df = df[~df.index.duplicated(keep="first")].sort_index()
        df = df.dropna(subset=["close"])
        df = df[df["close"] > 0]
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.to_parquet(f)
        out[sym] = df
    return out


def coverage_report(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-symbol span, bar count, and inferred bars/day — used to scale bar-based params."""
    rows = []
    for sym, df in data.items():
        days = df.index.normalize().nunique()
        rows.append({
            "symbol": sym,
            "bars": len(df),
            "sessions": days,
            "bars_per_day": round(len(df) / max(days, 1), 2),
            "start": df.index[0].strftime("%Y-%m-%d %H:%M"),
            "end": df.index[-1].strftime("%Y-%m-%d %H:%M"),
        })
    return pd.DataFrame(rows).set_index("symbol")


if __name__ == "__main__":
    import sys

    interval = sys.argv[1] if len(sys.argv) > 1 else "60m"
    period = sys.argv[2] if len(sys.argv) > 2 else "730d"
    print(f"Loading universe: {UNIVERSE}")
    data = load_universe(interval=interval, period=period)
    print()
    print(coverage_report(data).to_string())
    total = sum(len(d) for d in data.values())
    print(f"\n{len(data)} symbols, {total:,} total bars")
