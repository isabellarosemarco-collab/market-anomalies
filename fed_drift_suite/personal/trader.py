"""
Broker layer: Alpaca order placement with the auction constraints the strategy actually depends on.

WHY AUCTION ORDERS AND NOT PLAIN MARKET ORDERS

The backtested edge is ~18-30bp per trade. A round trip through the bid-ask spread on QQQ costs
roughly 1-2bp in calm conditions and considerably more in the last minute of the session, so
execution quality is not a rounding error here — it is a meaningful fraction of the edge. The
research priced every fill at the official closing and opening auction crosses, where there is no
spread to cross. Reproducing that requires:

    entry: MarketOrderRequest(..., time_in_force=TimeInForce.CLS)   # market-on-close
    exit:  MarketOrderRequest(..., time_in_force=TimeInForce.OPG)   # market-on-open

TWO HARD BROKER CONSTRAINTS, both of which contradict the obvious reading of "trade at 3:59 PM":

  * MOC (CLS) orders are REJECTED if submitted at or after 15:50 ET. Entry must be staged by ~15:45.
  * MOO (OPG) orders are REJECTED if submitted at or after 09:28 ET. Exit must be staged pre-open.

A market order at 15:59 or 09:31 is a different trade from the one that was tested: it pays the
spread and, at the open, lands in the most volatile minute of the session rather than at the cross.
It is implemented as `style="market"` for the fallback path, and `Order.degraded` is set True so the
notifier can say so plainly instead of reporting it as a clean fill.

FRACTIONAL SHARES. Alpaca supports fractional quantities only for DAY-TIF market orders, never for
CLS/OPG. So auction orders size to whole shares (rounding DOWN, so buying power is never exceeded),
and fractional sizing is available only in market mode. `size_position()` returns both so the caller
can see what was given up.
"""
from __future__ import annotations

import math
import time as _time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from config import MOC_CUTOFF, MOO_CUTOFF, Config
from core.analytics import expectancy


class BrokerError(RuntimeError):
    pass


@dataclass
class Order:
    """Normalized order result, so callers never touch SDK objects."""
    id: str
    symbol: str
    side: str
    qty: float
    style: str                      # 'auction' | 'market'
    tif: str
    status: str
    submitted_at: datetime | None = None
    filled_qty: float = 0.0
    filled_avg_price: float | None = None
    degraded: bool = False          # True when a market order replaced the intended auction fill
    raw: Any = field(default=None, repr=False)

    @property
    def is_filled(self) -> bool:
        return self.status in ("filled",) and self.filled_qty > 0


@dataclass
class Sizing:
    symbol: str
    target_notional: float
    price: float
    qty_whole: int
    qty_fractional: float
    notional_whole: float
    reason: str = ""
    # Pulled from core.analytics so the dollar figure in the phone alert and the bp figure the API
    # sells to clients are computed from the same trade file. The HOLDOUT expectancy is used, not the
    # full-sample one, because the full sample includes the period the rule was designed on.
    expected_net_bp: float = 0.0
    expected_pnl_usd: float = 0.0
    historical_win_rate: float = 0.0

    @property
    def tradable(self) -> bool:
        return self.qty_whole > 0


class BaseTrader:
    """Interface shared by the live and dry-run traders."""

    def describe_account(self) -> dict: raise NotImplementedError
    def is_trading_day(self, day: date) -> bool: raise NotImplementedError
    def is_market_open(self) -> bool: raise NotImplementedError
    def last_price(self, symbol: str) -> float: raise NotImplementedError
    def get_position(self, symbol: str) -> dict | None: raise NotImplementedError
    def submit(self, symbol: str, side: str, qty: float, style: str) -> Order:
        raise NotImplementedError
    def refresh_order(self, order: Order) -> Order: return order

    # ------------------------------------------------------------------ shared logic
    def size_position(self, cfg: Config, symbol: str, equity: float | None = None,
                      buying_power: float | None = None) -> Sizing:
        """How many shares to buy, given the sizing policy and the account's actual capacity."""
        price = self.last_price(symbol)
        if price <= 0:
            raise BrokerError(f"{symbol}: bad reference price {price}")

        per_symbol = max(1, len(cfg.symbols))
        if cfg.sizing_mode == "fixed_cash":
            target = cfg.sizing_value / per_symbol
        else:
            if equity is None:
                raise BrokerError("pct_equity sizing needs account equity")
            target = equity * cfg.sizing_value / per_symbol

        reason = ""
        target = min(target, cfg.max_notional / per_symbol)
        if buying_power is not None and target > buying_power:
            target = buying_power
            reason = "capped by buying power"

        qty_whole = int(math.floor(target / price))
        if qty_whole == 0:
            reason = (reason + "; " if reason else "") + (
                f"target ${target:,.2f} is below one share at ${price:,.2f}")

        notional = qty_whole * price
        try:
            exp = expectancy(symbol)
            bp, win = exp["net_bp_holdout"], exp["win_rate"]
        except (KeyError, FileNotFoundError):
            bp, win = 0.0, 0.0          # research data absent: size anyway, just do not quote a number
        return Sizing(symbol=symbol, target_notional=target, price=price, qty_whole=qty_whole,
                      qty_fractional=round(target / price, 6),
                      notional_whole=notional, reason=reason,
                      expected_net_bp=bp, expected_pnl_usd=notional * bp / 1e4,
                      historical_win_rate=win)

    @staticmethod
    def auction_window_open(now_et: datetime, side: str) -> bool:
        """Can an auction order still be accepted right now?"""
        t = now_et.time()
        return t < MOC_CUTOFF if side == "buy" else t < MOO_CUTOFF


class AlpacaTrader(BaseTrader):
    """Live/paper trading against Alpaca via the modern alpaca-py SDK."""

    def __init__(self, cfg: Config):
        # Imported lazily so the dry-run path and the test suite do not require the SDK.
        from alpaca.trading.client import TradingClient

        self.cfg = cfg
        self.client = TradingClient(cfg.api_key, cfg.api_secret, paper=cfg.paper)
        self._data = None
        self._trading_days: dict[int, set[date]] = {}

    # ------------------------------------------------------------------- account
    def describe_account(self) -> dict:
        a = self.client.get_account()
        if getattr(a, "trading_blocked", False):
            raise BrokerError("account is trading-blocked")
        return {
            "account_number": getattr(a, "account_number", "?"),
            "status": str(getattr(a, "status", "?")),
            "equity": float(a.equity or 0),
            "cash": float(a.cash or 0),
            "buying_power": float(a.buying_power or 0),
            "pattern_day_trader": bool(getattr(a, "pattern_day_trader", False)),
            "shorting_enabled": bool(getattr(a, "shorting_enabled", False)),
            "fractional": bool(getattr(a, "fractional_trading", True)),
        }

    def is_market_open(self) -> bool:
        return bool(self.client.get_clock().is_open)

    def is_trading_day(self, day: date) -> bool:
        """Authoritative from the broker's calendar, cached one year at a time.

        A weekday check would misfire on the ~9 market holidays a year, and an FOMC decision the day
        after a holiday would then stage entry on a closed session.
        """
        year = day.year
        if year not in self._trading_days:
            from alpaca.trading.requests import GetCalendarRequest
            cal = self.client.get_calendar(GetCalendarRequest(
                start=date(year, 1, 1), end=date(year, 12, 31)))
            self._trading_days[year] = {
                (c.date if isinstance(c.date, date) else c.date.date()) for c in cal}
        return day in self._trading_days[year]

    def last_price(self, symbol: str) -> float:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestTradeRequest

        if self._data is None:
            self._data = StockHistoricalDataClient(self.cfg.api_key, self.cfg.api_secret)
        q = self._data.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbol))
        return float(q[symbol].price)

    # -------------------------------------------------------------------- positions
    def get_position(self, symbol: str) -> dict | None:
        try:
            p = self.client.get_open_position(symbol)
        except Exception:
            return None                      # alpaca raises on "no position", which is not an error
        return {"symbol": symbol, "qty": float(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "market_value": float(p.market_value or 0),
                "unrealized_pl": float(p.unrealized_pl or 0)}

    # ----------------------------------------------------------------------- orders
    def submit(self, symbol: str, side: str, qty: float, style: str) -> Order:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        if style == "auction":
            tif = TimeInForce.CLS if side == "buy" else TimeInForce.OPG
            qty = int(qty)                   # CLS/OPG reject fractional quantities
            if qty <= 0:
                raise BrokerError(f"{symbol}: auction order needs a whole-share qty, got {qty}")
        else:
            tif = TimeInForce.DAY

        req = MarketOrderRequest(
            symbol=symbol, qty=qty, side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=tif)
        o = self.client.submit_order(req)
        return Order(id=str(o.id), symbol=symbol, side=side, qty=float(qty), style=style,
                     tif=str(tif.value if hasattr(tif, "value") else tif),
                     status=str(o.status.value if hasattr(o.status, "value") else o.status),
                     submitted_at=getattr(o, "submitted_at", None),
                     filled_qty=float(o.filled_qty or 0),
                     filled_avg_price=float(o.filled_avg_price) if o.filled_avg_price else None,
                     degraded=(style == "market"), raw=o)

    def refresh_order(self, order: Order) -> Order:
        o = self.client.get_order_by_id(order.id)
        order.status = str(o.status.value if hasattr(o.status, "value") else o.status)
        order.filled_qty = float(o.filled_qty or 0)
        order.filled_avg_price = float(o.filled_avg_price) if o.filled_avg_price else None
        order.raw = o
        return order

    def await_fill(self, order: Order, timeout_s: float = 120, poll_s: float = 5) -> Order:
        """Poll until terminal. Auction orders stay 'accepted' until the cross, so the caller should
        only await after the auction has run (post-16:00 / post-09:30)."""
        deadline = _time.monotonic() + timeout_s
        while _time.monotonic() < deadline:
            order = self.refresh_order(order)
            if order.status in ("filled", "canceled", "rejected", "expired"):
                return order
            _time.sleep(poll_s)
        return order


class DryRunTrader(BaseTrader):
    """Deterministic in-memory broker. No network, no keys, no SDK.

    Used by `DRY_RUN=true` and by the test suite. Prices follow a fixed stub so assertions on P&L
    are exact; a `price_map` can be injected to drive specific scenarios, and `fail_next` forces
    submit() to raise so the retry path can be exercised.
    """

    STUB_PRICES = {"QQQ": 500.0, "SPY": 600.0, "IWM": 230.0}

    def __init__(self, cfg: Config, equity: float = 100_000.0,
                 price_map: dict[str, float] | None = None,
                 trading_days: set[date] | None = None, market_open: bool = True):
        self.cfg = cfg
        self.equity = equity
        self.prices = dict(self.STUB_PRICES, **(price_map or {}))
        self._trading_days = trading_days
        self._market_open = market_open
        self.positions: dict[str, dict] = {}
        self.orders: list[Order] = []
        self.fail_next = 0
        self._seq = 0

    def describe_account(self) -> dict:
        return {"account_number": "DRYRUN0001", "status": "ACTIVE", "equity": self.equity,
                "cash": self.equity, "buying_power": self.equity * 2,
                "pattern_day_trader": False, "shorting_enabled": True, "fractional": True}

    def is_market_open(self) -> bool:
        return self._market_open

    def is_trading_day(self, day: date) -> bool:
        if self._trading_days is not None:
            return day in self._trading_days
        return day.weekday() < 5        # weekday fallback; holidays are not modelled in dry run

    def last_price(self, symbol: str) -> float:
        return float(self.prices.get(symbol, 100.0))

    def set_price(self, symbol: str, price: float) -> None:
        self.prices[symbol] = float(price)

    def get_position(self, symbol: str) -> dict | None:
        return self.positions.get(symbol)

    def submit(self, symbol: str, side: str, qty: float, style: str) -> Order:
        if self.fail_next > 0:
            self.fail_next -= 1
            raise BrokerError(f"simulated broker failure for {symbol} {side}")
        if style == "auction":
            qty = int(qty)
            if qty <= 0:
                raise BrokerError(f"{symbol}: auction order needs a whole-share qty, got {qty}")
        self._seq += 1
        px = self.last_price(symbol)
        # Dry run fills immediately at the reference price. Real auction orders do not fill until
        # the cross, so the engine must not rely on immediate fills — it re-checks via refresh.
        if side == "buy":
            self.positions[symbol] = {"symbol": symbol, "qty": float(qty),
                                      "avg_entry_price": px, "market_value": qty * px,
                                      "unrealized_pl": 0.0}
        else:
            self.positions.pop(symbol, None)
        o = Order(id=f"dry-{self._seq:04d}", symbol=symbol, side=side, qty=float(qty), style=style,
                  tif="cls" if (style == "auction" and side == "buy")
                  else ("opg" if style == "auction" else "day"),
                  status="filled", submitted_at=datetime.now(), filled_qty=float(qty),
                  filled_avg_price=px, degraded=(style == "market"))
        self.orders.append(o)
        return o

    def await_fill(self, order: Order, timeout_s: float = 0, poll_s: float = 0) -> Order:
        return order


def build_trader(cfg: Config) -> BaseTrader:
    return DryRunTrader(cfg) if cfg.dry_run else AlpacaTrader(cfg)
