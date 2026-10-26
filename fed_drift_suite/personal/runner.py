"""
The engine: a clock-driven state machine for one trade per FOMC meeting.

TIME WINDOWS, NOT INSTANTS

Every phase owns a window rather than a moment. A cron that fires exactly at 15:45:00 loses the trade
if the process was restarting, the network blipped, or the host was descheduled for 40 seconds. A
window means a process that comes back up at 15:47 still acts, and one that comes back at 15:52 knows
it missed the MOC cutoff and says so instead of submitting an order that will be rejected.

  stage           15:00 -> entry_at     announce, size, pre-flight checks
  entry           15:45 -> 15:50        MOC buy  (hard broker cutoff at 15:50)
  entry_fallback  15:57 -> 15:59:30     plain market buy, only if MOC never went in
  entry_confirm   16:05 -> 16:45        read the cross, persist fill price, notify
  exit            09:25 -> 09:28        MOO sell (hard broker cutoff at 09:28)
  exit_fallback   09:31 -> 10:00        plain market sell, only if the position is still open
  exit_confirm    09:36 -> 11:00        read the fill, compute overnight P&L, clear state

IDEMPOTENCE

Each phase records itself in `state.json` under the Eastern date, so a crash-restart loop cannot
double-submit. The position itself is also persisted, which is what makes the overnight hand-off
safe: the morning exit does not recompute whether yesterday was FOMC eve, it simply acts on the
recorded position. A calendar that changed overnight therefore cannot orphan a real position.

DAYLIGHT SAVING

All comparisons happen on `datetime.now(pytz.timezone('America/New_York'))`. The wall-clock times
above are Eastern by definition, so ET-local comparison is correct through both transitions without
any offset arithmetic. Never compare against UTC or host-local time here.
"""
from __future__ import annotations

import json
import logging
import sys
import time as _time
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Callable

import pytz

# So `python personal/runner.py` works as well as `python -m personal.runner`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import MARKET_TZ, MOC_CUTOFF, MOO_CUTOFF, Config       # noqa: E402
from core.calendar import FomcCalendar                             # noqa: E402
from personal.notifier import Notifier                             # noqa: E402
from personal.trader import BaseTrader, BrokerError, Order         # noqa: E402

log = logging.getLogger("feddrift.runner")
ET = pytz.timezone(MARKET_TZ)


def now_et() -> datetime:
    return datetime.now(ET)


@dataclass
class Phase:
    name: str
    start: time
    end: time
    handler: Callable[[datetime], None]


class State:
    """Tiny JSON-backed store. Written after every mutation — the process may die at any point."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.data: dict = {"phases": {}, "open_position": None}
        self.load()

    def load(self) -> None:
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text())
            except (ValueError, json.JSONDecodeError):
                log.error("state file at %s is corrupt; starting clean", self.path)
        self.data.setdefault("phases", {})
        self.data.setdefault("open_position", None)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, default=str))
        tmp.replace(self.path)              # atomic, so a kill mid-write cannot corrupt the file

    def done(self, day: date, phase: str) -> bool:
        return phase in self.data["phases"].get(day.isoformat(), [])

    def mark(self, day: date, phase: str) -> None:
        self.data["phases"].setdefault(day.isoformat(), []).append(phase)
        self.save()

    @property
    def position(self) -> dict | None:
        return self.data["open_position"]

    def set_position(self, pos: dict | None) -> None:
        self.data["open_position"] = pos
        self.save()

    def prune(self, before: date) -> None:
        self.data["phases"] = {k: v for k, v in self.data["phases"].items()
                               if k >= before.isoformat()}
        self.save()


class Engine:
    def __init__(self, cfg: Config, trader: BaseTrader, notifier: Notifier,
                 calendar: FomcCalendar, state: State | None = None,
                 now_fn: Callable[[], datetime] = now_et):
        self.cfg = cfg
        self.trader = trader
        self.notifier = notifier
        self.calendar = calendar
        self.state = state or State(cfg.state_path)
        self.now_fn = now_fn
        self.mode = cfg.redacted()["mode"]

    # ------------------------------------------------------------------ orchestration
    def phases(self) -> list[Phase]:
        c = self.cfg
        return [
            Phase("stage", c.stage_at, c.entry_at, self._stage),
            Phase("entry", c.entry_at, MOC_CUTOFF, self._entry),
            Phase("entry_fallback", c.entry_fallback_at, time(15, 59, 30), self._entry_fallback),
            Phase("entry_confirm", c.entry_confirm_at, time(16, 45), self._entry_confirm),
            Phase("exit", c.exit_at, MOO_CUTOFF, self._exit),
            Phase("exit_fallback", c.exit_fallback_at, time(10, 0), self._exit_fallback),
            Phase("exit_confirm", c.exit_confirm_at, time(11, 0), self._exit_confirm),
        ]

    def tick(self, now: datetime | None = None) -> list[str]:
        """One evaluation of the clock. Returns the phases that ran, for tests and logs."""
        now = now or self.now_fn()
        today, t, ran = now.date(), now.time(), []
        for ph in self.phases():
            if not (ph.start <= t < ph.end) or self.state.done(today, ph.name):
                continue
            try:
                ph.handler(now)
            except Exception as exc:                          # noqa: BLE001
                log.exception("phase %s failed", ph.name)
                self.notifier.error(self.mode, f"phase '{ph.name}' raised", str(exc))
            finally:
                self.state.mark(today, ph.name)
                ran.append(ph.name)
        return ran

    def run_forever(self) -> None:
        import schedule

        self.startup()
        schedule.every(max(5.0, self.cfg.poll_seconds)).seconds.do(self._guarded_tick)
        log.info("entering main loop, polling every %.0fs", self.cfg.poll_seconds)
        while True:
            schedule.run_pending()
            _time.sleep(1)

    def _guarded_tick(self) -> None:
        """The loop must survive anything a tick can throw, including broker outages."""
        try:
            self.tick()
        except Exception as exc:                              # noqa: BLE001
            log.exception("tick failed")
            self.notifier.error(self.mode, "engine tick failed", str(exc))

    # --------------------------------------------------------------------- lifecycle
    def startup(self) -> dict:
        now = self.now_fn()
        info = self.cfg.redacted()
        try:
            acct = self.trader.describe_account()
            info["account"] = acct["account_number"]
            info["equity"] = f"${acct['equity']:,.2f}"
        except Exception as exc:                              # noqa: BLE001
            info["account"] = f"UNAVAILABLE ({exc})"

        refreshed = self.calendar.refresh_if_thin(now.date())
        if refreshed:
            log.info("FOMC calendar re-scraped (cache was running out of future dates)")
        up = self.calendar.upcoming(now.date(), 4, self.trader.is_trading_day)
        self.state.prune(now.date() - timedelta(days=30))
        self.notifier.startup(self.mode, info, up)

        # A position recorded yesterday that is still open means an exit was missed. Say so loudly.
        pos = self.state.position
        if pos and pos.get("entry_date") != now.date().isoformat():
            self.notifier.error(
                self.mode, "stale open position found at startup",
                f"Recorded entry on {pos.get('entry_date')} has not been closed. "
                f"Legs: {pos.get('legs')}. The next exit window will attempt to flatten it.")
        return info

    # ------------------------------------------------------------------------ phases
    def _is_eve(self, day: date) -> tuple[bool, date | None]:
        return self.calendar.is_eve(day, self.trader.is_trading_day)

    def _plan(self) -> list:
        acct = self.trader.describe_account()
        plans = []
        for sym in self.cfg.symbols:
            plans.append(self.trader.size_position(
                self.cfg, sym, equity=acct["equity"], buying_power=acct["buying_power"]))
        return plans

    def _stage(self, now: datetime) -> None:
        is_eve, decision = self._is_eve(now.date())
        if not is_eve:
            log.debug("%s is not FOMC eve", now.date())
            return
        if self.state.position:
            self.notifier.warn(self.mode, "FOMC eve, but a position is already open",
                               "Skipping entry to avoid stacking exposure. "
                               f"Existing: {self.state.position.get('legs')}")
            return
        plans = self._plan()
        log.info("FOMC eve for %s; plans=%s", decision, [(p.symbol, p.qty_whole) for p in plans])
        self.notifier.staged(self.mode, decision, self.cfg.symbols, plans, now)
        if not any(p.tradable for p in plans):
            self.notifier.warn(self.mode, "no tradable size",
                               "; ".join(f"{p.symbol}: {p.reason}" for p in plans))

    def _submit_entry(self, now: datetime, style: str) -> None:
        is_eve, decision = self._is_eve(now.date())
        if not is_eve or self.state.position:
            return
        if not self.trader.is_market_open():
            self.notifier.warn(self.mode, "market reported closed during entry window",
                               "No order submitted.")
            return

        legs, orders = [], []
        for plan in self._plan():
            if not plan.tradable:
                continue
            order = self._retry(f"BUY {plan.symbol}", self.trader.submit,
                                plan.symbol, "buy", plan.qty_whole, style)
            if order is None:
                continue
            orders.append(order)
            legs.append({"symbol": plan.symbol, "qty": order.qty, "order_id": order.id,
                         "style": style, "degraded": order.degraded,
                         "entry_price": order.filled_avg_price, "ref_price": plan.price})

        if not legs:
            self.notifier.error(self.mode, "entry produced no orders",
                                "All legs failed or were untradable. No exposure taken.")
            return

        self.state.set_position({"decision_date": decision.isoformat(),
                                 "entry_date": now.date().isoformat(),
                                 "style": style, "legs": legs})
        self.notifier.entered(self.mode, orders, now)

    def _entry(self, now: datetime) -> None:
        if self.cfg.order_style == "market":
            return                     # market mode uses the fallback window only
        self._submit_entry(now, "auction")

    def _entry_fallback(self, now: datetime) -> None:
        """Only fires if the MOC never went in. A market buy here pays the spread, and the notifier
        flags it, because it is not the fill the strategy was measured at."""
        if self.state.position:
            return
        is_eve, _ = self._is_eve(now.date())
        if not is_eve:
            return
        if self.cfg.order_style == "auction":
            self.notifier.warn(
                self.mode, "MOC window missed — falling back to a market order",
                "The auction cross was unavailable, so this fill will pay the bid-ask spread. "
                "Backtested edge was measured at the cross.")
        self._submit_entry(now, "market")

    def _entry_confirm(self, now: datetime) -> None:
        """Read back the auction cross and persist real fill prices for tomorrow's P&L."""
        pos = self.state.position
        if not pos or pos.get("entry_date") != now.date().isoformat():
            return
        filled = []
        for leg in pos["legs"]:
            price = self._resolve_fill(leg["order_id"], leg["symbol"])
            if price:
                leg["entry_price"] = price
            filled.append(leg)
        pos["legs"] = filled
        self.state.set_position(pos)

        unpriced = [l["symbol"] for l in filled if not l.get("entry_price")]
        summary = ", ".join(
            f"{l['symbol']} {l['qty']:g} @ "
            + (f"${l['entry_price']:,.2f}" if l.get("entry_price") else "UNCONFIRMED")
            for l in filled)
        if unpriced:
            self.notifier.error(self.mode, "entry fill unconfirmed",
                                f"No fill price for {unpriced}. Verify manually in Alpaca before "
                                f"the open. Recorded: {summary}")
        else:
            self.notifier.warn(self.mode, "Entry confirmed at the close", summary)

    def _resolve_fill(self, order_id: str, symbol: str) -> float | None:
        """Fill price from the order if the broker has it, else from the resulting position."""
        try:
            probe = Order(id=order_id, symbol=symbol, side="buy", qty=0, style="auction",
                          tif="cls", status="new")
            o = self.trader.refresh_order(probe)
            if o.filled_avg_price:
                return float(o.filled_avg_price)
        except Exception as exc:                              # noqa: BLE001
            log.warning("could not refresh order %s: %s", order_id, exc)
        p = self.trader.get_position(symbol)
        return float(p["avg_entry_price"]) if p else None

    def _exit(self, now: datetime) -> None:
        if self.cfg.order_style == "market":
            return
        self._submit_exit(now, "auction")

    def _submit_exit(self, now: datetime, style: str) -> None:
        pos = self.state.position
        if not pos:
            return
        orders = []
        for leg in pos["legs"]:
            live = self.trader.get_position(leg["symbol"])
            qty = float(live["qty"]) if live else float(leg["qty"])
            if qty <= 0:
                continue
            order = self._retry(f"SELL {leg['symbol']}", self.trader.submit,
                                leg["symbol"], "sell", qty, style)
            if order is None:
                continue
            leg["exit_order_id"] = order.id
            leg["exit_style"] = style
            leg["exit_degraded"] = order.degraded
            if order.filled_avg_price:
                leg["exit_price"] = order.filled_avg_price
            orders.append(order)
        self.state.set_position(pos)
        if not orders:
            self.notifier.error(self.mode, "exit submitted no orders",
                                f"Position on record: {pos['legs']}. Verify in Alpaca immediately.")

    def _exit_fallback(self, now: datetime) -> None:
        """Still holding after the open means the MOO did not go in or did not fill. Flatten."""
        pos = self.state.position
        if not pos:
            return
        still_open = [l for l in pos["legs"] if self.trader.get_position(l["symbol"])]
        if not still_open:
            return
        self.notifier.warn(self.mode, "still holding after the open — flattening at market",
                           f"Legs: {[l['symbol'] for l in still_open]}")
        self._submit_exit(now, "market")

    def _exit_confirm(self, now: datetime) -> None:
        pos = self.state.position
        if not pos:
            return
        closes, unresolved = [], []
        for leg in pos["legs"]:
            exit_px = leg.get("exit_price")
            if not exit_px and leg.get("exit_order_id"):
                exit_px = self._resolve_exit_fill(leg)
            entry_px = leg.get("entry_price") or leg.get("ref_price")
            if not (exit_px and entry_px):
                unresolved.append(leg["symbol"])
                continue
            qty = float(leg["qty"])
            closes.append({"symbol": leg["symbol"], "qty": qty, "entry_price": float(entry_px),
                           "exit_price": float(exit_px), "cost_basis": qty * float(entry_px),
                           "pnl": qty * (float(exit_px) - float(entry_px)),
                           "degraded": bool(leg.get("exit_degraded") or leg.get("degraded"))})

        if self.trader.get_position(pos["legs"][0]["symbol"]) and not closes:
            self.notifier.error(self.mode, "position still open at exit_confirm",
                                "Automated exit did not complete. Flatten manually now.")
            return

        if closes:
            self.notifier.exited(self.mode, closes, now)
        if unresolved:
            self.notifier.warn(self.mode, "some legs had no resolvable fill price",
                               f"Unresolved: {unresolved}. P&L above excludes them.")
        self.state.data.setdefault("history", []).append(
            {"decision_date": pos["decision_date"], "entry_date": pos["entry_date"],
             "exit_date": now.date().isoformat(), "closes": closes})
        self.state.set_position(None)

    def _resolve_exit_fill(self, leg: dict) -> float | None:
        try:
            probe = Order(id=leg["exit_order_id"], symbol=leg["symbol"], side="sell",
                          qty=0, style=leg.get("exit_style", "auction"), tif="opg", status="new")
            o = self.trader.refresh_order(probe)
            return float(o.filled_avg_price) if o.filled_avg_price else None
        except Exception as exc:                              # noqa: BLE001
            log.warning("could not refresh exit order: %s", exc)
            return None

    # ------------------------------------------------------------------------- retry
    def _retry(self, label: str, fn: Callable, *args):
        """Up to `max_retries` attempts with linear backoff, then alert and give up.

        Giving up is deliberate. Retrying an order indefinitely across the MOC cutoff would either
        get rejected or, worse, land as a different kind of fill than intended.
        """
        last = None
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                return fn(*args)
            except (BrokerError, Exception) as exc:            # noqa: BLE001
                last = exc
                log.warning("%s attempt %d/%d failed: %s", label, attempt,
                            self.cfg.max_retries, exc)
                if attempt < self.cfg.max_retries:
                    _time.sleep(self.cfg.retry_backoff_s * attempt)
        self.notifier.error(self.mode, f"{label} failed after {self.cfg.max_retries} attempts",
                            f"Last error: {last}")
        return None


def setup_logging(path: Path, verbose: bool = False) -> None:
    fmt = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    try:
        handlers.append(logging.FileHandler(path))
    except OSError:
        pass
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format=fmt, handlers=handlers, force=True)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


# ============================================================================== CLI

def build(verbose: bool = False) -> tuple[Config, "Engine"]:
    from config import load
    from personal.trader import build_trader

    cfg = load()
    setup_logging(cfg.log_path, verbose)
    engine = Engine(cfg, build_trader(cfg), Notifier.from_config(cfg),
                    FomcCalendar(cfg.cache_path), State(cfg.state_path))
    return cfg, engine


def cmd_status(args) -> int:
    from core.analytics import summary_table

    cfg, eng = build(args.verbose)
    print("\n  CONFIGURATION")
    for k, v in cfg.redacted().items():
        print(f"    {k:>14}: {v}")

    print("\n  ACCOUNT")
    try:
        for k, v in eng.trader.describe_account().items():
            print(f"    {k:>14}: {v}")
    except Exception as exc:                                  # noqa: BLE001
        print(f"    unavailable: {exc}")

    today = eng.now_fn().date()
    is_eve, decision = eng._is_eve(today)
    print(f"\n  TODAY ({today})")
    print(f"    {'FOMC eve':>14}: {'YES -> decision ' + str(decision) if is_eve else 'no'}")
    print(f"    {'market open':>14}: {eng.trader.is_market_open()}")

    print("\n  UPCOMING")
    for u in eng.calendar.upcoming(today, 6, eng.trader.is_trading_day):
        eve = u["eve"].isoformat() if u["eve"] else "?"
        print(f"    decision {u['decision']}  eve {eve}  (in {u['days_away']}d)")

    print("\n  RESEARCH (computed from core/data/fomc_eve_returns.csv)")
    print("    " + summary_table().to_string().replace("\n", "\n    "))
    print(f"\n  OPEN POSITION: {eng.state.position or 'none'}\n")
    return 0


def cmd_check(args) -> int:
    _, eng = build(args.verbose)
    day = date.fromisoformat(args.date) if args.date else eng.now_fn().date()
    is_eve, decision = eng._is_eve(day)
    print(f"{day}: trading_day={eng.trader.is_trading_day(day)}  fomc_eve={is_eve}"
          + (f"  decision={decision}" if decision else ""))
    return 0 if is_eve else 1


def cmd_calendar(args) -> int:
    _, eng = build(args.verbose)
    dates = eng.calendar.refresh() if args.refresh else eng.calendar.dates()
    today = eng.now_fn().date()
    print(f"{len(dates)} decision dates cached ({dates[0]} .. {dates[-1]})")
    nxt = eng.calendar.next_decision(today)
    for d in [x for x in dates if x.year >= today.year - 1]:
        eve = eng.calendar.eve_of(d, eng.trader.is_trading_day)
        flag = "  <-- next" if d == nxt else ""
        print(f"  {d:%Y-%m-%d %a}   eve {eve:%Y-%m-%d %a}{flag}" if eve else f"  {d}")
    return 0


def cmd_notify_test(args) -> int:
    cfg, eng = build(args.verbose)
    ok = eng.notifier.send(f"✅ *Fed Drift Suite test* — {cfg.mode} mode, "
                           f"{eng.now_fn():%Y-%m-%d %H:%M} ET. Channels are working.")
    print(f"configured={eng.notifier.configured}  delivered={ok}")
    if not eng.notifier.configured:
        print("  (no channel configured: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)")
    return 0 if ok or not eng.notifier.configured else 1


def cmd_once(args) -> int:
    """Force one phase now, ignoring its time window. Respects DRY_RUN."""
    _, eng = build(args.verbose)
    handlers = {p.name: p.handler for p in eng.phases()}
    if args.phase not in handlers:
        print(f"unknown phase {args.phase!r}; choose from {sorted(handlers)}")
        return 2
    now = eng.now_fn()
    print(f"running phase {args.phase!r} at {now:%Y-%m-%d %H:%M:%S %Z}")
    handlers[args.phase](now)
    print(f"position now: {eng.state.position}")
    for msg in eng.notifier.sent:
        print("-" * 60, msg, sep="\n")
    return 0


def cmd_run(args) -> int:
    cfg, eng = build(args.verbose)
    print(f"starting Fed Drift personal runner in {cfg.mode} mode; Ctrl-C to stop")
    if cfg.is_live_money:
        print("  *** LIVE MONEY *** orders will be real")
    try:
        eng.run_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    from config import ConfigError

    p = argparse.ArgumentParser(prog="personal/runner.py",
                               description="Layer 1 — personal FOMC-eve trader (Alpaca + Telegram)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("status", help="config, account, calendar, research table").set_defaults(
        fn=cmd_status)
    c = sub.add_parser("check", help="is a given date FOMC eve?")
    c.add_argument("--date"); c.set_defaults(fn=cmd_check)
    c = sub.add_parser("calendar", help="list decision dates and their eves")
    c.add_argument("--refresh", action="store_true"); c.set_defaults(fn=cmd_calendar)
    sub.add_parser("notify-test", help="send one Telegram message").set_defaults(fn=cmd_notify_test)
    c = sub.add_parser("once", help="force one phase now")
    c.add_argument("--phase", required=True); c.set_defaults(fn=cmd_once)
    sub.add_parser("run", help="the daemon").set_defaults(fn=cmd_run)

    args = p.parse_args(argv)
    if not getattr(args, "fn", None):
        args.fn, args.verbose = cmd_run, getattr(args, "verbose", False)   # bare invocation runs it
    try:
        return args.fn(args)
    except ConfigError as exc:
        print(f"\nCONFIG ERROR: {exc}\n", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
