"""
Watch a full FOMC eve happen, right now, without waiting for the real one.

The next real decision is weeks away, which is an awkward thing to test. This script time-travels the
daemon's clock to the next actual FOMC eve on the calendar and plays the night minute by minute,
printing every Telegram message as it would arrive on your phone.

Nothing here is a mock-up of the output. It runs the real `Engine` with the real phase schedule and
the real notifier message bodies; only two things are substituted: the clock, and the broker (a
deterministic in-memory exchange instead of Alpaca). That is the same setup `test_suite.py` asserts
against, so what you see below is what the live daemon does.

    python demo.py                     # a good night: +1.4% overnight
    python demo.py --move -0.9         # a losing night
    python demo.py --symbols QQQ,IWM   # two positions, capital split between them
    python demo.py --late              # arrive at 15:58 and miss the auction window
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

import pytz

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config                                  # noqa: E402
from core import analytics                                 # noqa: E402
from core.calendar import FomcCalendar                     # noqa: E402
from personal.notifier import CaptureNotifier              # noqa: E402
from personal.runner import Engine, State                  # noqa: E402
from personal.trader import DryRunTrader                   # noqa: E402

ET = pytz.timezone("America/New_York")


def et(d: date, hh: int, mm: int) -> datetime:
    return ET.localize(datetime(d.year, d.month, d.day, hh, mm))


def rule(text: str = "") -> None:
    print(f"\n\033[2m{'─' * 78}\033[0m" + (f"\n{text}" if text else ""))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Simulate one FOMC eve end to end.")
    ap.add_argument("--symbols", default="QQQ", help="comma-separated, e.g. QQQ or QQQ,IWM")
    ap.add_argument("--cash", type=float, default=10_000.0, help="total dollars to deploy")
    ap.add_argument("--move", type=float, default=1.4,
                    help="overnight move in percent (try a negative number)")
    ap.add_argument("--late", action="store_true",
                    help="start at 15:58, past Alpaca's 15:50 MOC cutoff")
    args = ap.parse_args(argv)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    tmp = Path(tempfile.mkdtemp(prefix="feddrift-demo-"))

    cfg = Config(
        api_key="", api_secret="", dry_run=True, paper=True,
        symbols=symbols, sizing_mode="fixed_cash", sizing_value=args.cash,
        order_style="auction", retry_backoff_s=0.0,
        state_path=tmp / "state.json", log_path=tmp / "demo.log")
    cfg.validate()

    calendar = FomcCalendar(cfg.cache_path, auto_fetch=False)
    trader = DryRunTrader(cfg)
    notifier = CaptureNotifier(dry_run=True)

    # The next real FOMC eve on the shipped calendar.
    today = datetime.now(ET).date()
    decision = calendar.next_decision(today + timedelta(days=1))
    eve = calendar.eve_of(decision, trader.is_trading_day)

    clock = {"now": et(eve, 15, 0)}
    engine = Engine(cfg, trader, notifier, calendar, State(cfg.state_path),
                    now_fn=lambda: clock["now"])

    print(f"\n\033[1mSimulating the next real FOMC eve\033[0m")
    print(f"  decision      {decision:%A %d %B %Y} (statement 14:00 ET)")
    print(f"  eve           {eve:%A %d %B %Y}  ← the night we trade")
    print(f"  symbols       {', '.join(symbols)} at ${args.cash:,.0f} total")
    print(f"  overnight     {args.move:+.2f}% (simulated)")
    print(f"  broker        simulated — no keys, no network, no orders leave this machine")

    # Each step is (label, time, what a real daemon would be doing then).
    steps = [
        ("stage",          (15, 0),  "scan the calendar, size the position, alert"),
        ("entry",          (15, 45), "submit market-on-close, before the 15:50 cutoff"),
        ("entry_fallback", (15, 58), "only fires if the MOC never went in"),
        ("entry_confirm",  (16, 5),  "read the closing cross back, record the fill"),
    ]
    if args.late:
        steps = [("late start", (15, 58), "you booted the daemon after the MOC cutoff")] + steps[2:]

    seen = 0
    for label, (hh, mm), what in steps:
        clock["now"] = et(eve, hh, mm)
        ran = engine.tick()
        if not ran:
            continue
        rule(f"\033[1m{hh:02d}:{mm:02d} ET\033[0m  ·  {', '.join(ran)}  \033[2m— {what}\033[0m")
        if len(notifier.sent) == seen:
            print("\n  \033[2m(no alert — this phase had nothing to do, which is the "
                  "normal case)\033[0m")
        for msg in notifier.sent[seen:]:
            print(f"\n{msg}")
        seen = len(notifier.sent)

    # Overnight. The position sits through the night; this is the only risk the strategy takes.
    for sym in symbols:
        trader.set_price(sym, trader.last_price(sym) * (1 + args.move / 100))
    rule(f"\033[2m…overnight… index moves {args.move:+.2f}%…\033[0m")

    for label, (hh, mm), what in [
            ("exit",          (9, 25), "submit market-on-open, before the 09:28 cutoff"),
            ("exit_fallback", (9, 31), "flatten anything still open after the bell"),
            ("exit_confirm",  (9, 36), "resolve fills, report realized P&L")]:
        clock["now"] = et(decision, hh, mm)
        ran = engine.tick()
        if not ran:
            continue
        rule(f"\033[1m{hh:02d}:{mm:02d} ET\033[0m  ·  {', '.join(ran)}  \033[2m— {what}\033[0m")
        if len(notifier.sent) == seen:
            print("\n  \033[2m(no alert — this phase had nothing to do, which is the "
                  "normal case)\033[0m")
        for msg in notifier.sent[seen:]:
            print(f"\n{msg}")
        seen = len(notifier.sent)

    rule()
    orders = [f"{o.side.upper()} {o.qty:g} {o.symbol} [{o.tif.upper()}]"
              + ("  ⚠ degraded" if o.degraded else "") for o in trader.orders]
    print(f"\033[1mOrders the broker received:\033[0m  {'  ·  '.join(orders)}")
    print(f"\033[1mPosition now:\033[0m  "
          f"{'flat — nothing held into the 14:00 statement' if engine.state.position is None else engine.state.position}")

    exp = analytics.expectancy(symbols[0])
    print(f"\n\033[2mFor scale: one night is noise. {symbols[0]}'s out-of-sample expectancy is "
          f"{exp['net_bp_holdout']:+.1f}bp "
          f"(${args.cash * exp['net_bp_holdout'] / 1e4:+,.2f} on ${args.cash:,.0f}), "
          f"won {100 * exp['win_rate']:.0f}% of {exp['n_trades']} trades.\n"
          f"The simulated move above is {abs(args.move) * 100 / max(exp['net_bp_holdout'], 1e-9):.0f}x "
          f"the edge — which is the actual shape of this trade.\033[0m")
    print(f"\n\033[2mstate + logs: {tmp}\033[0m\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
