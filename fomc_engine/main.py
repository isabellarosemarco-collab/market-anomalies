"""
CLI entrypoint.

    python main.py status                 config, account, next decisions, open position
    python main.py check [--date Y-M-D]   is that date FOMC eve?
    python main.py calendar [--refresh]   list decision dates (optionally re-scrape)
    python main.py notify-test            send one message through the configured channels
    python main.py once --phase entry     run a single phase now, ignoring its time window
    python main.py run                    the daemon
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime

from alpaca_trader import build_trader
from config import ConfigError, load
from fed_calendar import FomcCalendar
from notifier import Notifier
from scheduler import ET, Engine, State, setup_logging


def build(verbose: bool = False):
    cfg = load()
    setup_logging(cfg.log_path, verbose)
    trader = build_trader(cfg)
    notifier = Notifier.from_config(cfg)
    calendar = FomcCalendar(cfg.cache_path)
    engine = Engine(cfg, trader, notifier, calendar, State(cfg.state_path))
    return cfg, engine


def cmd_status(args) -> int:
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

    pos = eng.state.position
    print(f"\n  OPEN POSITION: {pos if pos else 'none'}\n")
    return 0


def cmd_check(args) -> int:
    _, eng = build(args.verbose)
    day = date.fromisoformat(args.date) if args.date else eng.now_fn().date()
    is_eve, decision = eng._is_eve(day)
    trading = eng.trader.is_trading_day(day)
    print(f"{day}: trading_day={trading}  fomc_eve={is_eve}"
          + (f"  decision={decision}" if decision else ""))
    return 0 if is_eve else 1


def cmd_calendar(args) -> int:
    _, eng = build(args.verbose)
    dates = eng.calendar.refresh() if args.refresh else eng.calendar.dates()
    today = eng.now_fn().date()
    shown = [d for d in dates if d.year >= today.year - 1]
    print(f"{len(dates)} decision dates cached ({dates[0]} .. {dates[-1]})")
    for d in shown:
        eve = eng.calendar.eve_of(d, eng.trader.is_trading_day)
        flag = "  <-- next" if d >= today and d == eng.calendar.next_decision(today) else ""
        print(f"  {d:%Y-%m-%d %a}   eve {eve:%Y-%m-%d %a}{flag}" if eve else f"  {d}")
    return 0


def cmd_notify_test(args) -> int:
    cfg, eng = build(args.verbose)
    ok = eng.notifier.send(
        f"✅ *FOMC engine test* — {cfg.redacted()['mode']} mode, "
        f"{datetime.now(ET):%Y-%m-%d %H:%M} ET. Channels are working.")
    print(f"configured={eng.notifier.configured}  delivered={ok}")
    if not eng.notifier.configured:
        print("  (no channel configured: set TELEGRAM_BOT_TOKEN+TELEGRAM_CHAT_ID "
              "or DISCORD_WEBHOOK_URL)")
    return 0 if ok or not eng.notifier.configured else 1


def cmd_once(args) -> int:
    """Force one phase now. Useful for rehearsal; respects DRY_RUN, ignores the time window."""
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
    mode = cfg.redacted()["mode"]
    print(f"starting FOMC Eve engine in {mode} mode; Ctrl-C to stop")
    if cfg.is_live_money:
        print("  *** LIVE MONEY *** orders will be real")
    try:
        eng.run_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="fomc_engine", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status").set_defaults(fn=cmd_status)
    c = sub.add_parser("check"); c.add_argument("--date"); c.set_defaults(fn=cmd_check)
    c = sub.add_parser("calendar"); c.add_argument("--refresh", action="store_true")
    c.set_defaults(fn=cmd_calendar)
    sub.add_parser("notify-test").set_defaults(fn=cmd_notify_test)
    c = sub.add_parser("once"); c.add_argument("--phase", required=True); c.set_defaults(fn=cmd_once)
    sub.add_parser("run").set_defaults(fn=cmd_run)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except ConfigError as exc:
        print(f"\nCONFIG ERROR: {exc}\n", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
