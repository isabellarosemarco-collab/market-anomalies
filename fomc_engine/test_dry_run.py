"""
Dry-run test suite: simulated clock, in-memory broker, no network, no API keys.

The centrepiece is `test_full_pipeline_on_fomc_eve`, which walks a simulated clock through a real
FOMC eve and the following decision morning and asserts on every artifact the engine produces:
orders, persisted state, fill prices, notification text and overnight P&L.

Two tests exist specifically to pin down the broker-cutoff corrections:

  * `test_spec_time_1558_uses_market_fallback` — the originally specified 15:59 entry lands AFTER
    Alpaca's 15:50 MOC cutoff, so it must degrade to a flagged market order rather than silently
    submitting an order the broker would reject.
  * `test_config_rejects_entry_after_moc_cutoff` — configuring the auction entry past the cutoff is
    a startup error, not a runtime surprise on the one day per quarter that matters.

Run:  pytest test_dry_run.py -v        or      python test_dry_run.py
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, time
from pathlib import Path

import pytest
import pytz

sys.path.insert(0, str(Path(__file__).resolve().parent))

from alpaca_trader import BrokerError, DryRunTrader                       # noqa: E402
from config import MOC_CUTOFF, MOO_CUTOFF, Config, ConfigError           # noqa: E402
from fed_calendar import FomcCalendar, _dedupe_adjacent, _resolve        # noqa: E402
from notifier import CaptureNotifier                                     # noqa: E402
from scheduler import ET, Engine, State                                   # noqa: E402

# A real scheduled meeting: Wednesday 28 Jan 2026. Eve is Tuesday 27 Jan 2026.
DECISION = date(2026, 1, 28)
EVE = date(2026, 1, 27)
OTHER_DECISION = date(2026, 3, 18)


def et(d: date, hh: int, mm: int, ss: int = 0) -> datetime:
    """Eastern-local aware datetime. pytz.localize picks the right DST offset for the date."""
    return ET.localize(datetime(d.year, d.month, d.day, hh, mm, ss))


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    p = tmp_path / "fomc_dates.json"
    p.write_text(json.dumps([DECISION.isoformat(), OTHER_DECISION.isoformat()]))
    return p


def make_cfg(tmp_path: Path, cache: Path, **over) -> Config:
    base = dict(
        api_key="", api_secret="", paper=True, dry_run=True,
        symbols=["QQQ"], sizing_mode="fixed_cash", sizing_value=10_000.0,
        order_style="auction", retry_backoff_s=0.0,
        state_path=tmp_path / "state.json", cache_path=cache, log_path=tmp_path / "engine.log",
    )
    base.update(over)
    cfg = Config(**base)
    cfg.validate()
    return cfg


def build_engine(tmp_path: Path, cache: Path, now: datetime | None = None, **over):
    cfg = make_cfg(tmp_path, cache, **over)
    trader = DryRunTrader(cfg)
    notifier = CaptureNotifier(dry_run=True)
    calendar = FomcCalendar(cfg.cache_path, auto_fetch=False)
    clock = {"now": now or et(EVE, 15, 45)}
    engine = Engine(cfg, trader, notifier, calendar, State(cfg.state_path),
                    now_fn=lambda: clock["now"])
    return engine, trader, notifier, clock


# --------------------------------------------------------------------- calendar parsing

@pytest.mark.parametrize("mon,day,year,expect", [
    ("January", "27-28", 2026, date(2026, 1, 28)),          # decision is the LAST day
    ("Jan/Feb", "31-1", 2023, date(2023, 2, 1)),            # abbreviated straddle
    ("April/May", "30-1", 2019, date(2019, 5, 1)),
    ("July", "31-August 1", 2012, date(2012, 8, 1)),        # month name inside the day field
    ("December", "17-18", 2024, date(2024, 12, 18)),
    ("Dec/Jan", "31-1", 2019, date(2020, 1, 1)),            # year rolls forward
    ("March", "18", 2003, date(2003, 3, 18)),
])
def test_resolve_meeting_labels(mon, day, year, expect):
    assert _resolve(mon, day, year) == expect


def test_two_day_meeting_split_across_headings_collapses():
    """'September 15 Meeting' + 'September 16 Meeting' is one meeting; only the 16th decides."""
    got = _dedupe_adjacent([date(2003, 9, 15), date(2003, 9, 16), date(2003, 10, 28)])
    assert got == [date(2003, 9, 16), date(2003, 10, 28)]


def test_eve_is_previous_trading_day(cache):
    cal = FomcCalendar(cache, auto_fetch=False)
    weekday = lambda d: d.weekday() < 5
    assert cal.eve_of(DECISION, weekday) == EVE
    assert cal.is_eve(EVE, weekday) == (True, DECISION)
    assert cal.is_eve(DECISION, weekday)[0] is False
    assert cal.is_eve(date(2026, 1, 26), weekday)[0] is False


def test_eve_skips_a_holiday(cache):
    """If the day before a decision is closed, eve walks back to the previous open session."""
    cal = FomcCalendar(cache, auto_fetch=False)
    closed = {EVE}
    is_open = lambda d: d.weekday() < 5 and d not in closed
    assert cal.eve_of(DECISION, is_open) == date(2026, 1, 26)   # Monday


# ------------------------------------------------------------------------- sizing

def test_fixed_cash_sizing_rounds_down_to_whole_shares(tmp_path, cache):
    eng, trader, _, _ = build_engine(tmp_path, cache)
    trader.set_price("QQQ", 501.0)
    s = trader.size_position(eng.cfg, "QQQ", equity=100_000, buying_power=200_000)
    assert s.qty_whole == 19                      # 10000/501 = 19.96 -> 19, never up
    assert s.notional_whole == pytest.approx(19 * 501.0)
    assert s.qty_fractional == pytest.approx(19.960080, abs=1e-5)


def test_sizing_capped_by_buying_power(tmp_path, cache):
    eng, trader, _, _ = build_engine(tmp_path, cache)
    s = trader.size_position(eng.cfg, "QQQ", equity=5_000, buying_power=3_000)
    assert s.qty_whole == 6                       # 3000/500
    assert "buying power" in s.reason


def test_pct_equity_sizing_splits_across_symbols(tmp_path, cache):
    eng, trader, _, _ = build_engine(tmp_path, cache, symbols=["QQQ", "IWM"],
                                     sizing_mode="pct_equity", sizing_value=0.5)
    s = trader.size_position(eng.cfg, "QQQ", equity=100_000, buying_power=200_000)
    assert s.target_notional == pytest.approx(25_000)     # 50% equity / 2 symbols
    assert s.qty_whole == 50


# ------------------------------------------------------------- the full trade lifecycle

def test_full_pipeline_on_fomc_eve(tmp_path, cache):
    eng, trader, notif, clock = build_engine(tmp_path, cache)

    # --- 15:00 eve: detect and stage
    clock["now"] = et(EVE, 15, 0)
    assert eng.tick() == ["stage"]
    assert "FOMC Eve detected" in notif.sent[0]
    assert "28 Jan 2026" in notif.sent[0]
    assert eng.state.position is None              # staging must not take exposure

    # --- 15:45 eve: MOC entry inside the cutoff
    clock["now"] = et(EVE, 15, 45)
    assert eng.tick() == ["entry"]
    assert len(trader.orders) == 1
    o = trader.orders[0]
    assert (o.side, o.style, o.tif, o.qty) == ("buy", "auction", "cls", 20.0)
    assert o.degraded is False
    pos = eng.state.position
    assert pos["decision_date"] == DECISION.isoformat()
    assert pos["legs"][0] == dict(symbol="QQQ", qty=20.0, order_id=o.id, style="auction",
                                  degraded=False, entry_price=500.0, ref_price=500.0)
    assert "Entry submitted" in notif.sent[1] and "BUY 20 QQQ" in notif.sent[1]

    # --- 16:05 eve: confirm the cross, persist the real fill
    clock["now"] = et(EVE, 16, 5)
    assert eng.tick() == ["entry_confirm"]
    assert eng.state.position["legs"][0]["entry_price"] == 500.0
    assert "Entry confirmed at the close" in notif.sent[2]

    # --- overnight gap up, then 09:25 decision day: MOO exit before the 09:28 cutoff
    trader.set_price("QQQ", 505.0)
    clock["now"] = et(DECISION, 9, 25)
    assert eng.tick() == ["exit"]
    sell = trader.orders[1]
    assert (sell.side, sell.style, sell.tif, sell.qty) == ("sell", "auction", "opg", 20.0)
    assert trader.get_position("QQQ") is None

    # --- 09:31: the fallback poll fires but is a no-op, because the MOO already flattened us
    clock["now"] = et(DECISION, 9, 31)
    assert eng.tick() == ["exit_fallback"]
    assert len(trader.orders) == 2                  # no second sell

    # --- 09:36: P&L reported and state cleared
    clock["now"] = et(DECISION, 9, 36)
    assert eng.tick() == ["exit_confirm"]
    msg = notif.sent[-1]
    assert "Position closed" in msg
    assert "SOLD 20 QQQ @ $505.00" in msg
    assert "+100.00" in msg and "+1.00%" in msg     # 20 sh x $5 on a $10,000 basis
    assert eng.state.position is None
    assert eng.state.data["history"][0]["closes"][0]["pnl"] == pytest.approx(100.0)


def test_losing_night_reports_negative_pnl(tmp_path, cache):
    eng, trader, notif, clock = build_engine(tmp_path, cache)
    clock["now"] = et(EVE, 15, 45)
    eng.tick()
    trader.set_price("QQQ", 493.0)
    clock["now"] = et(DECISION, 9, 25)
    eng.tick()
    clock["now"] = et(DECISION, 9, 36)
    eng.tick()
    assert "-140.00" in notif.sent[-1] and "-1.40%" in notif.sent[-1]


def test_multi_symbol_entry_and_exit(tmp_path, cache):
    eng, trader, notif, clock = build_engine(
        tmp_path, cache, symbols=["QQQ", "IWM"], sizing_value=10_000.0)
    clock["now"] = et(EVE, 15, 45)
    eng.tick()
    assert {o.symbol for o in trader.orders} == {"QQQ", "IWM"}
    assert [l["qty"] for l in eng.state.position["legs"]] == [10.0, 21.0]   # 5000/500, 5000/230
    trader.set_price("QQQ", 510.0)
    trader.set_price("IWM", 232.0)
    clock["now"] = et(DECISION, 9, 25)
    eng.tick()
    clock["now"] = et(DECISION, 9, 36)
    eng.tick()
    # 10 x 10.00 + 21 x 2.00 = 142.00
    assert "+142.00" in notif.sent[-1]


# ------------------------------------------------- the spec's 15:59 entry, and the cutoffs

def test_spec_time_1558_uses_market_fallback(tmp_path, cache):
    """A 15:58 start is past the 15:50 MOC cutoff, so the engine must degrade — and say so.

    This is the literal timing from the original specification. It cannot produce an auction fill,
    and the notification has to make that visible rather than implying a clean cross.
    """
    eng, trader, notif, clock = build_engine(tmp_path, cache, now=et(EVE, 15, 58))
    assert eng.tick() == ["entry_fallback"]
    o = trader.orders[0]
    assert (o.style, o.tif, o.degraded) == ("market", "day", True)
    assert any("MOC window missed" in m for m in notif.sent)
    assert "market order, not the auction cross" in notif.sent[-1]


def test_config_rejects_entry_after_moc_cutoff(tmp_path, cache):
    with pytest.raises(ConfigError, match="MOC cutoff"):
        make_cfg(tmp_path, cache, entry_at=time(15, 59))


def test_config_rejects_exit_after_moo_cutoff(tmp_path, cache):
    with pytest.raises(ConfigError, match="MOO cutoff"):
        make_cfg(tmp_path, cache, exit_at=time(9, 31))


def test_cutoff_constants_match_broker_rules():
    assert (MOC_CUTOFF, MOO_CUTOFF) == (time(15, 50), time(9, 28))


def test_auction_order_rejects_fractional_quantity(tmp_path, cache):
    eng, trader, _, _ = build_engine(tmp_path, cache)
    with pytest.raises(BrokerError, match="whole-share"):
        trader.submit("QQQ", "buy", 0.4, "auction")


# ------------------------------------------------------------------- negative paths

def test_does_nothing_on_a_non_eve_day(tmp_path, cache):
    eng, trader, notif, clock = build_engine(tmp_path, cache, now=et(date(2026, 2, 10), 15, 45))
    eng.tick()
    assert trader.orders == []
    assert eng.state.position is None
    assert notif.sent == []


def test_no_order_when_market_reports_closed(tmp_path, cache):
    eng, trader, notif, clock = build_engine(tmp_path, cache)
    trader._market_open = False
    clock["now"] = et(EVE, 15, 45)
    eng.tick()
    assert trader.orders == []
    assert eng.state.position is None
    assert any("market reported closed" in m for m in notif.sent)


def test_phase_runs_once_per_day(tmp_path, cache):
    """Idempotence: a restart loop inside the entry window must not stack positions."""
    eng, trader, _, clock = build_engine(tmp_path, cache, now=et(EVE, 15, 46))
    assert eng.tick() == ["entry"]
    assert eng.tick() == []
    assert eng.tick() == []
    assert len(trader.orders) == 1


def test_state_survives_process_restart(tmp_path, cache):
    """The morning exit acts on persisted state, not on a re-derived calendar decision."""
    eng, trader, _, clock = build_engine(tmp_path, cache, now=et(EVE, 15, 45))
    eng.tick()
    saved = json.loads((tmp_path / "state.json").read_text())
    assert saved["open_position"]["legs"][0]["symbol"] == "QQQ"

    eng2, trader2, notif2, clock2 = build_engine(tmp_path, cache, now=et(DECISION, 9, 25))
    trader2.positions = dict(trader.positions)          # broker still holds it
    trader2.set_price("QQQ", 502.5)
    assert eng2.state.position is not None              # loaded from disk
    assert eng2.tick() == ["exit"]
    clock2["now"] = et(DECISION, 9, 36)
    eng2.tick()
    assert "+50.00" in notif2.sent[-1]


def test_skips_entry_when_a_position_is_already_open(tmp_path, cache):
    eng, trader, notif, clock = build_engine(tmp_path, cache, now=et(EVE, 15, 0))
    eng.state.set_position({"decision_date": DECISION.isoformat(), "entry_date": "2026-01-20",
                            "style": "auction", "legs": [{"symbol": "QQQ", "qty": 5}]})
    eng.tick()
    assert any("already open" in m for m in notif.sent)
    clock["now"] = et(EVE, 15, 45)
    eng.tick()
    assert trader.orders == []


def test_exit_fallback_flattens_a_stuck_position(tmp_path, cache):
    """MOO submitted but unfilled: at 09:31 the engine must flatten at market and flag it."""
    eng, trader, notif, clock = build_engine(tmp_path, cache, now=et(EVE, 15, 45))
    eng.tick()
    clock["now"] = et(DECISION, 9, 31)
    trader.set_price("QQQ", 506.0)
    assert eng.tick() == ["exit_fallback"]
    assert any("still holding after the open" in m for m in notif.sent)
    assert trader.orders[-1].style == "market"
    assert trader.get_position("QQQ") is None


# ---------------------------------------------------------------------- retry behaviour

def test_retries_then_succeeds(tmp_path, cache):
    eng, trader, notif, clock = build_engine(tmp_path, cache, now=et(EVE, 15, 45),
                                             max_retries=3)
    trader.fail_next = 2                     # fail twice, succeed on the third attempt
    eng.tick()
    assert len(trader.orders) == 1
    assert eng.state.position is not None
    assert not any("ACTION REQUIRED" in m for m in notif.sent)


def test_gives_up_after_max_retries_and_alerts(tmp_path, cache):
    eng, trader, notif, clock = build_engine(tmp_path, cache, now=et(EVE, 15, 45),
                                             max_retries=3)
    trader.fail_next = 99
    eng.tick()
    assert trader.orders == []
    assert eng.state.position is None        # no phantom position recorded
    assert any("failed after 3 attempts" in m for m in notif.sent)
    assert any("entry produced no orders" in m for m in notif.sent)


def test_tick_survives_a_raising_phase(tmp_path, cache):
    """A phase that explodes must be recorded as run and must alert, not kill the daemon."""
    eng, _, notif, clock = build_engine(tmp_path, cache, now=et(EVE, 15, 45))
    eng._entry = lambda now: (_ for _ in ()).throw(RuntimeError("boom"))
    assert eng.tick() == ["entry"]
    assert any("boom" in m for m in notif.sent)
    assert eng.state.done(EVE, "entry")


# -------------------------------------------------------------- safety and time handling

def test_live_money_requires_explicit_confirmation(tmp_path, cache):
    with pytest.raises(ConfigError, match="LIVE_TRADING_CONFIRMED"):
        make_cfg(tmp_path, cache, paper=False, dry_run=False,
                 api_key="k", api_secret="s", live_confirmed="")


def test_live_money_allowed_with_confirmation(tmp_path, cache):
    cfg = make_cfg(tmp_path, cache, paper=False, dry_run=False, api_key="k", api_secret="s",
                   live_confirmed="I_UNDERSTAND_THE_RISK")
    assert cfg.is_live_money and cfg.base_url == "https://api.alpaca.markets"


def test_paper_endpoint_selected_by_default(tmp_path, cache):
    cfg = make_cfg(tmp_path, cache)
    assert cfg.base_url == "https://paper-api.alpaca.markets"
    assert cfg.redacted()["mode"] == "DRY RUN"


def test_eastern_time_handles_dst_both_ways():
    """Wall-clock ET comparisons are correct across both transitions without offset arithmetic."""
    est = et(date(2026, 1, 27), 15, 45)      # standard time
    edt = et(date(2026, 6, 16), 15, 45)      # daylight time
    assert est.utcoffset().total_seconds() / 3600 == -5
    assert edt.utcoffset().total_seconds() / 3600 == -4
    assert est.time() == edt.time() == time(15, 45)


def test_calendar_refuses_to_invent_dates_without_network(tmp_path):
    from fed_calendar import CalendarError
    cal = FomcCalendar(tmp_path / "missing.json", auto_fetch=False)
    with pytest.raises(CalendarError):
        cal.dates()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--tb=short"]))
