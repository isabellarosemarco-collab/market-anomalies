"""
End-to-end tests for both layers.

LAYER 1 is verified by mocking the clock onto a known FOMC eve and walking the daemon through every
phase it would hit in a real session — stage, entry, confirm, then the next morning's exit — asserting
that mock Alpaca orders are submitted and mock Telegram alerts are produced. There is no network and
no broker account: `DryRunTrader` is a deterministic in-memory exchange and `CaptureNotifier` records
messages instead of sending them, so P&L assertions are exact.

LAYER 2 is verified through FastAPI's TestClient against every endpoint, including a test that
enumerates the app's own route table and asserts each `/v1` path rejects an unauthenticated request.
Written that way on purpose: asserting auth on a hand-written list of paths passes forever after
someone adds a fifth endpoint and forgets it, which is precisely the bug worth catching.

    python -m pytest test_suite.py -v
"""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import date, datetime, time
from pathlib import Path

import pytest
import pytz

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as config_mod                                         # noqa: E402
from commercial import market_days, webhook_worker                   # noqa: E402
from commercial.api import create_app                                # noqa: E402
from commercial.endpoints import SignalService                       # noqa: E402
from config import Config                                           # noqa: E402
from core import analytics                                          # noqa: E402
from core.calendar import FomcCalendar                               # noqa: E402
from core.models import WebhookSubscribeRequest                      # noqa: E402
from personal.notifier import CaptureNotifier                        # noqa: E402
from personal.runner import Engine, State                            # noqa: E402
from personal.trader import DryRunTrader                             # noqa: E402

ET = pytz.timezone("America/New_York")

DECISION = date(2026, 1, 28)          # Wednesday
EVE = date(2026, 1, 27)               # Tuesday
DECISIONS_2026 = [date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29), date(2026, 6, 17),
                  date(2026, 7, 29), date(2026, 9, 16), date(2026, 10, 28), date(2026, 12, 9)]

TEST_KEY = "fds_test_key_0123456789abcdef"
OTHER_KEY = "fds_other_key_fedcba9876543210"


def et(d: date, hh: int, mm: int, ss: int = 0) -> datetime:
    return ET.localize(datetime(d.year, d.month, d.day, hh, mm, ss))


# ------------------------------------------------------------------------------ fixtures

@pytest.fixture(scope="session")
def cache(tmp_path_factory) -> Path:
    """A calendar cache with known dates, so tests do not depend on a live scrape."""
    p = tmp_path_factory.mktemp("cal") / "fomc_dates.json"
    p.write_text(json.dumps([d.isoformat() for d in
                             [date(2025, 12, 10)] + DECISIONS_2026]))
    return p


def make_cfg(tmp_path: Path, cache: Path, **over) -> Config:
    base = dict(
        api_key="", api_secret="", paper=True, dry_run=True,
        symbols=["QQQ"], sizing_mode="fixed_cash", sizing_value=10_000.0,
        order_style="auction", retry_backoff_s=0.0,
        api_keys_raw=f"{TEST_KEY}:TestFund,{OTHER_KEY}:OtherFund",
        rate_limit_per_min=1000,
        state_path=tmp_path / "state.json", subs_path=tmp_path / "subs.json",
        cache_path=cache, log_path=tmp_path / "test.log")
    base.update(over)
    cfg = Config(**base)
    cfg.validate()
    return cfg


def build_engine(tmp_path, cache, now=None, **over):
    cfg = make_cfg(tmp_path, cache, **over)
    trader = DryRunTrader(cfg)
    notifier = CaptureNotifier(dry_run=True)
    calendar = FomcCalendar(cfg.cache_path, auto_fetch=False)
    clock = {"now": now or et(EVE, 14, 0)}
    engine = Engine(cfg, trader, notifier, calendar, State(cfg.state_path),
                    now_fn=lambda: clock["now"])
    return engine, trader, notifier, clock


@pytest.fixture
def client(tmp_path, cache):
    """TestClient with the clock pinned to an idle day and the background worker off."""
    from fastapi.testclient import TestClient
    cfg = make_cfg(tmp_path, cache)
    app = create_app(cfg, now_fn=lambda: et(date(2026, 2, 10), 11, 0), start_worker=False)
    with TestClient(app) as c:
        yield c


def auth(**extra) -> dict:
    return {"X-API-KEY": TEST_KEY, **extra}


# =============================================================== LAYER 1 — personal
# The mandated dry-run clock simulation.

def test_dry_run_fomc_eve_submits_orders_and_sends_alerts(tmp_path, cache):
    """Walk a full FOMC eve and the following morning with a mocked clock.

    Asserts the four things that have to be true on the night: staging happens, an auction buy is
    submitted before the broker's 15:50 cutoff, the fill is read back, and the position is sold at
    the next open with P&L reported.
    """
    eng, trader, notif, clock = build_engine(tmp_path, cache, now=et(EVE, 15, 0))

    # 15:00 — detection and staging alert.
    assert eng.tick() == ["stage"]
    assert any("FOMC Eve detected" in m for m in notif.sent)
    assert any("Wed 28 Jan 2026" in m for m in notif.sent)
    assert trader.orders == []                       # staging does not trade

    # 15:45 — the MOC goes in. QQQ stub price 500, $10k target -> 20 shares.
    clock["now"] = et(EVE, 15, 45)
    assert eng.tick() == ["entry"]
    assert len(trader.orders) == 1
    buy = trader.orders[0]
    assert (buy.side, buy.symbol, buy.qty) == ("buy", "QQQ", 20)
    assert (buy.style, buy.tif, buy.degraded) == ("auction", "cls", False)
    assert any("Entry submitted" in m for m in notif.sent)
    assert eng.state.position is not None

    # 16:05 — read the cross back and persist the real fill price.
    clock["now"] = et(EVE, 16, 5)
    assert eng.tick() == ["entry_confirm"]
    assert any("Entry confirmed at the close" in m for m in notif.sent)
    assert eng.state.position["legs"][0]["entry_price"] == 500.0

    # Overnight move: +1.4%, the kind of night the strategy is buying.
    trader.set_price("QQQ", 507.0)

    # 09:25 — MOO submitted before the 09:28 cutoff.
    clock["now"] = et(DECISION, 9, 25)
    assert eng.tick() == ["exit"]
    sell = trader.orders[1]
    assert (sell.side, sell.qty, sell.style, sell.tif) == ("sell", 20, "auction", "opg")
    assert trader.get_position("QQQ") is None

    # 09:31 — the fallback poll a real daemon makes; a no-op because we are already flat.
    clock["now"] = et(DECISION, 9, 31)
    assert eng.tick() == ["exit_fallback"]
    assert len(trader.orders) == 2                   # crucially, no second sell

    # 09:36 — realized P&L.
    clock["now"] = et(DECISION, 9, 36)
    assert eng.tick() == ["exit_confirm"]
    closed = notif.sent[-1]
    assert "Position closed" in closed
    assert "+140.00" in closed and "+1.40%" in closed
    assert eng.state.position is None


def test_staging_alert_quotes_shared_expectancy(tmp_path, cache):
    """The dollar figure on the phone comes from core.analytics, not a literal in the notifier."""
    eng, _, notif, _ = build_engine(tmp_path, cache, now=et(EVE, 15, 0))
    eng.tick()
    staged = next(m for m in notif.sent if "FOMC Eve detected" in m)
    bp = analytics.expectancy("QQQ")["net_bp_holdout"]
    assert f"{10_000 * bp / 1e4:+,.2f}" in staged      # out-of-sample number, not full-sample
    assert "One night's variance is many times this" in staged


def test_literal_spec_time_1559_degrades_to_a_market_order(tmp_path, cache):
    """'Buy at 3:59 PM MOC' is not implementable: Alpaca rejects CLS at or after 15:50.

    The engine still trades, but as a plain market order, and says so — the fill pays a spread the
    backtest did not pay, and round-trip friction is already about a fifth of the edge.
    """
    eng, trader, notif, _ = build_engine(tmp_path, cache, now=et(EVE, 15, 58))
    assert eng.tick() == ["entry_fallback"]
    o = trader.orders[0]
    assert (o.style, o.tif, o.degraded) == ("market", "day", True)
    assert any("MOC window missed" in m for m in notif.sent)
    assert "market order, not the auction cross" in notif.sent[-1]


def test_non_eve_days_do_nothing(tmp_path, cache):
    eng, trader, notif, clock = build_engine(tmp_path, cache, now=et(date(2026, 2, 10), 15, 0))
    for hh, mm in [(15, 0), (15, 45), (15, 58), (16, 5)]:
        clock["now"] = et(date(2026, 2, 10), hh, mm)
        eng.tick()
    assert trader.orders == []
    assert notif.sent == []
    assert eng.state.position is None


def test_state_survives_a_restart_between_entry_and_exit(tmp_path, cache):
    eng, trader, _, clock = build_engine(tmp_path, cache, now=et(EVE, 15, 45))
    eng.tick()
    assert eng.state.position is not None

    reloaded = State(eng.cfg.state_path)               # simulate the process dying overnight
    assert reloaded.position["legs"][0]["symbol"] == "QQQ"
    assert reloaded.done(EVE, "entry") is True


def test_multi_symbol_sizing_splits_the_target(tmp_path, cache):
    eng, trader, _, _ = build_engine(tmp_path, cache, now=et(EVE, 15, 45),
                                     symbols=["QQQ", "SPY"], sizing_value=12_000.0)
    eng.tick()
    qty = {o.symbol: o.qty for o in trader.orders}
    assert qty == {"QQQ": 12.0, "SPY": 10.0}           # $6k each at 500 and 600


def test_config_rejects_a_schedule_the_broker_would_reject():
    from config import ConfigError
    with pytest.raises(ConfigError, match="MOC cutoff"):
        Config(entry_at=time(15, 59), order_style="auction", dry_run=True).validate()
    with pytest.raises(ConfigError, match="MOO cutoff"):
        Config(exit_at=time(9, 31), order_style="auction", dry_run=True).validate()


def test_live_money_needs_two_independent_switches():
    from config import ConfigError
    with pytest.raises(ConfigError, match="Refusing to trade live money"):
        Config(paper=False, dry_run=False, api_key="k", api_secret="s",
               live_confirmed="").validate()
    Config(paper=False, dry_run=False, api_key="k", api_secret="s",
           live_confirmed="I_UNDERSTAND_THE_RISK").validate()


# =============================================================== LAYER 2 — commercial

def test_health_is_open_and_v1_is_not(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/v1/status").status_code == 401


def test_every_v1_route_enforces_the_api_key(client):
    """Enumerated from the app's own OpenAPI schema — its published surface — so a newly added
    endpoint cannot quietly skip auth. Auth runs as a router-level dependency, so it fires before
    the handler and a nonexistent path parameter still yields 401/403 rather than 404."""
    paths = client.app.openapi()["paths"]
    checked = []
    for path, ops in paths.items():
        if not path.startswith("/v1"):
            continue
        url = path.replace("{sub_id}", "sub_probe")
        for method in ops:
            if method.upper() in ("HEAD", "OPTIONS"):
                continue
            body = {} if method.upper() in ("POST", "PUT", "PATCH") else None
            r = client.request(method.upper(), url, json=body)
            assert r.status_code == 401, f"{method} {path} returned {r.status_code} with no key"
            r = client.request(method.upper(), url, headers={"X-API-KEY": "wrong"}, json=body)
            assert r.status_code == 403, f"{method} {path} accepted a bad key"
            checked.append(f"{method.upper()} {path}")

    assert len(checked) >= 9, checked
    for required in ("GET /v1/status", "GET /v1/signal/next", "GET /v1/research/metrics",
                     "POST /v1/webhooks/subscribe"):
        assert required in checked


def test_valid_key_is_accepted(client):
    assert client.get("/v1/status", headers=auth()).status_code == 200
    assert client.get("/v1/status", headers={"X-API-KEY": OTHER_KEY}).status_code == 200


def test_missing_configuration_fails_closed(tmp_path, cache):
    """No API_KEYS must mean 503, never an open API."""
    from fastapi.testclient import TestClient
    cfg = make_cfg(tmp_path, cache, api_keys_raw="")
    with TestClient(create_app(cfg, start_worker=False)) as c:
        r = c.get("/v1/status", headers=auth())
        assert r.status_code == 503
        assert "no API keys" in r.json()["detail"]


def test_rate_limit_returns_429(tmp_path, cache):
    from fastapi.testclient import TestClient
    cfg = make_cfg(tmp_path, cache, rate_limit_per_min=3)
    with TestClient(create_app(cfg, start_worker=False)) as c:
        codes = [c.get("/v1/status", headers=auth()).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]


def test_status_idle_eve_and_decision_day(tmp_path, cache):
    from fastapi.testclient import TestClient
    cfg = make_cfg(tmp_path, cache)

    def at(when):
        app = create_app(cfg, now_fn=lambda: when, start_worker=False)
        with TestClient(app) as c:
            return c.get("/v1/status", headers=auth()).json()

    idle = at(et(date(2026, 2, 10), 11, 0))
    assert (idle["window"], idle["is_fomc_eve"]) == ("idle", False)
    assert idle["next_decision_date"] == "2026-03-18"

    eve = at(et(EVE, 15, 0))
    assert (eve["window"], eve["is_fomc_eve"]) == ("eve", True)
    assert eve["decision_date"] == "2026-01-28"
    assert "ACTIVE" in eve["message"]

    dec = at(et(DECISION, 10, 0))
    assert (dec["window"], dec["is_decision_day"]) == ("decision", True)
    assert "09:30 ET opening cross" in dec["message"]


def test_signal_next_returns_drift_and_drawdown(client):
    s = client.get("/v1/signal/next", headers=auth()).json()
    assert s["symbol"] in ("SPY", "QQQ", "IWM")
    assert s["decision_date"] == "2026-03-18"
    assert s["eve_date"] == "2026-03-17"
    assert s["expected_drift_bp"] > 0
    assert s["expected_drift_bp_holdout"] > 0
    # Sizing must be against the smaller, out-of-sample figure.
    assert s["expected_drift_bp_holdout"] <= s["expected_drift_bp"]
    assert s["historical_max_drawdown_pct"] < 0
    assert 0 < s["historical_win_rate"] < 1
    assert s["n_historical_trades"] > 100
    assert s["caveats"] and any("holdout" in c for c in s["caveats"])
    assert "15:50" in s["entry"] and "09:28" in s["exit"]


def test_signal_next_accepts_a_symbol_and_rejects_nonsense(client):
    for sym in ("SPY", "QQQ", "IWM", "iwm"):
        r = client.get("/v1/signal/next", params={"symbol": sym}, headers=auth())
        assert r.status_code == 200
        assert r.json()["symbol"] == sym.upper()
    bad = client.get("/v1/signal/next", params={"symbol": "TSLA"}, headers=auth())
    assert bad.status_code == 400


def test_signal_on_eve_says_enter_today(tmp_path, cache):
    from fastapi.testclient import TestClient
    cfg = make_cfg(tmp_path, cache)
    app = create_app(cfg, now_fn=lambda: et(EVE, 14, 0), start_worker=False)
    with TestClient(app) as c:
        s = c.get("/v1/signal/next", headers=auth()).json()
    assert s["days_until_eve"] == 0
    assert s["action"].startswith("ENTER TODAY")
    assert "15:50" in s["action"]


def test_research_metrics_publishes_holdout_and_failures(client):
    m = client.get("/v1/research/metrics", headers=auth()).json()

    assert m["provenance"]["n_trades"] == 675
    assert m["provenance"]["holdout_end"] == "2017-12-31"
    assert set(m["symbols"]) == {"SPY", "QQQ", "IWM"}

    for sym, r in m["symbols"].items():
        for label in ("full", "holdout", "design"):
            s = r[label]
            assert s["n_trades"] > 0
            assert s["net_bp_mean"] > 0, f"{sym} {label} expectancy is not positive"
            assert 0 <= s["permutation_p"] <= 1
        # The claim that matters: positive on data never used to design the rule.
        assert r["holdout"]["net_bp_mean"] > 0
        # And the arithmetic ceiling, so nobody reads this as a 3-Sharpe strategy.
        assert r["full"]["sharpe_annualized"] < 1.3
        assert 6 <= r["full"]["trades_per_year"] <= 10

    assert m["recommended"] in ("SPY", "QQQ", "IWM")
    assert len(m["known_negative_results"]) >= 5
    assert any("FAILS" in n for n in m["known_negative_results"])
    assert any("LOSER" in n for n in m["known_negative_results"])
    assert len(m["structural_limits"]) >= 4


def test_research_metrics_matches_the_committed_trade_file(client):
    """The API must not be able to publish a statistic the shipped trades do not support."""
    import pandas as pd
    m = client.get("/v1/research/metrics", headers=auth()).json()
    csv = client.get("/v1/research/trades", params={"format": "csv"}, headers=auth())
    df = pd.read_csv(pd.io.common.StringIO(csv.text))

    for sym, r in m["symbols"].items():
        g = df[df["symbol"] == sym]
        assert r["full"]["n_trades"] == len(g)
        assert round(g["net_bp"].mean(), 2) == r["full"]["net_bp_mean"]
        h = g[g["sample"] == "holdout"]
        assert round(h["net_bp"].mean(), 2) == r["holdout"]["net_bp_mean"]


def test_research_trades_filters_and_formats(client):
    all_rows = client.get("/v1/research/trades", headers=auth()).json()
    assert len(all_rows) == 675
    assert set(all_rows[0]) >= {"symbol", "decision_date", "eve_date", "net_bp", "sample"}

    qqq = client.get("/v1/research/trades", params={"symbol": "QQQ"}, headers=auth()).json()
    assert len(qqq) == 220 and all(r["symbol"] == "QQQ" for r in qqq)

    holdout = client.get("/v1/research/trades",
                         params={"symbol": "QQQ", "sample": "holdout"}, headers=auth()).json()
    assert len(holdout) == 151
    assert all(r["decision_date"] <= "2017-12-31" for r in holdout)

    csv = client.get("/v1/research/trades", params={"symbol": "SPY", "format": "csv"},
                     headers=auth())
    assert csv.headers["content-type"].startswith("text/csv")
    assert len(csv.text.strip().splitlines()) == 246          # 245 trades + header

    assert client.get("/v1/research/trades", params={"symbol": "SPY", "sample": "nope"},
                      headers=auth()).status_code == 400


# ------------------------------------------------------------------------- webhooks

def test_webhook_subscribe_and_lifecycle(client):
    r = client.post("/v1/webhooks/subscribe", headers=auth(), json={
        "url": "https://client.example.com/hooks/fomc",
        "symbols": ["QQQ"], "secret": "a" * 32, "label": "Example Capital"})
    assert r.status_code == 201
    sub = r.json()
    assert sub["id"].startswith("sub_")
    assert sub["signed"] is True
    assert sub["active"] is True
    assert "secret" not in sub, "the API must never echo a shared secret back"

    listed = client.get("/v1/webhooks/subscriptions", headers=auth()).json()
    assert [s["id"] for s in listed] == [sub["id"]]
    assert all("secret" not in s for s in listed)

    assert client.delete(f"/v1/webhooks/{sub['id']}", headers=auth()).json()["active"] is False
    assert client.delete("/v1/webhooks/sub_nope", headers=auth()).status_code == 404


def test_webhook_subscribe_rejects_bad_input(client):
    plain_http = client.post("/v1/webhooks/subscribe", headers=auth(),
                             json={"url": "http://client.example.com/hook"})
    assert plain_http.status_code == 422
    assert "https" in plain_http.text

    short_secret = client.post("/v1/webhooks/subscribe", headers=auth(),
                               json={"url": "https://x.example.com/h", "secret": "tooshort"})
    assert short_secret.status_code == 422

    bad_symbol = client.post("/v1/webhooks/subscribe", headers=auth(),
                             json={"url": "https://x.example.com/h", "symbols": ["TSLA"]})
    assert bad_symbol.status_code == 422


def test_subscription_file_is_not_world_readable(tmp_path, cache, client):
    client.post("/v1/webhooks/subscribe", headers=auth(),
                json={"url": "https://x.example.com/h", "secret": "b" * 20})
    path = client.app.state.cfg.subs_path
    assert path.exists()
    assert oct(path.stat().st_mode)[-3:] == "600"
    # The secret is on disk because HMAC signing needs it; it must not be anywhere else.
    assert "b" * 20 in path.read_text()


def test_hmac_signature_roundtrip():
    body = b'{"event":"fomc_eve_signal"}'
    ts = "1769000000"
    sig = webhook_worker.sign(body, "s" * 32, ts)
    assert sig.startswith("sha256=")
    assert webhook_worker.verify(body, "s" * 32, ts, sig)
    assert not webhook_worker.verify(body, "s" * 32, "1769000001", sig)   # timestamp is signed
    assert not webhook_worker.verify(body + b" ", "s" * 32, ts, sig)      # body is signed
    assert not webhook_worker.verify(body, "x" * 32, ts, sig)             # wrong secret


class _StubResponse:
    def __init__(self, status_code: int):
        self.status_code = status_code


class _StubClient:
    """Stands in for httpx.AsyncClient. Captures posts; `status` drives the response."""
    captured: list[dict] = []
    status = 200

    def __init__(self, *a, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, content=None, headers=None):
        _StubClient.captured.append({"url": url, "content": content, "headers": headers})
        return _StubResponse(_StubClient.status)


@pytest.fixture
def stub_http(monkeypatch):
    import httpx
    _StubClient.captured = []
    _StubClient.status = 200
    monkeypatch.setattr(httpx, "AsyncClient", _StubClient)
    return _StubClient


def _dispatch_setup(tmp_path, cache):
    cfg = make_cfg(tmp_path, cache)
    store = webhook_worker.SubscriptionStore(cfg.subs_path)
    sub = store.add(WebhookSubscribeRequest(
        url="https://client.example.com/hook", symbols=["QQQ"], secret="k" * 32))
    dispatcher = webhook_worker.WebhookDispatcher(store, timeout_s=1.0, max_attempts=2)
    calendar = FomcCalendar(cfg.cache_path, auto_fetch=False)
    service = SignalService(calendar, now_fn=lambda: et(EVE, 15, 0))
    return store, sub, dispatcher, service


def test_dispatch_signs_the_delivery(tmp_path, cache, stub_http):
    store, sub, dispatcher, service = _dispatch_setup(tmp_path, cache)
    ok, detail = asyncio.run(dispatcher.deliver(sub, service.next_signal("QQQ")))
    assert (ok, detail) == (True, "200")

    sent = stub_http.captured[0]
    assert sent["url"] == "https://client.example.com/hook"
    h = sent["headers"]
    assert h["X-Event"] == "fomc_eve_signal"
    assert webhook_worker.verify(sent["content"], "k" * 32, h["X-Timestamp"], h["X-Signature"])

    payload = json.loads(sent["content"])
    assert payload["signal"]["symbol"] == "QQQ"
    assert payload["signal"]["days_until_eve"] == 0
    assert "secret" not in sent["content"].decode()

    # Idempotency key is stable for the same day, so a client-side timeout retry is detectable.
    assert h["X-Delivery-Id"] == webhook_worker.delivery_id(sub.id, "fomc_eve_signal", date.today())
    assert store.get(sub.id).last_delivery_status == "200"
    assert store.get(sub.id).consecutive_failures == 0


def test_dispatch_records_failures_and_stops_on_4xx(tmp_path, cache, stub_http):
    store, sub, dispatcher, service = _dispatch_setup(tmp_path, cache)
    stub_http.status = 404
    ok, detail = asyncio.run(dispatcher.deliver(sub, service.next_signal("QQQ")))
    assert ok is False and detail == "HTTP 404"
    assert len(stub_http.captured) == 1, "a 404 is the client's config, not worth retrying"
    assert store.get(sub.id).consecutive_failures == 1


def test_broadcast_fans_out_per_symbol(tmp_path, cache, stub_http):
    cfg = make_cfg(tmp_path, cache)
    store = webhook_worker.SubscriptionStore(cfg.subs_path)
    store.add(WebhookSubscribeRequest(url="https://a.example.com/h", symbols=["QQQ"]))
    store.add(WebhookSubscribeRequest(url="https://b.example.com/h", symbols=["SPY", "IWM"]))
    inactive = store.add(WebhookSubscribeRequest(url="https://c.example.com/h", symbols=["QQQ"]))
    store.deactivate(inactive.id)

    dispatcher = webhook_worker.WebhookDispatcher(store, timeout_s=1.0, max_attempts=1)
    service = SignalService(FomcCalendar(cfg.cache_path, auto_fetch=False),
                            now_fn=lambda: et(EVE, 15, 0))
    result = asyncio.run(dispatcher.broadcast(service.signals_for_all()))

    assert (result["attempted"], result["delivered"], result["failed"]) == (2, 2, 0)
    by_url = {c["url"]: json.loads(c["content"])["signal"]["symbol"] for c in stub_http.captured}
    assert by_url == {"https://a.example.com/h": "QQQ", "https://b.example.com/h": "SPY"}


def test_eve_watcher_fires_once_inside_the_window(tmp_path, cache, stub_http):
    store, sub, dispatcher, service = _dispatch_setup(tmp_path, cache)
    calendar = FomcCalendar(service.calendar.cache_path, auto_fetch=False)
    watcher = webhook_worker.EveWatcher(
        dispatcher, signal_fn=service.signals_for_all,
        is_eve_fn=lambda d: calendar.is_eve(d, market_days.is_trading_day)[0])

    assert asyncio.run(watcher.tick(et(EVE, 14, 59))) is None       # before the window
    fired = asyncio.run(watcher.tick(et(EVE, 15, 2)))               # inside it
    assert fired["delivered"] == 1
    assert asyncio.run(watcher.tick(et(EVE, 15, 20))) is None       # already sent today
    assert asyncio.run(watcher.tick(et(date(2026, 2, 10), 15, 0))) is None   # not an eve
    assert len(stub_http.captured) == 1


def test_webhook_test_delivery_endpoint(client, stub_http):
    sub = client.post("/v1/webhooks/subscribe", headers=auth(),
                      json={"url": "https://x.example.com/h", "secret": "z" * 20}).json()
    r = client.post(f"/v1/webhooks/{sub['id']}/test", headers=auth())
    assert r.json() == {"id": sub["id"], "delivered": True, "detail": "200"}
    assert json.loads(stub_http.captured[0]["content"])["event"] == "test"
    assert client.post("/v1/webhooks/sub_nope/test", headers=auth()).status_code == 404


# ------------------------------------------------------------------------ market days

def test_trading_day_calendar():
    assert market_days.is_trading_day(date(2026, 1, 27))            # ordinary Tuesday
    assert not market_days.is_trading_day(date(2026, 1, 25))        # Sunday
    assert not market_days.is_trading_day(date(2026, 1, 19))        # MLK Jr Day
    assert not market_days.is_trading_day(date(2026, 4, 3))         # Good Friday
    assert not market_days.is_trading_day(date(2026, 7, 3))         # Jul 4 is a Saturday
    assert not market_days.is_trading_day(date(2026, 6, 19))        # Juneteenth
    assert market_days.is_trading_day(date(2021, 6, 18))            # not yet observed in 2021
    assert not market_days.is_trading_day(date(2022, 6, 20))        # Sunday Jun 19 -> Monday
    assert market_days.is_trading_day(date(2021, 12, 31))           # Jan 1 2022 was a Saturday
    assert market_days.easter(2026) == date(2026, 4, 5)


def test_eve_walks_back_over_a_holiday(cache):
    """A decision day whose previous session is a holiday must move the entry, not skip it."""
    cal = FomcCalendar(cache, auto_fetch=False)
    eve = cal.eve_of(date(2026, 4, 6), market_days.is_trading_day)   # Monday after Good Friday
    assert eve == date(2026, 4, 2)                                    # Thursday


# ------------------------------------------------------------------------------- shared

def test_both_layers_read_the_same_expectancy(tmp_path, cache, client):
    """The dollar figure in the phone alert and the bp figure sold to clients are one number."""
    api = client.get("/v1/signal/next", params={"symbol": "QQQ"}, headers=auth()).json()
    cfg = make_cfg(tmp_path, cache)
    sizing = DryRunTrader(cfg).size_position(cfg, "QQQ", equity=100_000, buying_power=200_000)
    assert sizing.expected_net_bp == api["expected_drift_bp_holdout"]
    assert sizing.historical_win_rate == api["historical_win_rate"]


def test_recommendation_refuses_to_rank_noise():
    sym, why = analytics.recommended_symbol()
    holdouts = {s: analytics.research(s).holdout.net_bp_mean for s in analytics.symbols()}
    top = max(holdouts.values())
    tied = [s for s, v in holdouts.items() if top - v <= analytics.TIE_BAND_BP]
    if len(tied) > 1:
        assert "TIE" in why, "a sub-2bp holdout gap is noise and must be disclosed as such"
        assert sym in tied
