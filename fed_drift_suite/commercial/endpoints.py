"""
The four sellable endpoints, plus the two the first support email would otherwise ask for.

    GET  /v1/status                  is today an active FOMC window?
    GET  /v1/signal/next             next meeting, asset, expected drift, historical drawdown
    GET  /v1/research/metrics        the 28-year holdout stats, for due diligence
    GET  /v1/research/trades         every individual trade, so the stats can be recomputed
    POST /v1/webhooks/subscribe      register an endpoint for the 15:00 ET signal push
    GET  /v1/webhooks/subscriptions  what you have registered, and whether delivery is working

WHY /v1/research/trades EXISTS, when it was not asked for. `/v1/research/metrics` returns a Sharpe
ratio and a t-statistic. Any diligence team worth selling to will ask for the underlying trades, and
the answer "we don't provide those" ends the conversation. Shipping the 675 rows makes every published
statistic checkable in about ten minutes, which is a stronger sales argument than the Sharpe itself.

The signal is computed, never stored as a literal: `core.analytics` reads the trade file, so the bp
figure this API sells is the same object the personal trader sizes against.
"""
from __future__ import annotations

import io
from datetime import date, datetime

import pytz
from fastapi import APIRouter, HTTPException, Query, Request, Response

from config import MARKET_TZ
from core import analytics
from core.calendar import FomcCalendar
from core.models import (NextSignal, ResearchMetrics, StatusResponse, TradeRow, Window,
                         WebhookSubscribeRequest, WebhookSubscription)
from commercial.market_days import is_trading_day

router = APIRouter(prefix="/v1")
ET = pytz.timezone(MARKET_TZ)


# --------------------------------------------------------------------------- service

class SignalService:
    """Calendar + analytics, assembled into the shapes the endpoints return."""

    def __init__(self, calendar: FomcCalendar, now_fn=None):
        self.calendar = calendar
        self._now_fn = now_fn or (lambda: datetime.now(ET))

    def now(self) -> datetime:
        return self._now_fn()

    # -------------------------------------------------------------------- /v1/status
    def status(self) -> StatusResponse:
        now = self.now()
        today = now.date()
        is_eve, eve_decision = self.calendar.is_eve(today, is_trading_day)
        is_decision = self.calendar.is_decision_day(today)
        nxt = self.calendar.next_decision(today)

        if is_eve:
            window, msg = Window.EVE, (
                f"ACTIVE — today is FOMC eve. Entry is the closing auction at 16:00 ET; "
                f"the decision lands {eve_decision:%a %d %b} at 14:00 ET.")
        elif is_decision:
            window, msg = Window.DECISION, (
                "ACTIVE — today is a decision day. Any position from last night exits at the "
                "09:30 ET opening cross. Do not hold through the 14:00 statement.")
        elif nxt is None:
            window, msg = Window.NONE, (
                "No future decision dates in the calendar cache. Refresh required — see "
                "/v1/status again after the operator reruns the calendar fetch.")
        else:
            window, msg = Window.IDLE, (
                f"Idle. No position. Next decision {nxt:%a %d %b %Y}, "
                f"{(nxt - today).days} days away.")

        return StatusResponse(
            as_of=now, market_date=today, window=window,
            is_fomc_eve=is_eve, is_decision_day=is_decision,
            decision_date=eve_decision if is_eve else (today if is_decision else None),
            next_decision_date=nxt,
            days_to_next_decision=(nxt - today).days if nxt else None,
            message=msg)

    # --------------------------------------------------------------- /v1/signal/next
    def next_signal(self, symbol: str | None = None) -> NextSignal:
        today = self.now().date()
        sym = symbol or analytics.recommended_symbol()[0]
        decision = self.calendar.next_decision(today)
        if decision is None:
            raise HTTPException(503, "calendar cache holds no future decision dates")

        eve = self.calendar.eve_of(decision, is_trading_day)
        exp = analytics.expectancy(sym)
        days = (eve - today).days if eve else None

        if days == 0:
            action = (f"ENTER TODAY. Buy {sym} in the 16:00 ET closing auction (MOC order must be "
                      f"submitted before 15:50 ET).")
        elif days == 1:
            action = f"Stage tomorrow. {sym} entry is tomorrow's closing auction."
        else:
            action = (f"No action. Next entry {eve:%a %d %b %Y} at the close"
                      + (f", {days} days away." if days is not None else "."))

        return NextSignal(
            as_of=today, decision_date=decision, eve_date=eve, days_until_eve=days,
            action=action, symbol=sym,
            entry="Market-on-close, FOMC eve. Submit before 15:50 ET; the fill is the 16:00 cross.",
            exit="Market-on-open, decision day. Submit before 09:28 ET; the fill is the 09:30 cross. "
                 "Flat well before the 14:00 ET statement.",
            expected_drift_bp=exp["net_bp_full"],
            expected_drift_bp_holdout=exp["net_bp_holdout"],
            historical_win_rate=exp["win_rate"],
            historical_max_drawdown_pct=exp["max_drawdown_pct"],
            n_historical_trades=exp["n_trades"],
            permutation_p=exp["permutation_p"],
            confidence=analytics.confidence_label(sym),
            caveats=analytics.caveats(sym))

    def signals_for_all(self) -> dict[str, NextSignal]:
        """One signal per covered symbol — what the webhook broadcast fans out."""
        return {s: self.next_signal(s) for s in analytics.symbols()}


# -------------------------------------------------------------------------- accessors

def service(request: Request) -> SignalService:
    return request.app.state.service


def store(request: Request):
    return request.app.state.subs


# ------------------------------------------------------------------------- endpoints

@router.get("/status", response_model=StatusResponse, summary="Is today an active FOMC window?")
def get_status(request: Request) -> StatusResponse:
    return service(request).status()


@router.get("/signal/next", response_model=NextSignal, summary="Next signal with expected drift")
def get_next_signal(
    request: Request,
    symbol: str | None = Query(
        default=None, description="SPY, QQQ or IWM. Omit for the recommended asset."),
) -> NextSignal:
    if symbol is not None:
        symbol = symbol.upper()
        if symbol not in analytics.symbols():
            raise HTTPException(400, f"unknown symbol {symbol!r}; covered: {analytics.symbols()}")
    return service(request).next_signal(symbol)


@router.get("/research/metrics", response_model=ResearchMetrics,
            summary="28-year holdout statistics for due diligence")
def get_research_metrics() -> ResearchMetrics:
    """Full-sample, holdout (1996-2017) and design-period (2018+) statistics per symbol.

    The holdout block is the one that matters: the rule was specified using 2018+ data, so only the
    earlier period is genuinely out-of-sample. `known_negative_results` lists the tests this
    hypothesis failed, which is the part that tells you how much searching produced the winner.
    """
    try:
        return analytics.metrics()
    except FileNotFoundError as exc:
        raise HTTPException(503, f"research dataset not built: {exc}") from exc


@router.get("/research/trades", response_model=None,
            summary="Every individual trade, so the stats can be recomputed")
def get_research_trades(
    symbol: str | None = Query(default=None),
    sample: str | None = Query(default=None, description="'holdout' or 'design'"),
    format: str = Query(default="json", pattern="^(json|csv)$"),
):
    try:
        df = analytics.trades()
    except FileNotFoundError as exc:
        raise HTTPException(503, f"research dataset not built: {exc}") from exc

    if symbol:
        df = df[df["symbol"] == symbol.upper()]
    if sample:
        if sample not in ("holdout", "design"):
            raise HTTPException(400, "sample must be 'holdout' or 'design'")
        df = df[df["sample"] == sample]
    if df.empty:
        raise HTTPException(404, "no trades match that filter")

    if format == "csv":
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        return Response(buf.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="fomc_trades.csv"'})
    return [TradeRow(**r) for r in df.to_dict("records")]


@router.post("/webhooks/subscribe", response_model=WebhookSubscription, status_code=201,
             summary="Register an endpoint for the 15:00 ET signal push")
def subscribe(request: Request, body: WebhookSubscribeRequest) -> WebhookSubscription:
    """Registers an HTTPS endpoint. On each FOMC eve at 15:00 ET it receives a POST with the signal.

    If you supply a `secret`, every delivery carries `X-Signature: sha256=<hmac>` over
    `"{X-Timestamp}." + body`. Verify it before acting — an unauthenticated POST claiming to be a
    trading signal is worth nothing. The secret is stored so deliveries can be signed, is never
    returned by any endpoint, and cannot be retrieved; rotate by subscribing again and deleting the
    old subscription.
    """
    sub = store(request).add(body)
    return sub


@router.get("/webhooks/subscriptions", response_model=list[WebhookSubscription],
            summary="Your registered endpoints and their delivery health")
def list_subscriptions(request: Request) -> list[WebhookSubscription]:
    return store(request).list()


@router.delete("/webhooks/{sub_id}", summary="Deactivate a subscription")
def unsubscribe(request: Request, sub_id: str) -> dict:
    if not store(request).deactivate(sub_id):
        raise HTTPException(404, f"no subscription {sub_id}")
    return {"id": sub_id, "active": False}


@router.post("/webhooks/{sub_id}/test", summary="Send a test delivery now")
async def test_delivery(request: Request, sub_id: str) -> dict:
    st = store(request)
    sub = st.get(sub_id)
    if sub is None:
        raise HTTPException(404, f"no subscription {sub_id}")
    signal = service(request).next_signal(sub.symbols[0] if sub.symbols else None)
    ok, detail = await request.app.state.dispatcher.deliver(sub, signal, event="test")
    return {"id": sub_id, "delivered": ok, "detail": detail}


@router.get("/webhooks/last-broadcast", summary="Result of the most recent scheduled broadcast")
def last_broadcast(request: Request) -> dict:
    watcher = getattr(request.app.state, "watcher", None)
    if watcher is None or watcher.last_result is None:
        return {"status": "no broadcast has run in this process"}
    return watcher.last_result
