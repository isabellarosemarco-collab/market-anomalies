"""
FastAPI application: authentication, rate limiting, and the webhook worker's lifecycle.

AUTHENTICATION is a single `X-API-KEY` header, compared with `secrets.compare_digest`. Not `==`:
string comparison returns early on the first differing byte, and the timing difference is measurable
across a network, so a plain `==` on a secret leaks it one character at a time. A key buys read access
to signals and research only — this service never touches a client's brokerage account and holds no
credential that could, which is the single most important security property of the product and the
reason it can be sold to a fund at all.

KEYS COME FROM THE ENVIRONMENT (`API_KEYS=key:ClientName,...`), never from a file in the image and
never from the repo. ARCC's guidance on secret handling is explicit that plaintext values sitting
where a broader audience than the owner can read them is the failure mode to design against; an env
var injected by the platform's secret store keeps the value out of the image, out of git, and out of
logs. The service logs only a four-character fingerprint and the client label, never the key.

NO KEYS CONFIGURED means every authenticated route returns 503, not 200. An API that falls open when
its configuration is missing is worse than one that fails, because nothing visibly breaks.

RATE LIMITING is a per-key sliding window, in process. That is honest about what it is: it protects
this instance from a runaway client loop, and it resets on redeploy and does not coordinate across
replicas. One `$5/month` box is exactly the deployment it fits.
"""
from __future__ import annotations

import asyncio
import logging
import os
import secrets
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime

import pytz
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

import config as config_mod
from commercial.endpoints import SignalService, router
from commercial.market_days import is_trading_day
from commercial.webhook_worker import EveWatcher, SubscriptionStore, WebhookDispatcher
from core import analytics
from core.calendar import FomcCalendar

log = logging.getLogger("feddrift.api")
ET = pytz.timezone(config_mod.MARKET_TZ)

DESCRIPTION = """
Signals and research for the FOMC-eve overnight drift: buy the index ETF at the closing auction on
the session before a scheduled Federal Reserve decision, sell at the next opening auction.

**What this is.** An eight-trades-a-year overlay with a net expectancy around +12 to +19bp per trade
depending on the index, positive on a 1996-2017 holdout that was never used to design the rule.

**What this is not.** A standalone return stream. Eight trades a year caps annualized Sharpe near 1.2
by arithmetic, and the return on capital reserved year-round is around 1.6%/yr. `/v1/research/metrics`
returns those limits, and the tests this hypothesis failed, in the same response as the results.

Authenticate with `X-API-KEY`. Not investment advice.
"""


# ---------------------------------------------------------------------------- limiter

class RateLimiter:
    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = {}

    def check(self, key: str) -> tuple[bool, int]:
        if self.per_minute <= 0:
            return True, 0
        now = time.monotonic()
        q = self._hits.setdefault(key, deque())
        while q and now - q[0] > 60.0:
            q.popleft()
        if len(q) >= self.per_minute:
            return False, int(60 - (now - q[0])) + 1
        q.append(now)
        return True, 0


# ------------------------------------------------------------------------------ auth

def fingerprint(key: str) -> str:
    return f"{key[:4]}…{len(key)}c" if len(key) >= 8 else "short-key"


# Declared as a security scheme rather than a plain Header parameter so it appears in the OpenAPI
# document under `securitySchemes`. That is what puts the "Authorize" button in /docs and lets a
# generated client library carry the key automatically — /docs is the client documentation, so a key
# field repeated on every endpoint is a worse product than one login box.
# auto_error=False because we own the 401-vs-403-vs-503 distinction below; its default would collapse
# "no key" and "wrong key" into one status and would answer 403 even when the service is unconfigured.
api_key_scheme = APIKeyHeader(
    name="X-API-KEY", auto_error=False,
    description="Your client key. Sent on every /v1 request. Contact the operator to be issued one.")


async def require_api_key(request: Request,
                          x_api_key: str | None = Security(api_key_scheme)) -> str:
    """Validate the key, then rate-limit it. Returns the client label for handlers/logs."""
    keys: dict[str, str] = request.app.state.api_keys
    if not keys:
        raise HTTPException(503, "Service not configured: no API keys are set. The operator must "
                                "set API_KEYS=<key>:<ClientName> in the environment.")
    if not x_api_key:
        raise HTTPException(401, "Missing X-API-KEY header.")

    label = None
    for candidate, name in keys.items():
        # compare_digest against every key, and do not break early, so neither the validity of a key
        # nor its position in the list is inferable from response timing.
        if secrets.compare_digest(x_api_key, candidate):
            label = name
    if label is None:
        log.warning("rejected request with key %s to %s", fingerprint(x_api_key), request.url.path)
        raise HTTPException(403, "Invalid API key.")

    ok, retry = request.app.state.limiter.check(x_api_key)
    if not ok:
        raise HTTPException(429, f"Rate limit exceeded ({request.app.state.limiter.per_minute}/min). "
                                f"Retry in {retry}s.")
    request.state.client = label
    return label


# --------------------------------------------------------------------------- app setup

def create_app(cfg=None, now_fn=None, start_worker: bool | None = None) -> FastAPI:
    """Build the app. `now_fn` and `start_worker` exist so the test suite can drive the clock and
    skip the background loop; production passes neither."""
    cfg = cfg or config_mod.Config()
    if start_worker is None:
        start_worker = os.getenv("ENABLE_WEBHOOK_WORKER", "true").strip().lower() in (
            "1", "true", "yes", "on")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            cov = app.state.calendar.coverage()
            log.info("calendar: %d dates %s..%s", cov["count"], cov["first"], cov["last"])
        except Exception as exc:                          # noqa: BLE001
            log.error("calendar unavailable at startup: %s", exc)
        try:
            log.info("research: %d trades, recommended %s",
                     analytics.provenance().n_trades, analytics.recommended_symbol()[0])
        except Exception as exc:                          # noqa: BLE001
            log.error("research dataset unavailable: %s — /v1/research/* will return 503", exc)
        log.info("api keys configured: %d", len(app.state.api_keys))

        task = None
        if start_worker:
            task = asyncio.create_task(
                app.state.watcher.run_forever(lambda: datetime.now(ET)))
        try:
            yield
        finally:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    app = FastAPI(
        title=cfg.api_title, version="1.0.0", description=DESCRIPTION, lifespan=lifespan,
        docs_url="/docs" if cfg.allow_open_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if cfg.allow_open_docs else None)

    calendar = FomcCalendar(cfg.cache_path, auto_fetch=False)
    service = SignalService(calendar, now_fn=now_fn)
    subs = SubscriptionStore(cfg.subs_path)
    dispatcher = WebhookDispatcher(subs, timeout_s=cfg.webhook_timeout_s,
                                  max_attempts=cfg.webhook_max_attempts)
    watcher = EveWatcher(
        dispatcher, signal_fn=service.signals_for_all,
        is_eve_fn=lambda d: calendar.is_eve(d, is_trading_day)[0])

    app.state.cfg = cfg
    app.state.calendar = calendar
    app.state.service = service
    app.state.subs = subs
    app.state.dispatcher = dispatcher
    app.state.watcher = watcher
    app.state.api_keys = cfg.api_keys
    app.state.limiter = RateLimiter(cfg.rate_limit_per_min)

    # Every /v1 route requires a key. Applied at the router so a new endpoint is authenticated by
    # default — the opposite arrangement means one forgotten decorator is a public data leak.
    app.include_router(router, dependencies=[Depends(require_api_key)])

    @app.get("/health", include_in_schema=False)
    def health() -> dict:
        """Unauthenticated on purpose: a load balancer cannot hold an API key."""
        return {"status": "ok", "service": "fed-drift-suite", "version": "1.0.0"}

    @app.get("/", include_in_schema=False)
    def root() -> dict:
        return {"service": cfg.api_title, "docs": "/docs" if cfg.allow_open_docs else "disabled",
                "auth": "X-API-KEY header", "endpoints": [
                    "/v1/status", "/v1/signal/next", "/v1/research/metrics",
                    "/v1/research/trades", "/v1/webhooks/subscribe"]}

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Never surface a traceback: it can echo a config value or a path. Log it, return an id.
        ref = secrets.token_hex(4)
        log.exception("unhandled error ref=%s path=%s", ref, request.url.path)
        return JSONResponse(status_code=500,
                            content={"detail": f"internal error (ref {ref})"})

    return app


app = create_app()
