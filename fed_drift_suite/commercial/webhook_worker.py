"""
Enterprise webhook delivery: subscription storage and the 15:00 ET dispatch loop.

WHAT A PAYING CLIENT ACTUALLY NEEDS FROM A WEBHOOK, and what each requirement forces:

  * Authenticity. A trading signal arriving from an unauthenticated POST is worthless — anyone could
    forge one. Deliveries carry `X-Signature: sha256=<hmac>` over the exact request body using the
    per-subscription secret, so the client can verify it came from us and was not modified.
  * Replay resistance. `X-Timestamp` is included and signed. Clients should reject deliveries whose
    timestamp is more than a few minutes old.
  * Idempotence. `X-Delivery-Id` is stable per (subscription, event, date), so a retry after a client
    timeout is recognisable as the same signal rather than a second trade.
  * At-least-once with backoff. Exponential retry, then the subscription is marked failing. A signal
    that silently fails to arrive is worse than no service at all, so repeated failure is recorded on
    the subscription where the client can see it via GET.

ON SECRET STORAGE. ARCC's guidance on secret handling (queried for this module) is that a secret must
not sit in plaintext anywhere a broader audience than its owner can read it, and should be resolved
from a secrets store rather than embedded. HMAC signing needs the original secret, so it cannot be
one-way hashed like a password. The compromise implemented here: the subscriptions file is written
with mode 0600, the secret is never returned by any endpoint, never logged, and never included in a
delivery body. For a real deployment, mount `SUBS_PATH` on an encrypted volume or replace
`SubscriptionStore` with a secrets-manager-backed implementation — the class is small and has one
responsibility precisely so it can be swapped.

The API key that authenticates a client to us is separate and comes from the environment
(`API_KEYS`), which is the injection pattern that guidance recommends over an on-disk file.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import secrets
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

from core.models import NextSignal, WebhookSubscribeRequest, WebhookSubscription

log = logging.getLogger("feddrift.webhooks")

SIGNAL_EVENT = "fomc_eve_signal"
MAX_CONSECUTIVE_FAILURES = 10          # after this the subscription is deactivated


class SubscriptionStore:
    """JSON-backed subscription registry. Atomic writes, 0600 on disk."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._data: dict[str, dict] = {}
        self.load()

    # ------------------------------------------------------------------ plumbing
    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            self._data = json.loads(self.path.read_text())
        except (ValueError, json.JSONDecodeError):
            log.error("subscriptions file %s is corrupt; starting empty", self.path)
            self._data = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, default=str))
        try:
            os.chmod(tmp, 0o600)        # before the rename, so the file is never briefly readable
        except OSError:
            pass
        tmp.replace(self.path)

    # -------------------------------------------------------------------- public
    def add(self, req: WebhookSubscribeRequest) -> WebhookSubscription:
        sub_id = f"sub_{uuid.uuid4().hex[:16]}"
        self._data[sub_id] = {
            "id": sub_id,
            "url": str(req.url),
            "symbols": list(req.symbols),
            "label": req.label,
            "secret": req.secret,                 # never leaves this file; see module docstring
            "created_at": datetime.now(timezone.utc).isoformat(),
            "active": True,
            "last_delivery_at": None,
            "last_delivery_status": None,
            "consecutive_failures": 0,
        }
        self.save()
        return self.get(sub_id)                   # type: ignore[return-value]

    def get(self, sub_id: str) -> WebhookSubscription | None:
        raw = self._data.get(sub_id)
        return self._public(raw) if raw else None

    def list(self, active_only: bool = False) -> list[WebhookSubscription]:
        vals = self._data.values()
        if active_only:
            vals = [v for v in vals if v.get("active")]
        return [self._public(v) for v in vals]

    def deactivate(self, sub_id: str) -> bool:
        if sub_id not in self._data:
            return False
        self._data[sub_id]["active"] = False
        self.save()
        return True

    def secret_of(self, sub_id: str) -> str | None:
        return (self._data.get(sub_id) or {}).get("secret")

    def record(self, sub_id: str, ok: bool, status: str) -> None:
        raw = self._data.get(sub_id)
        if not raw:
            return
        raw["last_delivery_at"] = datetime.now(timezone.utc).isoformat()
        raw["last_delivery_status"] = status
        raw["consecutive_failures"] = 0 if ok else raw.get("consecutive_failures", 0) + 1
        if raw["consecutive_failures"] >= MAX_CONSECUTIVE_FAILURES:
            raw["active"] = False
            log.error("subscription %s deactivated after %d consecutive failures",
                      sub_id, raw["consecutive_failures"])
        self.save()

    @staticmethod
    def _public(raw: dict) -> WebhookSubscription:
        """Strip the secret. The API must never be able to echo it back."""
        return WebhookSubscription(
            id=raw["id"], url=raw["url"], symbols=raw["symbols"], label=raw.get("label"),
            signed=bool(raw.get("secret")),
            created_at=raw["created_at"], active=raw.get("active", True),
            last_delivery_at=raw.get("last_delivery_at"),
            last_delivery_status=raw.get("last_delivery_status"),
            consecutive_failures=raw.get("consecutive_failures", 0))


# ------------------------------------------------------------------------- signing

def sign(body: bytes, secret: str, timestamp: str) -> str:
    """HMAC-SHA256 over timestamp + body, so the timestamp cannot be altered independently."""
    mac = hmac.new(secret.encode(), f"{timestamp}.".encode() + body, hashlib.sha256)
    return f"sha256={mac.hexdigest()}"


def verify(body: bytes, secret: str, timestamp: str, signature: str) -> bool:
    """Reference implementation for clients. Constant-time compare, never `==`."""
    return hmac.compare_digest(sign(body, secret, timestamp), signature)


def delivery_id(sub_id: str, event: str, day: date) -> str:
    """Stable per subscription/event/day, so a retry is identifiable as the same signal."""
    return hashlib.sha256(f"{sub_id}:{event}:{day.isoformat()}".encode()).hexdigest()[:32]


def new_secret() -> str:
    return secrets.token_urlsafe(32)


# ------------------------------------------------------------------------ dispatch

class WebhookDispatcher:
    def __init__(self, store: SubscriptionStore, timeout_s: float = 10.0,
                 max_attempts: int = 4):
        self.store = store
        self.timeout_s = timeout_s
        self.max_attempts = max_attempts

    async def deliver(self, sub: WebhookSubscription, signal: NextSignal,
                      event: str = SIGNAL_EVENT, day: date | None = None) -> tuple[bool, str]:
        import httpx

        day = day or date.today()
        payload = {
            "event": event,
            "subscription_id": sub.id,
            "sent_at": datetime.now(timezone.utc).isoformat(),
            "signal": json.loads(signal.model_dump_json()),
        }
        body = json.dumps(payload, separators=(",", ":")).encode()
        ts = str(int(datetime.now(timezone.utc).timestamp()))
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "fed-drift-suite/1.0",
            "X-Event": event,
            "X-Timestamp": ts,
            "X-Delivery-Id": delivery_id(sub.id, event, day),
        }
        secret = self.store.secret_of(sub.id)
        if secret:
            headers["X-Signature"] = sign(body, secret, ts)

        last = "not attempted"
        for attempt in range(1, self.max_attempts + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                    resp = await client.post(str(sub.url), content=body, headers=headers)
                if 200 <= resp.status_code < 300:
                    self.store.record(sub.id, True, f"{resp.status_code}")
                    return True, f"{resp.status_code}"
                last = f"HTTP {resp.status_code}"
                # 4xx other than 408/429 is the client's configuration, not a transient fault.
                if 400 <= resp.status_code < 500 and resp.status_code not in (408, 429):
                    break
            except Exception as exc:                       # noqa: BLE001
                last = f"{type(exc).__name__}: {exc}"
            if attempt < self.max_attempts:
                await asyncio.sleep(2 ** attempt)          # 2s, 4s, 8s
        log.warning("webhook %s failed: %s", sub.id, last)
        self.store.record(sub.id, False, last)
        return False, last

    async def broadcast(self, signal_for: dict[str, NextSignal],
                        event: str = SIGNAL_EVENT) -> dict:
        """Fan out to every active subscription, each getting the signal for its own symbol."""
        subs = self.store.list(active_only=True)
        tasks, targets = [], []
        for sub in subs:
            sym = next((s for s in sub.symbols if s in signal_for), None)
            if sym is None:
                continue
            targets.append(sub)
            tasks.append(self.deliver(sub, signal_for[sym], event))
        results = await asyncio.gather(*tasks, return_exceptions=True)

        ok = sum(1 for r in results if isinstance(r, tuple) and r[0])
        return {"subscriptions": len(subs), "attempted": len(targets), "delivered": ok,
                "failed": len(targets) - ok,
                "detail": [{"id": s.id, "result": (r[1] if isinstance(r, tuple) else str(r))}
                           for s, r in zip(targets, results)]}


# -------------------------------------------------------------------- the 15:00 loop

class EveWatcher:
    """Fires the broadcast once, at 15:00 ET, on FOMC eve.

    A window rather than an instant, and a per-day marker, for the same reason the personal runner
    works that way: a container that restarts at 15:04 must still send, and must not send twice.
    The marker is in memory because a redeploy mid-window sending a duplicate is a far smaller
    problem than a missed signal — and `X-Delivery-Id` makes the duplicate detectable client-side.
    """

    def __init__(self, dispatcher: WebhookDispatcher, signal_fn, is_eve_fn,
                 fire_at=(15, 0), window_minutes: int = 30, poll_s: float = 60.0):
        self.dispatcher = dispatcher
        self.signal_fn = signal_fn                  # () -> dict[symbol, NextSignal]
        self.is_eve_fn = is_eve_fn                  # (date) -> bool
        self.fire_at = fire_at
        self.window_minutes = window_minutes
        self.poll_s = poll_s
        self.sent_on: set[date] = set()
        self.last_result: dict | None = None

    def due(self, now_et: datetime) -> bool:
        h, m = self.fire_at
        start = h * 60 + m
        mins = now_et.hour * 60 + now_et.minute
        return (start <= mins < start + self.window_minutes
                and now_et.date() not in self.sent_on
                and self.is_eve_fn(now_et.date()))

    async def tick(self, now_et: datetime) -> dict | None:
        if not self.due(now_et):
            return None
        self.sent_on.add(now_et.date())              # mark first: a crash must not cause a re-send
        try:
            result = await self.dispatcher.broadcast(self.signal_fn())
        except Exception as exc:                     # noqa: BLE001
            log.exception("webhook broadcast failed")
            result = {"error": str(exc)}
        self.last_result = {"at": now_et.isoformat(), **result}
        log.info("FOMC eve broadcast: %s", result)
        return self.last_result

    async def run_forever(self, now_fn) -> None:
        log.info("webhook watcher started; fires at %02d:%02d ET on FOMC eve", *self.fire_at)
        while True:
            try:
                await self.tick(now_fn())
            except asyncio.CancelledError:
                raise
            except Exception:                        # noqa: BLE001
                log.exception("watcher tick failed")
            await asyncio.sleep(self.poll_s)
