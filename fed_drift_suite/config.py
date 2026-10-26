"""
One configuration object for both layers, loaded from `.env`.

Defaults are deliberately inert: dry run on, paper on, notifications off, API keys empty. A fresh
clone of this repo cannot place an order or charge a customer, and that has to be true by default
rather than by the operator remembering to check.

THE LIVE-MONEY GUARD needs two independent switches (`PAPER_TRADING=false` AND
`LIVE_TRADING_CONFIRMED=I_UNDERSTAND_THE_RISK`) because a single boolean is one typo, one bad merge
or one copied-from-a-blog-post `.env` away from trading real money by accident.

BROKER CUTOFFS ARE HARD FACTS, not preferences, and they are enforced here rather than discovered at
15:59 on the one day a quarter that matters:

  * Alpaca REJECTS market-on-close (TimeInForce.CLS) submitted at or after 15:50 ET.
  * Alpaca REJECTS market-on-open (TimeInForce.OPG) submitted at or after 09:28 ET.

So "buy at 3:59 PM MOC" is not implementable. Entry is staged at 15:45 with a plain-market fallback
at 15:57; exit is submitted pre-open at 09:25 with a market fallback at 09:31. A fallback fill is
tagged degraded everywhere it appears, because it pays a spread the research did not pay, and
round-trip friction is already a fifth of the edge.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"
MARKET_TZ = "America/New_York"

# Broker-imposed auction cutoffs. Not tunable — Alpaca rejects orders outside these windows.
MOC_CUTOFF = time(15, 50)
MOO_CUTOFF = time(9, 28)

LIVE_PHRASE = "I_UNDERSTAND_THE_RISK"


class ConfigError(RuntimeError):
    pass


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _num(name: str, default: float) -> float:
    raw = os.getenv(name)
    try:
        return float(raw) if raw not in (None, "") else default
    except ValueError:
        raise ConfigError(f"{name}={raw!r} is not a number")


def _hhmm(name: str, default: time) -> time:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        h, m = raw.strip().split(":")
        return time(int(h), int(m))
    except Exception:
        raise ConfigError(f"{name}={raw!r} is not HH:MM (24-hour, Eastern)")


def _csv(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if not raw:
        return list(default)
    return [p.strip().upper() for p in raw.split(",") if p.strip()]


@dataclass
class Config:
    # ---- broker credentials -------------------------------------------------
    api_key: str = field(default_factory=lambda: os.getenv("APCA_API_KEY_ID", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("APCA_API_SECRET_KEY", ""))
    paper: bool = field(default_factory=lambda: _flag("PAPER_TRADING", True))
    live_confirmed: str = field(default_factory=lambda: os.getenv("LIVE_TRADING_CONFIRMED", ""))
    dry_run: bool = field(default_factory=lambda: _flag("DRY_RUN", True))

    # ---- strategy -----------------------------------------------------------
    symbols: list[str] = field(default_factory=lambda: _csv("SYMBOLS", ["QQQ"]))
    sizing_mode: str = field(default_factory=lambda: os.getenv("SIZING_MODE", "fixed_cash"))
    sizing_value: float = field(default_factory=lambda: _num("SIZING_VALUE", 10_000.0))
    max_notional: float = field(default_factory=lambda: _num("MAX_NOTIONAL", 100_000.0))
    order_style: str = field(default_factory=lambda: os.getenv("ORDER_STYLE", "auction"))

    # ---- schedule (Eastern wall clock) --------------------------------------
    stage_at: time = field(default_factory=lambda: _hhmm("STAGE_AT", time(15, 0)))
    entry_at: time = field(default_factory=lambda: _hhmm("ENTRY_AT", time(15, 45)))
    entry_fallback_at: time = field(default_factory=lambda: _hhmm("ENTRY_FALLBACK_AT", time(15, 57)))
    entry_confirm_at: time = field(default_factory=lambda: _hhmm("ENTRY_CONFIRM_AT", time(16, 5)))
    exit_at: time = field(default_factory=lambda: _hhmm("EXIT_AT", time(9, 25)))
    exit_fallback_at: time = field(default_factory=lambda: _hhmm("EXIT_FALLBACK_AT", time(9, 31)))
    exit_confirm_at: time = field(default_factory=lambda: _hhmm("EXIT_CONFIRM_AT", time(9, 36)))

    # ---- notifications ------------------------------------------------------
    telegram_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))
    # Optional second channel. Telegram is the one the personal layer is built around; a Discord
    # webhook is here because it is the cheapest way to fan the same alerts out to a paid signal
    # group without touching this code again.
    discord_webhook: str = field(default_factory=lambda: os.getenv("DISCORD_WEBHOOK_URL", ""))

    # ---- commercial layer ---------------------------------------------------
    api_keys_raw: str = field(default_factory=lambda: os.getenv("API_KEYS", ""))
    api_title: str = field(default_factory=lambda: os.getenv("API_TITLE", "Fed Drift Signal API"))
    allow_open_docs: bool = field(default_factory=lambda: _flag("ALLOW_OPEN_DOCS", True))
    webhook_timeout_s: float = field(default_factory=lambda: _num("WEBHOOK_TIMEOUT_S", 10.0))
    webhook_max_attempts: int = field(
        default_factory=lambda: int(_num("WEBHOOK_MAX_ATTEMPTS", 4)))
    rate_limit_per_min: int = field(default_factory=lambda: int(_num("RATE_LIMIT_PER_MIN", 60)))

    # ---- operations ---------------------------------------------------------
    max_retries: int = field(default_factory=lambda: int(_num("MAX_RETRIES", 3)))
    retry_backoff_s: float = field(default_factory=lambda: _num("RETRY_BACKOFF_S", 5.0))
    poll_seconds: float = field(default_factory=lambda: _num("POLL_SECONDS", 30.0))
    state_path: Path = field(
        default_factory=lambda: Path(os.getenv("STATE_PATH", str(ROOT / "state.json"))))
    subs_path: Path = field(
        default_factory=lambda: Path(os.getenv("SUBS_PATH", str(ROOT / "subscriptions.json"))))
    cache_path: Path = field(
        default_factory=lambda: Path(os.getenv("FOMC_CACHE_PATH",
                                               str(ROOT / "core" / "data" / "fomc_dates.json"))))
    log_path: Path = field(
        default_factory=lambda: Path(os.getenv("LOG_PATH", str(ROOT / "fed_drift.log"))))

    # ------------------------------------------------------------------ derived
    @property
    def base_url(self) -> str:
        return PAPER_URL if self.paper else LIVE_URL

    @property
    def is_live_money(self) -> bool:
        return (not self.paper) and (not self.dry_run)

    @property
    def mode(self) -> str:
        if self.dry_run:
            return "DRY RUN"
        return "PAPER" if self.paper else "LIVE"

    @property
    def api_keys(self) -> dict[str, str]:
        """`API_KEYS=key1:ClientName,key2:OtherFund` -> {key: label}.

        A bare `key` with no label is accepted too. Keys are compared with `secrets.compare_digest`
        at request time, never logged, and only ever shown as a fingerprint.
        """
        out: dict[str, str] = {}
        for part in self.api_keys_raw.split(","):
            part = part.strip()
            if not part:
                continue
            key, _, label = part.partition(":")
            key = key.strip()
            if key:
                out[key] = label.strip() or "unnamed client"
        return out

    # --------------------------------------------------------------- validation
    def validate(self) -> "Config":
        if self.is_live_money and self.live_confirmed != LIVE_PHRASE:
            raise ConfigError(
                "Refusing to trade live money. PAPER_TRADING=false and DRY_RUN=false were both set, "
                f"so set LIVE_TRADING_CONFIRMED={LIVE_PHRASE} to confirm you meant it.")
        if not self.dry_run and not (self.api_key and self.api_secret):
            raise ConfigError(
                "DRY_RUN=false needs APCA_API_KEY_ID and APCA_API_SECRET_KEY.")
        if self.sizing_mode not in ("fixed_cash", "pct_equity"):
            raise ConfigError(f"SIZING_MODE must be fixed_cash or pct_equity, got "
                              f"{self.sizing_mode!r}")
        if self.sizing_mode == "pct_equity" and not 0 < self.sizing_value <= 1:
            raise ConfigError("pct_equity SIZING_VALUE must be a fraction in (0, 1]")
        if self.sizing_value <= 0:
            raise ConfigError("SIZING_VALUE must be positive")
        if self.order_style not in ("auction", "market"):
            raise ConfigError(f"ORDER_STYLE must be auction or market, got {self.order_style!r}")
        if not self.symbols:
            raise ConfigError("SYMBOLS is empty")

        if self.order_style == "auction":
            if self.entry_at >= MOC_CUTOFF:
                raise ConfigError(
                    f"ENTRY_AT={self.entry_at:%H:%M} is at or past Alpaca's MOC cutoff "
                    f"({MOC_CUTOFF:%H:%M} ET). The order would be rejected. Use 15:45 or earlier.")
            if self.exit_at >= MOO_CUTOFF:
                raise ConfigError(
                    f"EXIT_AT={self.exit_at:%H:%M} is at or past Alpaca's MOO cutoff "
                    f"({MOO_CUTOFF:%H:%M} ET). The order would be rejected. Use 09:25 or earlier.")
        if self.stage_at >= self.entry_at:
            raise ConfigError("STAGE_AT must be before ENTRY_AT")
        return self

    def redacted(self) -> dict:
        def fp(s: str) -> str:
            return f"{s[:4]}...{s[-4:]} ({len(s)} chars)" if len(s) > 10 else ("set" if s else "")
        return {
            "mode": self.mode,
            "endpoint": "none (simulated broker)" if self.dry_run else self.base_url,
            "symbols": ",".join(self.symbols),
            "sizing": f"{self.sizing_mode} {self.sizing_value:g}",
            "order_style": self.order_style,
            "api_key": fp(self.api_key),
            "telegram": bool(self.telegram_token and self.telegram_chat),
            "api_clients": len(self.api_keys),
            "schedule_et": (f"stage {self.stage_at:%H:%M} / entry {self.entry_at:%H:%M} "
                            f"(fb {self.entry_fallback_at:%H:%M}) / exit {self.exit_at:%H:%M} "
                            f"(fb {self.exit_fallback_at:%H:%M})"),
        }


def load(env_file: str | Path | None = None) -> Config:
    """Read `.env` (if present) and build a validated Config."""
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file or ROOT / ".env")
    except ImportError:
        pass                       # python-dotenv is optional; real env vars still work
    return Config().validate()
