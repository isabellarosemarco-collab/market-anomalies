"""
Configuration for the FOMC Eve engine, loaded from the environment / .env.

SAFETY DEFAULTS. Every switch here defaults to the harmless setting: paper trading on, dry run on,
notifications off. Going live requires setting two independent variables, not one, because a single
typo'd boolean should not be able to put real money at risk. See `Config.validate()`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"
MARKET_TZ = "America/New_York"

# Broker-imposed auction cutoffs. These are not style choices — Alpaca REJECTS the order outside
# these windows, so the schedule below is built backwards from them.
MOC_CUTOFF = time(15, 50)      # market-on-close (TimeInForce.CLS) must be submitted before this
MOO_CUTOFF = time(9, 28)       # market-on-open  (TimeInForce.OPG) must be submitted before this


def _flag(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "y", "on")


def _num(name: str, default: float) -> float:
    v = os.getenv(name)
    try:
        return float(v) if v not in (None, "") else default
    except ValueError:
        return default


def _hhmm(name: str, default: str) -> time:
    raw = (os.getenv(name) or default).strip()
    h, m = raw.split(":")
    return time(int(h), int(m))


class ConfigError(RuntimeError):
    pass


@dataclass
class Config:
    # --- broker
    api_key: str = field(default_factory=lambda: os.getenv("APCA_API_KEY_ID", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("APCA_API_SECRET_KEY", ""))
    paper: bool = field(default_factory=lambda: _flag("PAPER_TRADING", True))
    live_confirmed: str = field(default_factory=lambda: os.getenv("LIVE_TRADING_CONFIRMED", ""))

    # --- strategy
    symbols: list[str] = field(default_factory=lambda: [
        s.strip().upper() for s in (os.getenv("SYMBOLS") or "QQQ").split(",") if s.strip()])
    sizing_mode: str = field(default_factory=lambda: (os.getenv("SIZING_MODE") or "fixed_cash"))
    sizing_value: float = field(default_factory=lambda: _num("SIZING_VALUE", 10_000.0))
    max_notional: float = field(default_factory=lambda: _num("MAX_NOTIONAL", 100_000.0))

    # 'auction' reproduces the backtest: MOC in, MOO out, both at the official cross with no spread
    # crossed. 'market' is the fallback path and DOES pay the spread — the backtested edge was
    # measured at auction prices, so 'market' is a degraded mode, not an equivalent one.
    order_style: str = field(default_factory=lambda: (os.getenv("ORDER_STYLE") or "auction"))

    # --- schedule (Eastern). Windows, not instants, so a restart inside a window still acts.
    stage_at: time = field(default_factory=lambda: _hhmm("STAGE_ALERT_AT", "15:00"))
    entry_at: time = field(default_factory=lambda: _hhmm("ENTRY_AT", "15:45"))
    entry_fallback_at: time = field(default_factory=lambda: _hhmm("ENTRY_FALLBACK_AT", "15:57"))
    entry_confirm_at: time = field(default_factory=lambda: _hhmm("ENTRY_CONFIRM_AT", "16:05"))
    exit_at: time = field(default_factory=lambda: _hhmm("EXIT_AT", "09:25"))
    exit_fallback_at: time = field(default_factory=lambda: _hhmm("EXIT_FALLBACK_AT", "09:31"))
    exit_confirm_at: time = field(default_factory=lambda: _hhmm("EXIT_CONFIRM_AT", "09:36"))

    # --- notifications
    telegram_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))
    discord_webhook: str = field(default_factory=lambda: os.getenv("DISCORD_WEBHOOK_URL", ""))

    # --- operations
    dry_run: bool = field(default_factory=lambda: _flag("DRY_RUN", True))
    max_retries: int = field(default_factory=lambda: int(_num("MAX_RETRIES", 3)))
    retry_backoff_s: float = field(default_factory=lambda: _num("RETRY_BACKOFF_S", 5.0))
    poll_seconds: float = field(default_factory=lambda: _num("POLL_SECONDS", 30.0))
    state_path: Path = field(default_factory=lambda: ROOT / (os.getenv("STATE_FILE") or "state.json"))
    cache_path: Path = field(default_factory=lambda: ROOT / "fomc_dates.json")
    log_path: Path = field(default_factory=lambda: ROOT / "engine.log")

    @property
    def base_url(self) -> str:
        return PAPER_URL if self.paper else LIVE_URL

    @property
    def is_live_money(self) -> bool:
        return not self.paper and not self.dry_run

    def validate(self) -> None:
        """Fail loudly at startup rather than at 15:45 on the one day of the quarter that matters."""
        if self.sizing_mode not in ("fixed_cash", "pct_equity"):
            raise ConfigError(f"SIZING_MODE must be fixed_cash or pct_equity, got {self.sizing_mode!r}")
        if self.order_style not in ("auction", "market"):
            raise ConfigError(f"ORDER_STYLE must be auction or market, got {self.order_style!r}")
        if self.sizing_value <= 0:
            raise ConfigError("SIZING_VALUE must be positive")
        if self.sizing_mode == "pct_equity" and self.sizing_value > 1.0:
            raise ConfigError("SIZING_VALUE must be a fraction (0-1) when SIZING_MODE=pct_equity")
        if not self.symbols:
            raise ConfigError("SYMBOLS is empty")
        if not self.dry_run and not (self.api_key and self.api_secret):
            raise ConfigError("APCA_API_KEY_ID / APCA_API_SECRET_KEY required unless DRY_RUN=true")

        # The two-key guard. PAPER_TRADING=false alone is not enough to trade real money.
        if self.is_live_money and self.live_confirmed != "I_UNDERSTAND_THE_RISK":
            raise ConfigError(
                "Refusing to trade live money. PAPER_TRADING=false and DRY_RUN=false are set, but "
                "LIVE_TRADING_CONFIRMED is not 'I_UNDERSTAND_THE_RISK'. Set it deliberately, or "
                "leave PAPER_TRADING=true.")

        # The entry schedule must respect the broker's auction cutoffs or orders will be rejected.
        if self.order_style == "auction":
            if self.entry_at >= MOC_CUTOFF:
                raise ConfigError(
                    f"ENTRY_AT {self.entry_at} is at/after Alpaca's {MOC_CUTOFF} MOC cutoff; "
                    "an MOC order submitted then is rejected.")
            if self.exit_at >= MOO_CUTOFF:
                raise ConfigError(
                    f"EXIT_AT {self.exit_at} is at/after Alpaca's {MOO_CUTOFF} MOO cutoff; "
                    "an MOO order submitted then is rejected.")

    def redacted(self) -> dict:
        mask = lambda s: f"{s[:4]}...{s[-2:]}" if len(s) > 8 else ("set" if s else "")
        return {
            "mode": "DRY RUN" if self.dry_run else ("PAPER" if self.paper else "*** LIVE MONEY ***"),
            "endpoint": self.base_url, "symbols": ",".join(self.symbols),
            "sizing": f"{self.sizing_mode}={self.sizing_value:g}",
            "order_style": self.order_style,
            "api_key": mask(self.api_key),
            "telegram": bool(self.telegram_token and self.telegram_chat),
            "discord": bool(self.discord_webhook),
            "schedule_et": (f"stage {self.stage_at:%H:%M} / entry {self.entry_at:%H:%M} "
                            f"(fb {self.entry_fallback_at:%H:%M}) / exit {self.exit_at:%H:%M} "
                            f"(fb {self.exit_fallback_at:%H:%M})"),
        }


def load() -> Config:
    cfg = Config()
    cfg.validate()
    return cfg
