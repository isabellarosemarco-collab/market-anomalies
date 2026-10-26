"""
Outbound alerts to Telegram and/or Discord.

Design rule: a notifier failure must never break the trade. Every send is wrapped, failures are
recorded and logged, and `send()` returns a bool rather than raising — losing an alert is an
annoyance, while an exception thrown between order submission and state persistence would leave the
engine not knowing whether it holds a position.

Both channels are optional. With neither configured the notifier becomes a local logger, which is
also what the test suite uses (`CaptureNotifier`).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

log = logging.getLogger("fomc.notifier")

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


@dataclass
class Notifier:
    telegram_token: str = ""
    telegram_chat: str = ""
    discord_webhook: str = ""
    dry_run: bool = False
    timeout: float = 10.0

    def __post_init__(self) -> None:
        self.sent: list[str] = []          # every message, in order — inspected by the tests

    @classmethod
    def from_config(cls, cfg) -> "Notifier":
        return cls(telegram_token=cfg.telegram_token, telegram_chat=cfg.telegram_chat,
                   discord_webhook=cfg.discord_webhook, dry_run=cfg.dry_run)

    @property
    def configured(self) -> bool:
        return bool((self.telegram_token and self.telegram_chat) or self.discord_webhook)

    # ---------------------------------------------------------------------- transport
    def send(self, text: str) -> bool:
        """Deliver to every configured channel. Returns True if at least one accepted it."""
        self.sent.append(text)
        log.info("ALERT: %s", text.replace("\n", " | "))
        if not self.configured:
            return False

        ok = False
        if self.telegram_token and self.telegram_chat:
            ok |= self._post_telegram(text)
        if self.discord_webhook:
            ok |= self._post_discord(text)
        return ok

    def _post_telegram(self, text: str) -> bool:
        try:
            import requests
            r = requests.post(
                TELEGRAM_API.format(token=self.telegram_token),
                json={"chat_id": self.telegram_chat, "text": text,
                      "parse_mode": "Markdown", "disable_web_page_preview": True},
                timeout=self.timeout)
            if r.status_code >= 300:
                log.error("telegram %s: %s", r.status_code, r.text[:200])
                return False
            return True
        except Exception as exc:                       # noqa: BLE001 - alerts must never raise
            log.error("telegram send failed: %s", exc)
            return False

    def _post_discord(self, text: str) -> bool:
        try:
            import requests
            r = requests.post(self.discord_webhook, json={"content": text}, timeout=self.timeout)
            if r.status_code >= 300:
                log.error("discord %s: %s", r.status_code, r.text[:200])
                return False
            return True
        except Exception as exc:                       # noqa: BLE001
            log.error("discord send failed: %s", exc)
            return False

    # ----------------------------------------------------------------- message bodies
    def _head(self, icon: str, title: str, mode: str) -> str:
        tag = {"DRY RUN": "[DRY RUN] ", "PAPER": "[PAPER] "}.get(mode, "")
        return f"{icon} *{tag}{title}*"

    def staged(self, mode: str, decision: date_like, symbols: list[str],
               plans: list, when: datetime) -> bool:
        lines = [self._head("🗓", "FOMC Eve detected", mode),
                 f"Decision: *{decision:%a %d %b %Y}* (statement 14:00 ET)",
                 f"Staging entry for {self._fmt_t(when)} ET — market-on-close."]
        for p in plans:
            lines.append(f"  • {p.symbol}: {p.qty_whole} sh @ ~${p.price:,.2f} "
                         f"= ${p.notional_whole:,.0f}" + (f"  ({p.reason})" if p.reason else ""))
        if not plans:
            lines.append("  • no tradable size — see log")
        return self.send("\n".join(lines))

    def entered(self, mode: str, orders: list, when: datetime) -> bool:
        lines = [self._head("🟢", "Entry submitted", mode)]
        for o in orders:
            px = f"${o.filled_avg_price:,.2f}" if o.filled_avg_price else "pending cross"
            lines.append(f"  • BUY {o.qty:g} {o.symbol} @ {px}  [{o.tif.upper()}]"
                         + ("  ⚠️ market order, not the auction cross" if o.degraded else ""))
        lines.append(f"Holding overnight. Exit staged pre-open tomorrow. ({self._fmt_t(when)} ET)")
        return self.send("\n".join(lines))

    def exited(self, mode: str, closes: list, when: datetime) -> bool:
        lines = [self._head("🔵", "Position closed", mode)]
        tot_pl = tot_cost = 0.0
        for c in closes:
            pl, cost = c["pnl"], c["cost_basis"]
            tot_pl += pl
            tot_cost += cost
            pct = (100 * pl / cost) if cost else 0.0
            lines.append(f"  • SOLD {c['qty']:g} {c['symbol']} @ ${c['exit_price']:,.2f} "
                         f"(in ${c['entry_price']:,.2f})  {pl:+,.2f} ({pct:+.2f}%)"
                         + ("  ⚠️ market order" if c.get("degraded") else ""))
        pct = (100 * tot_pl / tot_cost) if tot_cost else 0.0
        lines.append(f"*Overnight net P&L: {tot_pl:+,.2f} ({pct:+.2f}%)*  ({self._fmt_t(when)} ET)")
        return self.send("\n".join(lines))

    def warn(self, mode: str, title: str, detail: str) -> bool:
        return self.send("\n".join([self._head("⚠️", title, mode), detail]))

    def error(self, mode: str, title: str, detail: str) -> bool:
        return self.send("\n".join([self._head("🚨", f"ACTION REQUIRED — {title}", mode), detail]))

    def startup(self, mode: str, info: dict, upcoming: list) -> bool:
        lines = [self._head("🤖", "FOMC Eve engine online", mode)]
        lines += [f"  {k}: {v}" for k, v in info.items()]
        if upcoming:
            lines.append("Next decisions:")
            for u in upcoming:
                eve = f", eve {u['eve']:%a %d %b}" if u.get("eve") else ""
                lines.append(f"  • {u['decision']:%a %d %b %Y} (in {u['days_away']}d{eve})")
        return self.send("\n".join(lines))

    @staticmethod
    def _fmt_t(dt: datetime) -> str:
        return dt.strftime("%H:%M")


# Narrow alias purely for readability in signatures above.
date_like = object


class CaptureNotifier(Notifier):
    """Notifier that never touches the network — used by test_dry_run.py."""

    def send(self, text: str) -> bool:
        self.sent.append(text)
        return True
