# FOMC Eve Overnight Engine

Buys a broad index ETF at the closing auction on the business day before an FOMC rate decision and
sells it at the next morning's opening auction. Eight trades a year. Nothing else.

It runs as a long-lived daemon that wakes every 30 seconds, decides whether the current Eastern
wall-clock time falls inside one of seven action windows, and acts at most once per window per day.
Defaults are paper trading with a simulated broker and no live orders.

---

## 1. What the research actually found

Read this before deciding how much money to point at it.

FOMC eve, 1996–present, entering at the close and exiting at the open:

| Symbol | Net bp/trade | Trades/yr | Ann. Sharpe | Max DD | Perm. p |
|--------|-------------:|----------:|------------:|-------:|--------:|
| SPY | 18.1 | 8 | 1.01 | −1.3% | 0.04 |
| QQQ | 29.6 | 8 | 1.16 | −2.9% | 0.01 |
| IWM | 18.5 | 8 | 0.76 | −2.6% | 0.11 |

It held up on a 1996–2017 holdout that was never used to design the rule (SPY +10.0bp p=0.05,
QQQ +14.7bp p=0.05, IWM +14.8bp p=0.02), it survives deleting the two best trades, and it is
positive in every five-year block since 2001. That is more than almost any retail strategy can
claim.

**Now the part that matters for sizing.** Eight trades a year at ~20bp is about **1.6% per year on
full capital**, with the money idle 97% of the time. Annualized Sharpe is bounded by
`per-trade Sharpe × √(trades per year)`, so 8 nights a year structurally cannot produce a Sharpe
much above ~1.2 no matter how good each night is. This is a small overlay on capital that is doing
something else, not a standalone return stream. The binding constraint is frequency, not edge.

Two related findings, both negative, both worth knowing:

- The night *after* a decision is worth ~zero. The effect is pre-positioning drift, not statement
  digestion.
- NFP mornings are a **loser** (−2.2 / −10.6 / +1.9 bp). "Any 8:30 macro release" does not work.
  Pooling all catalysts (FOMC + CPI + NFP + OpEx) fails outright: 42 trades/yr, ~6–9bp, p≈0.20.

An unanticipated detail that explains the mechanism: the night before a decision is unusually
**quiet** (|overnight| 32.0bp vs 48.4bp baseline, t = −3.28). Positive drift on below-average
volatility is exactly the shape that survives costs.

---

## 2. Two corrections to the obvious design

Both are broker facts, not preferences, and the engine's timing follows from them.

**Alpaca rejects market-on-close orders submitted at or after 15:50 ET.** An entry at 3:59 PM would
fail every single time. Entry is therefore staged at **15:45**, with a plain market order at 15:57
as a last resort.

**Alpaca rejects market-on-open orders submitted at or after 09:28 ET.** A market order at 9:31 is
not the opening auction print the research priced — it pays the spread in the most volatile minute
of the day. Exit is submitted **pre-open at 09:25**, with a market fallback at 09:31.

Where a fallback fires, the order is tagged `degraded=True` and the alert says
`⚠️ market order, not the auction cross`, because that fill is a different trade from the one that
was measured. Round-trip friction is ~4bp against a ~20bp edge; execution quality is a fifth of the
thesis, not a rounding error.

A consequence: `CLS`/`OPG` orders reject fractional quantities, so auction sizing rounds **down** to
whole shares. Fractional sizing exists only in `ORDER_STYLE=market`.

---

## 3. Install

```bash
cd fomc_engine
python -m venv .venv && source .venv/bin/activate      # or use an existing venv
pip install -r requirements.txt
cp .env.example .env                                    # then edit
```

Verify without touching a broker — this needs no keys and no network:

```bash
pytest test_dry_run.py -v
```

## 4. Configure

Everything comes from `.env`. Create it from this template:

```ini
# ---- credentials (paper keys from alpaca.markets/paper-dashboard) ----
APCA_API_KEY_ID=
APCA_API_SECRET_KEY=

# ---- safety ----
PAPER_TRADING=true          # false = real money, and requires the line below
LIVE_TRADING_CONFIRMED=     # must be exactly I_UNDERSTAND_THE_RISK to trade live
DRY_RUN=true                # true = simulated broker, no API calls at all

# ---- strategy ----
SYMBOLS=QQQ                 # comma-separated; notional is split evenly
SIZING_MODE=fixed_cash      # fixed_cash | pct_equity
SIZING_VALUE=10000          # dollars, or a fraction of equity if pct_equity
MAX_NOTIONAL=100000
ORDER_STYLE=auction         # auction | market

# ---- alerts (optional; both channels fire if both are set) ----
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
DISCORD_WEBHOOK_URL=
```

`DRY_RUN=true` is the default and means no network calls to Alpaca whatsoever — an in-memory broker
fills at stub prices. Turn it off to hit the paper endpoint. Real money needs **two** independent
switches (`PAPER_TRADING=false` *and* `LIVE_TRADING_CONFIRMED=I_UNDERSTAND_THE_RISK`); a single
misconfigured variable cannot arm it, and `config.validate()` refuses to start otherwise.

### Telegram in two minutes

1. Message `@BotFather`, send `/newbot`, copy the token into `TELEGRAM_BOT_TOKEN`.
2. Send your new bot any message.
3. `curl https://api.telegram.org/bot<TOKEN>/getUpdates` → copy `result[0].message.chat.id` into
   `TELEGRAM_CHAT_ID`.
4. `python main.py notify-test`

### Discord

Server Settings → Integrations → Webhooks → New Webhook → Copy URL → `DISCORD_WEBHOOK_URL`.

## 5. Run

```bash
python main.py status                 # config, account, next decisions, open position
python main.py calendar               # every decision date + its eve
python main.py calendar --refresh     # re-scrape federalreserve.gov
python main.py check --date 2026-01-27
python main.py notify-test
python main.py once --phase stage     # force one phase now, ignoring its window
python main.py run                    # the daemon
```

Rehearse before trusting it with anything:

```bash
DRY_RUN=true python main.py once --phase stage      # prints the alert it would have sent
DRY_RUN=true python main.py once --phase entry
```

## 6. What it does, minute by minute

All times Eastern. A *window*, not an instant — a process that restarts at 15:47 still trades, and
one that comes back at 15:52 knows it missed the cutoff and says so.

| Phase | Window | Action |
|---|---|---|
| `stage` | 15:00 → 15:45 | Is tomorrow a decision day? Size the trade, alert, pre-flight checks. No exposure. |
| `entry` | 15:45 → 15:50 | Buy, `TimeInForce.CLS`. Hard broker cutoff at 15:50. |
| `entry_fallback` | 15:57 → 15:59:30 | Only if the MOC never went in. Market buy, flagged degraded. |
| `entry_confirm` | 16:05 → 16:45 | Read the cross, persist the real fill price, alert. |
| `exit` | 09:25 → 09:28 | Sell, `TimeInForce.OPG`. Hard broker cutoff at 09:28. |
| `exit_fallback` | 09:31 → 10:00 | Only if still holding. Market sell, flagged. |
| `exit_confirm` | 09:36 → 11:00 | Read the fill, compute overnight P&L, clear state. |

Each phase records itself in `state.json` under its Eastern date, so a crash-restart loop cannot
double-submit. The open position is persisted too: the morning exit acts on the recorded position
and never re-derives whether yesterday was FOMC eve, so a calendar that changed overnight cannot
orphan real exposure. `state.json` is written to a temp file and `replace()`d, so a kill mid-write
cannot corrupt it.

### Daylight saving

Every comparison runs against `datetime.now(pytz.timezone('America/New_York'))`. The window times
above are Eastern by definition, so ET-local comparison is correct through both transitions with no
offset arithmetic. Never schedule this against UTC or host-local time — a host in UTC would trade an
hour early for half the year.

## 7. The calendar

`fed_calendar.py` scrapes `federalreserve.gov/monetarypolicy/fomccalendars.htm` plus the
`fomchistorical{year}.htm` archive back to 1996, and caches to `fomc_dates.json`. It refreshes
automatically at startup when fewer than four future dates remain, so the cache cannot quietly run
dry mid-year.

Parsing the Fed's own pages is fiddlier than it looks, and four real bugs are pinned by tests:

- `Jan/Feb 31-1` — straddle months are abbreviated, and the meeting belongs to the **second** month.
- `August 22 (notation vote)` — an administrative ballot with no 14:00 statement. Excluded.
- Historical pages have no `fomc-meeting` CSS classes at all; headings read `April/May 30-1 Meeting - 2019`.
- `September 15 Meeting` + `September 16 Meeting` in 2003 is **one** two-day meeting split across
  two headings. Scheduled meetings are never on consecutive days, so adjacent dates collapse to the
  later one.

Current cache: 255 dates, 1996-01-31 → 2027-12-08, exactly 8 per year except 2020's 7 — correct,
because the scheduled 17–18 March 2020 meeting was cancelled and replaced by the unscheduled 15
March emergency cut, which is not a scheduled decision.

"FOMC eve" is the business day immediately preceding the statement, using **Alpaca's trading
calendar** rather than a weekday check — a weekday test misfires on the nine market holidays a year,
and a decision the day after a holiday would otherwise stage entry on a closed session.

## 8. Deploying it

Two things make or break an unattended run: the process must survive reboots, and the host clock
must be right.

**systemd (Linux)** — `/etc/systemd/system/fomc.service`:

```ini
[Unit]
Description=FOMC Eve overnight engine
After=network-online.target

[Service]
Type=simple
User=fomc
WorkingDirectory=/opt/fomc_engine
EnvironmentFile=/opt/fomc_engine/.env
ExecStart=/opt/fomc_engine/.venv/bin/python main.py run
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now fomc && journalctl -u fomc -f
timedatectl set-ntp true          # non-negotiable: a 3-minute clock skew misses the MOC cutoff
```

**launchd (macOS)** — same idea; put a `KeepAlive` plist in `~/Library/LaunchAgents/`. Note that a
sleeping laptop does not trade. This wants a machine that stays awake.

**Docker:**

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV TZ=America/New_York
CMD ["python", "main.py", "run"]
```

```bash
docker run -d --restart=unless-stopped --env-file .env \
  -v fomc-state:/app --name fomc fomc-engine
```

Mount a volume for `state.json` and `fomc_dates.json`. Without it, a container restart between the
close and the next open loses the record of the open position — the engine would still be holding
but would no longer know to sell it.

The daemon idles for weeks between trades. That is the design. `python main.py status` confirms it
is alive; the startup alert fires on every restart and lists the next four decision dates.

## 9. Failure handling

Failed orders retry up to `MAX_RETRIES` (3) with linear backoff, then **give up and alert**. Giving
up is deliberate: retrying an order indefinitely across the 15:50 cutoff either gets rejected or,
worse, lands as a different kind of fill than intended. A missed trade is a non-event; an
unrecognized fill is a real problem.

Escalated by webhook with `🚨 ACTION REQUIRED`:

- entry produced no orders (all legs failed or were untradable)
- entry fill unconfirmed at 16:05
- exit submitted no orders, or the position is still open at 09:36
- a stale open position found at startup (recorded yesterday, never closed)
- any phase raising an unexpected exception

A notifier failure never breaks a trade — every send is wrapped and returns a bool. Losing an alert
is an annoyance; an exception thrown between order submission and state persistence would leave the
engine unsure whether it holds a position.

## 10. Before you go live

1. Run `pytest test_dry_run.py` — 30+ tests, no network, no keys.
2. Run a full FOMC cycle on `DRY_RUN=true`. Watch the alerts arrive.
3. Run **at least two real meetings** on paper (`DRY_RUN=false`, `PAPER_TRADING=true`) and compare
   your fills against the official closing and opening prints. If your fills are not at the cross,
   stop — the edge is in the auction, and a market-order version of this strategy is materially
   worse than the table in §1.
4. Only then consider live money, and size it against 1.6%/yr on full capital.

`SYMBOLS=QQQ` is the default because QQQ has the largest measured effect (29.6bp). It is also the
most volatile of the three, and a single 2.9% drawdown night is well within its history.

## 11. If you plan to sell signals

Charging subscribers for trade recommendations is investment-adviser activity in most US
jurisdictions, regardless of whether you execute for them or send a Telegram message at 3 PM.
Registration thresholds, disclosure requirements, and performance-advertising rules all attach. Talk
to a securities lawyer before taking money — this is a real constraint on the business model, not a
formality, and the fix is cheap early and expensive late.

The same applies, more mildly, to publishing the §1 table: performance advertising rules govern how
backtested results may be presented, including required disclosure that they are hypothetical.

## 12. Files

```
config.py          .env loading, validation, the two-switch live-money guard
fed_calendar.py    Fed scraper, caching, eve calculation
alpaca_trader.py   orders, auction constraints, sizing; AlpacaTrader + DryRunTrader
notifier.py        Telegram/Discord; CaptureNotifier for tests
scheduler.py       the 7-phase state machine, State persistence, retries
main.py            CLI
test_dry_run.py    simulated clock, full pipeline, both cutoffs, retry exhaustion
```

Not financial advice. Past performance does not predict future returns, and an eight-trade-a-year
strategy takes decades to distinguish from luck at conventional confidence levels — even this one,
which passed an honest holdout.
