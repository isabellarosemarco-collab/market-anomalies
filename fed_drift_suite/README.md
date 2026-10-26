# fed_drift_suite

Two products, one set of numbers.

**Layer 1 — personal trader.** A daemon on your machine. On the ~8 nights a year before a scheduled
FOMC decision it buys an index ETF at the closing auction, sells at the next opening auction, and
texts you at each step. Alpaca paper or live.

**Layer 2 — commercial API.** A FastAPI service you deploy and sell access to. It serves the signal,
the expected drift, and the full 28-year research record to authenticated clients. It never touches a
client's brokerage account and holds no credential that could.

Both read `core/`, so the dollar figure on your phone and the basis-point figure a client is billed
for are the same number computed from the same 675 committed trades. They cannot drift apart.

---

## 1. The claim, stated the way a sceptic would want it

Buy the index at the close on FOMC eve, sell at the next open. Net of 0.5bp/side slippage, IBKR
commissions, SEC Section 31 fees and FINRA TAF:

| | trades | net bp/trade | **holdout bp** | win % | ann. Sharpe | max DD | HAC t | perm p | drop best 2 |
|---|---|---|---|---|---|---|---|---|---|
| **QQQ** | 220 | +19.34 | **+14.67** | 68.2 | 0.78 | −6.4% | 3.80 | 0.0042 | +17.44 |
| IWM | 210 | +16.03 | **+14.81** | 59.1 | 0.69 | −3.4% | 3.48 | 0.0112 | +13.24 |
| SPY | 245 | +12.29 | **+10.02** | 60.4 | 0.57 | −4.4% | 3.01 | 0.0128 | +9.56 |

*1996-01-31 → 2026-09-16. Regenerate with `python -m core.analytics`.*

**Read the holdout column, not the net column.** The rule was specified using 2018+ data, so only
1996–2017 is genuinely out-of-sample. Size against +14.7bp, not +19.3bp.

**Why the recommendation is QQQ, with a caveat printed in the response.** QQQ and IWM are 0.14bp
apart on holdout evidence. That is noise, not a ranking, so `recommended_symbol()` declares a tie,
breaks it on the full-sample permutation p-value, and says in plain text that either is defensible.

**Four things that constrain this, all published by the API:**

- **8 trades a year caps it.** Annualized Sharpe = per-trade Sharpe × √(trades per year). At 8
  trades, ~1.2 is the ceiling no matter how good each night is. Anyone quoting a 3 Sharpe here is
  quoting something else.
- **~1.5%/yr on reserved capital**, with the money idle ~97% of the time. This is an overlay on
  capital doing something else, not a return stream.
- **~4bp round-trip friction against a ~20bp gross edge.** Execution is a fifth of the thesis. A
  market-order implementation is materially worse than the table above.
- **~6 years of live trading** to distinguish this from luck at 95% confidence — `(1.96/Sharpe)²`.

**And the tests it failed**, shipped in the same API response as the wins, because otherwise a reader
cannot judge how much searching produced the winner:

- Pooling all macro catalysts (FOMC + CPI + NFP + OpEx) **fails**: 42 trades/yr, +5.5 to +8.6bp,
  permutation p 0.18–0.22. That was the pre-registered primary test.
- NFP mornings **lose** (−10.6bp QQQ). The effect is not "any 8:30am macro release".
- The night *after* a decision is worth ~zero, so this is pre-positioning drift, not post-statement
  digestion.
- An intraday volume-washout filter was **rejected**: QQQ negative at every setting while SPY and IWM
  were positive, and the sensitivity surface was a spike rather than a plateau. Both are
  fitted-artifact signatures.

---

## 2. Why not "3:59 PM MOC / 9:31 AM MOO"

That was the original design and it is not implementable. Two hard broker facts:

- Alpaca **rejects** market-on-close (`TimeInForce.CLS`) submitted at or after **15:50 ET**.
- Alpaca **rejects** market-on-open (`TimeInForce.OPG`) submitted at or after **09:28 ET**.

So entry is staged at **15:45** and exit at **09:25**, which still fills at the 16:00 and 09:30
crosses — the prices the research actually priced. A market order at 15:59 is a *different trade*: it
pays the spread and, at the open, lands in the most volatile minute of the session.

The fallback path exists anyway (15:57 and 09:31) because an unmanaged position is worse than a bad
fill. When it fires, every alert and log line carries `degraded=True` and the words "market order,
not the auction cross". `Config.validate()` refuses to start if you set `ENTRY_AT` past 15:50 in
auction mode, so this is caught at boot rather than at 15:59 on the one day a quarter that matters.

`test_literal_spec_time_1559_degrades_to_a_market_order` pins this behaviour.

---

## 3. Layout

```
fed_drift_suite/
├── core/                      shared math — both layers import this and nothing else
│   ├── calendar.py            FOMC dates, 1996→present, self-scraping with a JSON cache
│   ├── analytics.py           every published statistic, computed from the CSVs at load
│   ├── models.py              pydantic schemas = the API's wire format and its documentation
│   ├── build_dataset.py       rebuilds the trade file from yfinance (run only to refresh)
│   └── data/                  fomc_dates.json · fomc_eve_returns.csv (675) · all_nights.csv (22,018)
├── personal/
│   ├── trader.py              Alpaca client: auction orders, sizing, dry-run broker
│   ├── notifier.py            Telegram / Discord alerts; a send failure never breaks a trade
│   └── runner.py              the daemon + CLI (status, check, calendar, once, run)
├── commercial/
│   ├── api.py                 FastAPI app: X-API-KEY auth, rate limit, worker lifecycle
│   ├── endpoints.py           the six routes
│   ├── webhook_worker.py      HMAC-signed delivery + the 15:00 ET broadcast
│   └── market_days.py         NYSE trading days (rules-based, no data dependency)
├── config.py                  one Config for both layers, from .env
├── test_suite.py              33 tests, both layers
├── Dockerfile                 layer 2 only, non-root, no secrets in the image
└── requirements.txt
```

**Nothing in `analytics.py` is a literal.** If a client disputes the Sharpe, the answer is "here are
the 245 trades, here is the function, run it" — the only kind of performance claim worth making.
`/v1/research/trades` exists for exactly that, and `/v1/research/metrics` will 503 rather than serve a
number it cannot source.

---

## 4. Install and verify

```bash
cd fed_drift_suite
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest test_suite.py -q          # 33 passed
```

The suite is the acceptance test for both layers:

- **Layer 1** mocks the clock onto FOMC eve 2026-01-27 and walks the daemon through stage → entry →
  confirm → next-morning exit, asserting that mock Alpaca orders are submitted (20 QQQ, `cls`, not
  degraded) and mock Telegram alerts fire, ending with an exact `+140.00 (+1.40%)`.
- **Layer 2** drives every endpoint through FastAPI's `TestClient` and — reading the app's own OpenAPI
  schema rather than a hand-written list — asserts each `/v1` path returns **401** with no key and
  **403** with a wrong one. A future endpoint cannot quietly skip auth.

Sanity checks you can run now, no credentials needed:

```bash
python demo.py                             # time-travel to the next FOMC eve and play the whole night
python demo.py --move -0.9                 # ...as a losing night
python demo.py --late                      # ...having missed the 15:50 MOC cutoff
python -m core.analytics                   # the research table above
python personal/runner.py status           # mode, account, next eve, research, open position
python personal/runner.py calendar         # cached FOMC dates and coverage
```

`demo.py` runs the real `Engine` on the real phase schedule with the real alert bodies — only the
clock and the broker are substituted, so what it prints is what the live daemon does.

---

## 5. Run Layer 1 (your trades)

1. Paper keys from the [Alpaca dashboard](https://app.alpaca.markets/paper/dashboard/overview) →
   *API Keys*.
2. Telegram bot: message **@BotFather** → `/newbot` → copy the token. Message your new bot once, then
   read your chat id from `https://api.telegram.org/bot<TOKEN>/getUpdates`.
3. `cp .env.example .env` and fill in:

```ini
APCA_API_KEY_ID=PK...
APCA_API_SECRET_KEY=...
DRY_RUN=false
PAPER_TRADING=true
SYMBOLS=QQQ
SIZING_VALUE=10000
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
```

4. Check the wiring, then run:

```bash
python personal/runner.py status           # config, broker account, next eve, research
python personal/runner.py notify-test      # one message, to prove Telegram is wired
python personal/runner.py check --date 2026-10-27   # is that date an FOMC eve? (it is)
python personal/runner.py once --phase stage        # force a single phase now, no loop
python personal/runner.py run              # the daemon
```

What arrives on your phone on an FOMC eve:

```
🗓 [PAPER] FOMC Eve detected
Decision: Wed 28 Oct 2026 (statement 14:00 ET)
Staging entry for the closing auction; exit at tomorrow's open.
  • QQQ: 20 sh @ ~$500.00 = $10,000
Expected: +14.67 on out-of-sample history (68% of past trades positive).
One night's variance is many times this.

🟢 [PAPER] Entry submitted
  • BUY 20 QQQ @ pending cross  [CLS]
Holding overnight. Exit staged pre-open tomorrow. (15:45 ET)

🔵 [PAPER] Position closed
  • SOLD 20 QQQ @ $507.00 (in $500.00)  +140.00 (+1.40%)
Overnight net P&L: +140.00 (+1.40%)  (09:36 ET)
```

The staged alert quotes the **holdout** expectancy, not the full-sample one. Quoting the flattering
number on your own phone is how you end up believing it.

### Keeping it alive

`run` is a foreground loop; a laptop that sleeps through 15:45 places no trade. Use `launchd` (macOS)
or `systemd` (Linux):

```ini
# /etc/systemd/system/fed-drift.service
[Unit]
Description=FOMC eve overnight trader
After=network-online.target

[Service]
WorkingDirectory=/opt/fed_drift_suite
ExecStart=/opt/fed_drift_suite/.venv/bin/python personal/runner.py run
Restart=always
RestartSec=30
User=trader

[Install]
WantedBy=multi-user.target
```

Restart-safety is designed in, not hoped for. Phases are **windows, not instants** (`entry` is
15:45–15:50, not "at 15:45"), and `state.json` is written atomically with a per-day per-phase marker.
A process that dies at 15:46 and comes back at 15:48 still enters; one that comes back at 16:10 does
not enter twice. `test_state_survives_a_restart_between_entry_and_exit` covers it.

### Before you switch to live money

- [ ] One full paper cycle observed end to end, with the Telegram messages received.
- [ ] `python personal/runner.py check` clean.
- [ ] Sizing you can watch lose 2% overnight without flinching. The worst modelled single trade is
      around −190bp and the strategy holds through a scheduled macro event.
- [ ] Both switches set deliberately: `PAPER_TRADING=false` **and**
      `LIVE_TRADING_CONFIRMED=I_UNDERSTAND_THE_RISK`.
- [ ] You have read §7.

---

## 6. Run Layer 2 (the product)

```bash
python -c "import secrets; print('fds_live_' + secrets.token_urlsafe(32))"   # per client
export API_KEYS="fds_live_xxx:AcmeCapital,fds_live_yyy:FamilyOffice"
uvicorn commercial.api:app --port 8000
```

```bash
docker build -t fed-drift-api .
docker run -p 8000:8000 \
  -e API_KEYS="fds_live_xxx:AcmeCapital" \
  -v fed-drift-data:/data \
  fed-drift-api
```

Runs fine on a $5/month instance (Render, Railway, Fly, DigitalOcean). Set `API_KEYS` in the
platform's secret manager, not in a file. **Mount the volume** — `/data/subscriptions.json` holds
webhook signing secrets, and a redeploy without it silently loses every client's registration.

Keep it to **one worker**: the 15:00 ET broadcast runs in-process, so N workers would send N copies of
the same signal. Scale out only after moving the watcher to its own process.

### Endpoints

| | |
|---|---|
| `GET /health` | unauthenticated, for the load balancer |
| `GET /v1/status` | is today an active FOMC window? |
| `GET /v1/signal/next` | next meeting, recommended asset, expected drift in bp, historical max drawdown |
| `GET /v1/research/metrics` | full/holdout/design statistics per symbol, plus negative results and structural limits |
| `GET /v1/research/trades` | all 675 trades, JSON or CSV, filterable — so every number above is checkable |
| `POST /v1/webhooks/subscribe` | register an HTTPS endpoint for the 15:00 ET push |
| `GET /v1/webhooks/subscriptions` · `DELETE /v1/webhooks/{id}` · `POST /v1/webhooks/{id}/test` | manage and test delivery |

`/docs` is live, interactive OpenAPI — in practice it is the client documentation, so the field
descriptions in `core/models.py` are written for a client's diligence team rather than for us.

```bash
curl -H "X-API-KEY: fds_live_xxx" localhost:8000/v1/signal/next
curl -H "X-API-KEY: fds_live_xxx" localhost:8000/v1/research/metrics
curl -H "X-API-KEY: fds_live_xxx" "localhost:8000/v1/research/trades?format=csv" -o trades.csv
```

### Webhooks

```bash
curl -X POST localhost:8000/v1/webhooks/subscribe \
  -H "X-API-KEY: fds_live_xxx" -H "Content-Type: application/json" \
  -d '{"url":"https://client.example.com/hooks/fomc","symbols":["QQQ"],
       "secret":"<32+ random chars>","label":"Acme Capital"}'
```

At 15:00 ET on each FOMC eve every active subscription receives a POST with the signal. Deliveries
carry:

| header | meaning |
|---|---|
| `X-Signature` | `sha256=<hmac>` over `"{X-Timestamp}." + body`, keyed with your secret |
| `X-Timestamp` | signed, so it cannot be altered independently — reject anything older than a few minutes |
| `X-Delivery-Id` | stable per (subscription, event, day), so a retry after your timeout is identifiable as the same signal, not a second trade |

Verify before acting. An unauthenticated POST claiming to be a trading signal is worth nothing:

```python
import hashlib, hmac
def verify(body: bytes, secret: str, timestamp: str, signature: str) -> bool:
    mac = hmac.new(secret.encode(), f"{timestamp}.".encode() + body, hashlib.sha256)
    return hmac.compare_digest(f"sha256={mac.hexdigest()}", signature)   # never ==
```

Delivery is at-least-once with 2s/4s/8s backoff. A 4xx (other than 408/429) stops immediately — that
is a configuration problem, not a transient one. After 10 consecutive failures the subscription
deactivates, and `GET /v1/webhooks/subscriptions` shows `last_delivery_status` and
`consecutive_failures` so a client can see it themselves.

### Security posture

- **`X-API-KEY` compared with `secrets.compare_digest`**, against every configured key without an
  early break. `==` on a secret returns on the first differing byte, and that timing is measurable
  across a network.
- **Keys come from the environment**, never from a file in the image or the repo. Plaintext secrets
  sitting where a broader audience than the owner can read them is the failure mode to design
  against; env injection from the platform's secret store keeps the value out of image layers, out of
  git and out of `docker history`. Logs record a four-character fingerprint and the client label only.
- **Fails closed.** No `API_KEYS` configured ⇒ every `/v1` route returns **503**, never 200. An API
  that falls open when its configuration goes missing is worse than one that breaks, because nothing
  visibly breaks.
- **Auth is a router-level dependency**, so a newly added endpoint is authenticated by default. The
  opposite arrangement makes one forgotten decorator a public data leak.
- **Webhook secrets** must be retained in their original form to compute an HMAC, so they cannot be
  one-way hashed. Mitigations: the store is written mode **0600**, the secret is never returned by any
  endpoint, never logged, and never echoed in a delivery body. For a serious deployment, put
  `SUBS_PATH` on an encrypted volume or swap `SubscriptionStore` for a secrets-manager implementation
  — the class is deliberately small and single-purpose so it can be replaced.
- **No client credential ever enters this service.** It serves data; clients trade their own accounts.
  That is the property that makes it sellable to a fund at all, and it should stay true.
- Tracebacks are never returned. Errors log with a reference id; the client gets the id.
- The container runs as a non-root user and contains no secrets.

---

## 7. Limits, failure modes, and the legal bit

**What breaks, and what happens.** Broker errors retry with backoff and then alert
(`MAX_RETRIES`/`RETRY_BACKOFF_S`); a notifier failure is logged and swallowed rather than allowed to
raise between order submission and state persistence; a still-open position after the bell is
flattened at market by `exit_fallback` with the degradation flagged; a phase that raises is caught,
marked done, and reported, so one bad handler cannot wedge the loop. The FOMC calendar re-scrapes
itself when fewer than four future dates remain, and `market_days.py` computes NYSE holidays from
rules — ad-hoc closures (state funerals, Sandy) are not modelled, because they are unpredictable by
nature.

**Known limitations.** Dividends are attributed to the correct bucket by using raw prices and adding
the dividend explicitly (`auto_adjust=True` silently credits the whole dividend to the overnight
return — worth 1.3–1.8%/yr of spurious SPY drift, which is larger than the effect being measured).
The decomposition identity `(1+r_night)(1+r_day) == 1+r_total` is asserted at load. Capacity is large
but the auction print is a single price: a big enough order becomes a participant in the cross rather
than a taker of it. Rate limiting is in-process and resets on redeploy.

**Legal.** Backtested results are hypothetical, do not represent trading in any actual account, and
carry the benefit of hindsight. Nothing here is investment advice.

Selling *data* — signals, statistics, research — is generally fine. The moment you give personalised
recommendations for compensation, or manage anyone's money, US investment-adviser registration and
state rules are in play. If a client asks "how much should *I* allocate?", that is the line. Talk to a
securities lawyer before you answer it, and keep Layer 2 doing what it does now: serving numbers,
never touching an account.
