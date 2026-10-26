# Market Anomalies

*I tested dozens of "the market always does X on day Y" ideas against up to 28 years of data. Almost all of them failed. One held up.*

By [Bella Marco](https://bellamarco.co). Sister project: [Resonance Lab](https://github.com/isabellarosemarco-collab/resonance-lab), which tests Tesla's 3-6-9 and numerology the same way.

## What held up: the night before a Fed decision

Buy the index at the close on the evening before a scheduled Federal Reserve (FOMC) decision, then sell at the next morning's open. Prices tend to drift up overnight while everyone waits for the announcement.

| | trades | per trade, after costs | **out-of-sample** | win rate |
|---|---|---|---|---|
| QQQ | 220 | +0.19% | **+0.15%** | 68% |
| IWM | 210 | +0.16% | **+0.15%** | 59% |
| SPY | 245 | +0.12% | **+0.10%** | 60% |

*1996–2026. The rule was designed on 2018+ data, so 1996–2017 is the honest out-of-sample test.*

It's real, and it's small. It only fires about 8 nights a year, so it adds roughly 1.5% a year on money that sits idle the rest of the time. The night after the decision is worth about zero, which tells you this is anticipation, not reaction. The full write-up, including the tests it failed, is in [`fed_drift_suite/README.md`](fed_drift_suite/README.md).

## What didn't

- **Trend-following, volatility targeting, the Halloween effect, buyback windows, risk parity:** none of them beat simply holding SPY (about 11% a year). See [`edge_hunt_results.csv`](edge_hunt_results.csv).
- **Calendar effects** (turn of the month, after options expiry, weekends, pre-holiday, Fed decision day): turn of the month came closest, but none cleared a luck test strict enough for testing many ideas at once. See [`anomaly_battery_results.csv`](anomaly_battery_results.csv).
- **Every big macro release pooled together** (Fed, CPI, jobs report, options expiry): it failed. Jobs-report mornings actually lose money.
- Also tested: overnight drift, options-expiry pinning, index rebalances, the volatility risk premium, closed-end fund discounts, crypto funding carry, and trading on SEC 8-K filings faster than the market. Each has its own report in this folder (`*_report.txt`).

## How the testing works

Every idea was written down before it was run, and every result is reported, pass or fail. Costs are included. Each signal has to beat random trades on the same days of the year, survive a stricter significance bar because so many ideas were tested, and keep working in years it was never designed on. Most of the work here is about not fooling yourself.

## Layout

| Path | What it is |
|---|---|
| `structural_anomalies_tester.py` | the calendar-effect battery |
| `edge_hunt.py` | trend, volatility and seasonal strategies vs. buy-and-hold |
| `fed_drift_suite/` | the one that held up, built into a paper-trading bot (Alpaca) |
| `walk_forward*.py`, `backtester*.py` | the testing harness |
| `event_alpha/`, `special_sits/`, `vrp_alpha/`, `gamma_pin/`, `index_rebal/`, `crypto_carry/` | the other experiments |

*Not investment advice. This is research, and most of it lost.*
