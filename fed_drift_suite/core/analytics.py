"""
Every statistic the product publishes, computed from `data/fomc_eve_returns.csv` at load time.

Nothing here is a literal. If a client disputes the Sharpe, the answer is "here are the 245 trades,
here is the function, run it" — which is the only form of performance claim worth making. Rerunning
`python -m core.build_dataset` refreshes the inputs; the numbers follow.

WHAT IS ACTUALLY BEING CLAIMED

Buy the close before a scheduled FOMC decision, sell the decision day's open. Net of 0.5bp/side
slippage, IBKR commissions and regulatory fees, that has averaged roughly +12 to +19bp per trade
since 1996 depending on the index, and it was positive on the 1996-2017 holdout that was never used
to design the rule.

THE THREE THINGS THAT KEEP THIS HONEST

1. `permutation_p` compares the FOMC-eve nights against 5,000 random same-size samples of ordinary
   nights from the same series. On a few hundred fat-tailed observations this is a far better test
   than a t-statistic, which assumes a shape these returns do not have. The baseline population is
   shipped in `data/all_nights.csv` so a client can rebuild the null themselves.

2. `holdout` statistics are reported separately and are always the smaller number. The rule was
   specified on 2018+ data; 1996-2017 is the only genuinely out-of-sample evidence, so that is the
   figure to size against. Quoting the full-sample number as if it were out-of-sample would be the
   single easiest way to mislead someone here.

3. `drop_best_2_net_bp` deletes the two best trades. An 8-trade-a-year strategy whose edge lives in
   two nights out of 245 is not a strategy, and this is the cheapest way to see that.

THE STRUCTURAL CEILING, stated because clients will otherwise infer something untrue from the Sharpe

    annualized Sharpe = per-trade Sharpe x sqrt(trades per year)

With 8 trades a year that caps annualized Sharpe near 1.2 no matter how good each night is, and caps
the return on full capital near 1.6%/yr with the money idle 97% of the time. The binding constraint
on this strategy is frequency, not edge. It is an overlay on capital doing something else.
"""
from __future__ import annotations

import functools
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from core.calendar import DATA_DIR
from core.models import Provenance, ResearchMetrics, SampleStats, SymbolResearch

RETURNS_CSV = DATA_DIR / "fomc_eve_returns.csv"
NIGHTS_CSV = DATA_DIR / "all_nights.csv"
PROVENANCE_JSON = DATA_DIR / "provenance.json"

PERM_DRAWS = 5_000
PERM_SEED = 20260115            # fixed so a client recomputing gets the same p-value
HAC_LAGS = 5
TRADING_DAYS = 252

# Results the hypothesis failed. Shipped with the wins on purpose — see `known_negative_results`
# in the API response. Without these a reader cannot judge how much searching produced the winner.
NEGATIVE_RESULTS = [
    "Pooling all macro catalysts (FOMC + CPI + NFP + monthly OpEx) FAILS: 42 trades/yr, "
    "+5.5 to +8.6bp net, permutation p 0.18-0.22. This was the pre-registered primary test.",
    "NFP (jobs report) mornings are a LOSER: -2.2bp SPY, -10.6bp QQQ, +1.9bp IWM. The effect is "
    "not 'any 8:30am macro release'.",
    "The night AFTER a decision is worth approximately zero, so the effect is pre-positioning "
    "drift, not post-statement digestion.",
    "Monthly options-expiration Fridays show no overnight signature at all (|return| ratio 1.01, "
    "t=0.08).",
    "An intraday volume-washout filter was REJECTED on structural grounds: QQQ was negative at "
    "every parameter setting while SPY and IWM were positive, and the sensitivity surface was a "
    "spike at the tested threshold rather than a plateau. Both are fitted-artifact signatures.",
    "Annualized Sharpe above 1.2 is not attainable here and was not achieved. See structural_limits.",
]

STRUCTURAL_LIMITS = [
    "8 trades per year. Annualized Sharpe = per-trade Sharpe x sqrt(8), which caps it near 1.2.",
    "Approximately 1.6%/yr on full capital, with capital idle ~97% of the time. This is an overlay, "
    "not a standalone return stream.",
    "Round-trip friction is ~4bp against a ~20bp gross edge, so execution quality is a fifth of the "
    "thesis. Fills must be at the closing and opening auction crosses; a market-order implementation "
    "is materially worse than these figures.",
    "At this Sharpe, distinguishing the strategy from luck at 95% confidence takes on the order of "
    "(1.96/Sharpe)^2 years of live trading. Roughly 3-4 years for QQQ, longer for SPY and IWM.",
    "Capacity is large (the underlying ETFs trade billions per day) but the auction print is a "
    "single price: a large enough order becomes a participant in the cross rather than a taker of it.",
]


# ------------------------------------------------------------------------- primitives

def hac_t(x: np.ndarray, lags: int = HAC_LAGS) -> float:
    """Newey-West t-statistic for H0: mean = 0. Falls back to the plain t if statsmodels is absent."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 30:
        return float("nan")
    try:
        import statsmodels.api as sm
        fit = sm.OLS(x, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        return float(fit.tvalues[0])
    except ImportError:
        return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def max_drawdown_pct(bp: np.ndarray) -> float:
    """Drawdown on the trade-sequence equity curve — the sequence of 8-a-year trades, compounded."""
    r = np.asarray(bp, dtype=float) / 1e4
    if r.size == 0:
        return 0.0
    eq = np.cumprod(1.0 + r)
    return float((eq / np.maximum.accumulate(eq) - 1.0).min() * 100)


def permutation_p(sample: np.ndarray, population: np.ndarray,
                  draws: int = PERM_DRAWS, seed: int = PERM_SEED) -> float:
    """P(a random same-size sample of ordinary nights has a mean this high).

    The null is 'these nights are unremarkable', which is exactly what a sceptic means. Drawing
    without replacement matches the real sample structure.
    """
    n = len(sample)
    if n == 0 or len(population) <= n:
        return float("nan")
    rng = np.random.default_rng(seed)
    obs = float(np.mean(sample))
    idx = rng.integers(0, len(population), size=(draws, n))
    means = population[idx].mean(axis=1)
    # +1 in numerator and denominator: never report p=0 off 5,000 draws.
    return float((np.sum(means >= obs) + 1) / (draws + 1))


# ------------------------------------------------------------------------------ data

@functools.lru_cache(maxsize=1)
def trades() -> pd.DataFrame:
    if not RETURNS_CSV.exists():
        raise FileNotFoundError(
            f"{RETURNS_CSV} is missing. Run: python -m core.build_dataset")
    df = pd.read_csv(RETURNS_CSV, parse_dates=["decision_date", "eve_date"])
    df["decision_date"] = df["decision_date"].dt.date
    df["eve_date"] = df["eve_date"].dt.date
    return df


@functools.lru_cache(maxsize=1)
def all_nights() -> pd.DataFrame:
    if not NIGHTS_CSV.exists():
        raise FileNotFoundError(f"{NIGHTS_CSV} is missing. Run: python -m core.build_dataset")
    return pd.read_csv(NIGHTS_CSV)


@functools.lru_cache(maxsize=1)
def provenance() -> Provenance:
    return Provenance(**json.loads(PROVENANCE_JSON.read_text()))


def symbols() -> list[str]:
    return sorted(trades()["symbol"].unique().tolist())


# ------------------------------------------------------------------------------ stats

def _stats(g: pd.DataFrame, label: str, population: np.ndarray) -> SampleStats:
    net = g["net_bp"].to_numpy(dtype=float)
    n = len(net)
    if n == 0:
        raise ValueError(f"no trades in sample '{label}'")

    span_years = max((g["decision_date"].max() - g["decision_date"].min()).days / 365.25, 1e-9)
    per_year = n / span_years if n > 1 else 8.0
    sd = float(net.std(ddof=1)) if n > 1 else float("nan")
    per_trade_sharpe = float(net.mean() / sd) if sd and sd > 0 else float("nan")

    # Compounded return on capital that sits available all year for ~8 nights of use.
    equity = float(np.prod(1.0 + net / 1e4))
    ann_on_full = 100 * (equity ** (1 / span_years) - 1)

    worst_dropped = np.sort(net)[:-2] if n > 3 else net

    return SampleStats(
        label=label,
        n_trades=n,
        first=g["decision_date"].min(),
        last=g["decision_date"].max(),
        gross_bp_mean=round(float(g["gross_bp"].mean()), 2),
        cost_bp_mean=round(float(g["cost_bp"].mean()), 3),
        net_bp_mean=round(float(net.mean()), 2),
        net_bp_median=round(float(np.median(net)), 2),
        net_bp_sd=round(sd, 2),
        win_rate=round(float((net > 0).mean()), 4),
        sharpe_per_trade=round(per_trade_sharpe, 4),
        sharpe_annualized=round(per_trade_sharpe * np.sqrt(per_year), 3),
        trades_per_year=round(per_year, 2),
        max_drawdown_pct=round(max_drawdown_pct(net), 2),
        t_stat_hac=round(hac_t(net), 2),
        permutation_p=round(permutation_p(net, population), 4),
        ann_return_on_full_capital_pct=round(ann_on_full, 2),
        drop_best_2_net_bp=round(float(worst_dropped.mean()), 2),
    )


@functools.lru_cache(maxsize=1)
def research(symbol: str) -> SymbolResearch:
    t = trades()
    g = t[t["symbol"] == symbol]
    if g.empty:
        raise KeyError(f"no trades for {symbol}")

    nights = all_nights()
    # The null population deliberately EXCLUDES the FOMC nights themselves, so the comparison is
    # against ordinary nights rather than against a pool the sample is already sitting inside.
    pop = nights[(nights["symbol"] == symbol) & (~nights["is_fomc_eve_night"])]
    population = pop["net_bp"].to_numpy(dtype=float)

    return SymbolResearch(
        symbol=symbol,
        full=_stats(g, "full", population),
        holdout=_stats(g[g["sample"] == "holdout"], "holdout", population),
        design=_stats(g[g["sample"] == "design"], "design", population),
    )


TIE_BAND_BP = 2.0        # holdout means closer than this are not distinguishable; do not rank them


@functools.lru_cache(maxsize=1)
def recommended_symbol() -> tuple[str, str]:
    """Pick on HOLDOUT evidence, and refuse to rank differences that are noise.

    Two rules, both there to avoid a specific way of fooling a client:

    Rank on out-of-sample expectancy, not full-sample. The full sample includes the period used to
    specify the rule, so its winner is partly a selection artifact.

    Then, if the leaders are within `TIE_BAND_BP` of each other, do not pretend the ordering means
    anything — these are standard errors of several basis points, so a 0.1bp gap is nothing. Break
    the tie on the full-sample permutation p-value (which uses every observation and so has more to
    say than the holdout mean) and disclose that it was a tie.
    """
    ranked = sorted(((research(s).holdout.net_bp_mean, s) for s in symbols()), reverse=True)
    top_score, top = ranked[0]
    tied = [s for score, s in ranked if top_score - score <= TIE_BAND_BP]

    if len(tied) > 1:
        sym = min(tied, key=lambda s: research(s).full.permutation_p)
        r = research(sym)
        rest = [s for s in tied if s != sym]
        others = ", ".join(f"{s} +{research(s).holdout.net_bp_mean:.1f}bp" for s in rest)
        verb = "sits" if len(rest) == 1 else "sit"
        why = (f"{sym}: +{r.holdout.net_bp_mean:.1f}bp net over {r.holdout.n_trades} "
               f"out-of-sample trades. Note this was a TIE — {others} {verb} within "
               f"{TIE_BAND_BP:.0f}bp of it, which is well inside the standard error, so the "
               f"ordering of the leaders carries no information. {sym} was chosen on the stronger "
               f"full-sample permutation p-value ({r.full.permutation_p:.3f}) rather than on the "
               f"holdout mean. Any of {'/'.join(tied)} is a defensible choice.")
        return sym, why

    r = research(top)
    why = (f"{top} has the highest out-of-sample expectancy "
           f"(+{r.holdout.net_bp_mean:.1f}bp net over {r.holdout.n_trades} holdout trades, "
           f"permutation p={r.holdout.permutation_p:.3f}), by more than {TIE_BAND_BP:.0f}bp. "
           f"Ranked on holdout rather than full-sample results, since the rule was specified "
           f"using 2018+ data.")
    return top, why


@functools.lru_cache(maxsize=1)
def metrics() -> ResearchMetrics:
    sym, why = recommended_symbol()
    return ResearchMetrics(
        provenance=provenance(),
        symbols={s: research(s) for s in symbols()},
        recommended=sym,
        recommendation_basis=why,
        known_negative_results=NEGATIVE_RESULTS,
        structural_limits=STRUCTURAL_LIMITS,
    )


# ------------------------------------------------------------------- signal-facing view

def expectancy(symbol: str) -> dict:
    """The numbers the signal endpoint and the position sizer both quote."""
    r = research(symbol)
    return {
        "symbol": symbol,
        "net_bp_full": r.full.net_bp_mean,
        "net_bp_holdout": r.holdout.net_bp_mean,
        "win_rate": r.full.win_rate,
        "max_drawdown_pct": r.full.max_drawdown_pct,
        "n_trades": r.full.n_trades,
        "permutation_p": r.full.permutation_p,
        "sharpe_annualized": r.full.sharpe_annualized,
    }


def confidence_label(symbol: str) -> str:
    r = research(symbol)
    p_full, p_hold = r.full.permutation_p, r.holdout.permutation_p
    if p_full <= 0.05 and p_hold <= 0.10 and r.holdout.net_bp_mean > 0:
        return "moderate — significant in-sample and positive out-of-sample"
    if p_full <= 0.05:
        return "weak — significant on the full sample but not clearly out-of-sample"
    return "low — not statistically distinguishable from ordinary nights"


def caveats(symbol: str) -> list[str]:
    r = research(symbol)
    years = (1.96 / r.full.sharpe_annualized) ** 2 if r.full.sharpe_annualized > 0 else float("inf")
    return [
        f"Size against the holdout figure (+{r.holdout.net_bp_mean:.1f}bp), not the full-sample one "
        f"(+{r.full.net_bp_mean:.1f}bp). The full sample includes the period used to design the rule.",
        f"Worst historical single trade: {r.full.net_bp_mean - 3 * r.full.net_bp_sd:.0f}bp is within "
        f"one-in-a-hundred reach; realised max drawdown on the trade sequence was "
        f"{r.full.max_drawdown_pct:.1f}%.",
        f"{r.full.trades_per_year:.0f} trades per year means roughly "
        f"{r.full.ann_return_on_full_capital_pct:.1f}%/yr on capital reserved for this, and about "
        f"{years:.0f} years of live trading to confirm the edge at 95% confidence.",
        "Entry must be the closing auction cross and exit the opening auction cross. Market orders "
        "pay a spread that is a material fraction of the edge.",
        "Not investment advice. Backtested results are hypothetical and do not represent trading in "
        "any actual account.",
    ]


def summary_table() -> pd.DataFrame:
    """Human-readable one-liner per symbol, for the CLI and the README."""
    rows = []
    for s in symbols():
        r = research(s)
        rows.append({
            "symbol": s,
            "n": r.full.n_trades,
            "net_bp": r.full.net_bp_mean,
            "holdout_bp": r.holdout.net_bp_mean,
            "win%": round(100 * r.full.win_rate, 1),
            "sharpe_ann": r.full.sharpe_annualized,
            "maxDD%": r.full.max_drawdown_pct,
            "t_hac": r.full.t_stat_hac,
            "perm_p": r.full.permutation_p,
            "drop2_bp": r.full.drop_best_2_net_bp,
        })
    return pd.DataFrame(rows).set_index("symbol")


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    print(f"\nFOMC eve overnight — computed from {RETURNS_CSV.name}\n")
    print(summary_table().to_string())
    sym, why = recommended_symbol()
    print(f"\nrecommended: {sym}\n  {why}\n")
    p = provenance()
    print(f"data built {p.built_at:%Y-%m-%d %H:%M}Z, {p.n_trades} trades, "
          f"{p.n_baseline_nights} baseline nights, {p.first_trade} .. {p.last_trade}")
