"""Fixed-parameter market-neutral test: no grid, no selection, nothing to overfit.

Parameters are pinned a priori from dispersion_check.py (252-day lookback, the only setting with
a positive raw payoff) rather than fitted, so the whole 2015-2026 span is a single honest sample.
The regime filter is off by construction: a dollar-neutral book has no market exposure to hedge.
"""
import numpy as np, pandas as pd, statsmodels.api as sm
from backtester import compute_metrics
from backtester_xsec import run_xsec_backtest
from data_loader import BENCH, CASH, INDUSTRY, load_daily
from signals_xsec import build_panel

data = load_daily(INDUSTRY + [BENCH, CASH], start="2014-01-01")
held = [s for s in INDUSTRY if s in data]
px = build_panel(data)

rows = []
for n in (4, 6, 8):
    p = {"lookback": 252, "skip": 21, "n_sel": n, "short_book": 0.5, "long_book": 0.5,
         "use_regime": False, "require_positive": False, "max_weight": 0.35, "vol_window": 60}
    res = run_xsec_backtest(data, held, params=p, start=pd.Timestamp("2015-01-01"))
    r, m = res["net_ret"], res["metrics"]
    spy = px[BENCH].pct_change().reindex(r.index)
    df = pd.concat([r.rename("s"), spy.rename("b")], axis=1).dropna()
    fit = sm.OLS(df["s"], sm.add_constant(df["b"])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    rows.append({"n_sel": n, "sharpe": m["sharpe"], "sortino": m["sortino"],
                 "cagr_%": 100*((m["final_equity"]/10_000)**(1/yrs) - 1),
                 "maxdd_%": m["max_drawdown_pct"], "pf": m["profit_factor"],
                 "alpha_%/yr": 252*100*fit.params.iloc[0], "beta": fit.params.iloc[1],
                 "t_alpha": fit.tvalues.iloc[0], "cost_drag_%": 100*res["cost_drag"]})

out = pd.DataFrame(rows)
print("\n=== MARKET-NEUTRAL, FIXED PARAMS, 2015-01 .. present (no selection) ===")
print(out.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
spy_m = compute_metrics(10_000*(1+px[BENCH].pct_change()[px.index >= "2015-01-01"].fillna(0)).cumprod(), pd.DataFrame(), 1)
print(f"\nSPY same window: sharpe={spy_m['sharpe']:.3f}  maxdd={spy_m['max_drawdown_pct']:.2f}%")
print("\nt_alpha > 1.96 would mean the market-neutral factor payoff is real after costs.")
