import pandas as pd, numpy as np
from data_loader import BENCH, CASH, INDUSTRY, SECTORS, load_daily
from signals_xsec import build_panel, momentum_score, formation_dates

rows=[]
for name, u in (("9 sectors", SECTORS), ("41 industries", INDUSTRY)):
    d = load_daily(u+[BENCH,CASH], start="2014-01-01")
    held=[s for s in u if s in d]; px=build_panel(d)
    for lb in (126, 252):
        m = momentum_score(px[held], lb, 21)
        f = [x for x in formation_dates(px.index) if x >= pd.Timestamp("2019-01-01") and x in m.index]
        sub = m.loc[f]
        # cross-sectional spread per formation date
        sd   = sub.std(axis=1, skipna=True).mean()
        rng  = (sub.max(axis=1)-sub.min(axis=1)).mean()
        # top-n minus bottom-n realised NEXT-month return = the raw factor payoff
        nxt = px[held].pct_change(21).shift(-21)
        pay=[]
        for dt in f:
            s = sub.loc[dt].dropna()
            if len(s) < 8: continue
            n = 4 if len(s) > 20 else 3
            r = nxt.loc[dt].reindex(s.index)
            top = r[s.nlargest(n).index].mean(); bot = r[s.nsmallest(n).index].mean()
            if np.isfinite(top) and np.isfinite(bot): pay.append(top-bot)
        pay=np.array(pay)
        rows.append({"universe":name,"lookback":lb,"mean_xsec_sd":round(sd,3),
                     "mean_max_min_spread":round(rng,3),"n_obs":len(pay),
                     "mean_top_minus_bot_%":round(100*pay.mean(),3),
                     "t_stat":round(pay.mean()/(pay.std(ddof=1)/np.sqrt(len(pay))),2)})
print(pd.DataFrame(rows).to_string(index=False))
