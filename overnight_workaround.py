"""
overnight_workaround.py — can selective trading beat the round-trip friction barrier?

Every prior test held the overnight position EVERY night, so the ~4bp round trip was paid 252 times
a year against a ~6.7bp/night gross edge. That leaves most of the gross premium on the table. Both
workarounds here attack the same arithmetic from the same direction: trade far less often, and only
on nights where the expected premium is much larger than average.

The bar is explicit. A round trip costs ~4bp, so a filter is interesting only if net expectancy per
trade is comfortably above that, and the target set here is 15bp/trade.

WORKAROUND 1 — MACRO CATALYST FILTER
  Nights selected by a pre-known calendar: FOMC decision eve, CPI morning, NFP morning, OpEx eve.
  No estimation, no parameters, nothing fitted — the calendar is published years ahead, so there is
  no in-sample/out-of-sample distinction to make and no way to overfit a threshold. That is the
  main attraction of this variant and the reason it is worth testing before anything with a knob.

WORKAROUND 2 — INTRADAY WASHOUT
  Nights following a capitulation session: a large negative open-to-close move on unusually heavy
  volume. Two parameters (both sigma thresholds), so this one CAN be overfit, which is why it is run
  on 2000-present rather than 2018-present and why the sigma windows are reported as a sensitivity
  grid instead of a single chosen value.

WHAT THIS TEST CANNOT ESCAPE — STATISTICAL POWER

Cutting to ~44 catalyst nights a year cuts the friction bill by ~83%, but it cuts the sample by the
same ~83%, and a t-statistic scales with sqrt(n). Filtering to one fifth of the nights therefore
costs a factor of ~2.2 in t unless per-night expectancy rises by more than that factor. The
always-on IWM book reached t=2.45 over 2,193 nights; the same edge concentrated into 350 nights
would need to be more than twice as large per night to look as convincing. So every table below
reports `years_needed` = (1.96/Sharpe)^2 — the sample length required for the result to be
distinguishable from zero — and a permutation p-value against random nights of the same count.

MULTIPLE COMPARISONS. Four event types x three symbols is twelve tests, and at p=0.05 one of them
is expected to look good by chance. The POOLED "any catalyst" filter is therefore pre-registered
here as the primary result; the per-event breakdown is diagnostic only, and a single event type
standing out while the pool does nothing should be read as noise, not as a finding.

CALENDAR PROVENANCE. FOMC dates are scraped from federalreserve.gov. NFP and OpEx are calendar
rules. CPI dates are hardcoded because bls.gov blocks scraping — so `event_calendar.validate()` runs
first and checks each calendar against the market signature its release time implies. A wrong date
list produces a null result rather than an error, which is the failure mode worth guarding against.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import event_calendar as ec
from overnight_auction import BROKERS, round_trip_cost
from overnight_drift import ANN, decompose, hac_t, load_ohlc_div, max_dd, stats

SYMBOLS = ["SPY", "QQQ", "IWM"]
SLIP_BP = 0.5                      # per side, auction execution
COMM = BROKERS["IBKR fixed $0.0035/sh"]
TARGET_BP = 15.0                   # the expectancy bar set for this test
EVENT_START = "2018-01-01"         # catalyst window
WASHOUT_START = "2000-01-01"       # daily-bar window for the washout proxy


# --------------------------------------------------------------------------- mechanics

def night_book(r: pd.DataFrame, entries: pd.Series) -> pd.DataFrame:
    """P&L of going long at the close of t-1 and flat at the open of t, on selected nights.

    `entries` is indexed by the EXIT date t (the morning the position is closed), which is how both
    workarounds naturally express themselves: "the night before the CPI print" is the night whose
    exit lands on the CPI date. Costs are charged on the entry close price.
    """
    act = entries.reindex(r.index).fillna(False).astype(float)
    cost = round_trip_cost(r["close"].shift(1), SLIP_BP, COMM)
    gross = r["r_night"] * act
    net = gross - act * cost
    return pd.DataFrame({"act": act, "gross": gross, "net": net, "cost": act * cost})


def trade_stats(bk: pd.DataFrame, label: str) -> dict:
    """Per-trade and annualized figures for a sparse book.

    Annualized Sharpe of a book that sits in cash between trades equals the per-trade Sharpe times
    sqrt(trades per year), so the two columns are the same number viewed at two horizons. Idle days
    would earn T-bills in reality; that raises the return but not the Sharpe of the alpha, so it is
    left out rather than credited.
    """
    tr = bk.loc[bk["act"] > 0]
    n, yrs = len(tr), len(bk) / ANN
    if n < 10:
        return {"filter": label, "trades": n}
    g, net = tr["gross"], tr["net"]
    st = stats(bk["net"], label, bk["act"])
    sh_trade = float(net.mean() / net.std(ddof=1)) if net.std(ddof=1) > 0 else np.nan
    return {
        "filter": label,
        "trades": n,
        "trades/yr": n / yrs,
        "gross_bp/trade": 1e4 * g.mean(),
        "cost_bp/trade": 1e4 * tr["cost"].mean(),
        "net_bp/trade": 1e4 * net.mean(),
        "sd_bp/trade": 1e4 * net.std(ddof=1),
        "hit_%": 100 * float((net > 0).mean()),
        "net_ann_%": st["ann_ret_%"],
        "sharpe_ann": st["sharpe"],
        "maxdd_%": st["maxdd_%"],
        "t_hac": hac_t(net),
        "years_needed": (1.96 / (sh_trade * np.sqrt(n / yrs))) ** 2
        if sh_trade and sh_trade > 0 else np.nan,
        "beats_15bp": bool(1e4 * net.mean() > TARGET_BP),
    }


def permutation_p(bk: pd.DataFrame, n_draw: int, n_iter: int = 5000, seed: int = 3) -> float:
    """P(random nights of the same count do at least as well), one-sided.

    With a few hundred trades the HAC t-statistic leans on a normality assumption that overnight
    returns do not satisfy. This asks the question directly: is the selected set better than an
    arbitrary set of nights the same size drawn from the same period?
    """
    pool = bk["r_night_all"].dropna().values
    if len(pool) <= n_draw or n_draw < 5:
        return float("nan")
    rng = np.random.default_rng(seed)
    obs = bk.loc[bk["act"] > 0, "gross"].mean()
    draws = np.array([rng.choice(pool, n_draw, replace=False).mean() for _ in range(n_iter)])
    return float((draws >= obs).mean())


# ------------------------------------------------------------------- workaround 1

def catalyst_entries(idx: pd.DatetimeIndex, cals: dict) -> dict[str, pd.Series]:
    """Entry sets keyed by EXIT morning.

    CPI and NFP print at 08:30, so the release date IS the exit morning and the overnight window
    contains the number. FOMC decides at 14:00 and OpEx settles during Friday's session, so for
    those two the position is held into the event, not through it: exit on the event morning means
    the trade is a pre-event drift bet with no event risk taken overnight.

    `fomc_after` is the mirror case — hold the night AFTER the decision, which does contain the
    market's digestion of the statement. It is not in the spec but it is the same trade shifted one
    session, and reporting only one side of an event would leave the obvious variant untested.
    """
    on = lambda d: pd.Series(True, index=idx.intersection(pd.DatetimeIndex(d)))
    nxt = lambda s: pd.Series(
        True, index=pd.DatetimeIndex(
            [idx[i + 1] for i in idx.get_indexer(s.index) if 0 <= i < len(idx) - 1]))

    e = {"FOMC eve": on(cals["FOMC"]), "CPI morning": on(cals["CPI"]),
         "NFP morning": on(cals["NFP"]), "OpEx eve": on(cals["OPEX"])}
    e["FOMC after"] = nxt(e["FOMC eve"])
    pooled = pd.DatetimeIndex(sorted(set().union(
        *[set(e[k].index) for k in ("FOMC eve", "CPI morning", "NFP morning", "OpEx eve")])))
    e["ANY catalyst (primary)"] = pd.Series(True, index=pooled)
    return e


def workaround1(dec: dict[str, pd.DataFrame], cals: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, pv = [], []
    for sym, d in dec.items():
        r = d[d.index >= EVENT_START]
        ents = catalyst_entries(r.index, cals)
        base = night_book(r, pd.Series(True, index=r.index))
        rows.append({"symbol": sym, **trade_stats(base, "every night (control)")})
        for name, ent in ents.items():
            bk = night_book(r, ent)
            bk["r_night_all"] = r["r_night"]
            st = trade_stats(bk, name)
            rows.append({"symbol": sym, **st})
            if st.get("trades", 0) >= 10:
                pv.append({"symbol": sym, "filter": name, "trades": st["trades"],
                           "gross_bp/trade": st["gross_bp/trade"],
                           "random_night_bp": 1e4 * float(r["r_night"].mean()),
                           "perm_p": permutation_p(bk, st["trades"])})
    return (pd.DataFrame(rows).set_index(["symbol", "filter"]),
            pd.DataFrame(pv).set_index(["symbol", "filter"]))


# ------------------------------------------------------------------- workaround 2

def washout_entries(d: pd.DataFrame, ret_win: int, vol_win: int = 20,
                    ret_sig: float = 2.0, vol_sig: float = 2.0) -> pd.Series:
    """Normalized daily intraday liquidation proxy, keyed by EXIT morning.

    Trigger on day t: the open-to-close move is below its rolling mean by `ret_sig` sigma AND
    full-day volume is above its rolling mean by `vol_sig` sigma. Both conditions are observable at
    16:00 on t, the entry is the close of t, and the exit is the open of t+1 — so the series is
    shifted forward one session to key it by exit date.

    Rolling statistics are shifted one day so the threshold is set by data strictly BEFORE t. Letting
    t into its own sigma estimate would inflate the very sigma it has to exceed, making the trigger
    harder to fire on exactly the days it is meant to catch.
    """
    rd = (d["close"] - d["open"]) / d["open"]
    m = rd.rolling(ret_win, min_periods=ret_win).mean().shift(1)
    s = rd.rolling(ret_win, min_periods=ret_win).std(ddof=1).shift(1)
    v = d["volume"].astype(float)
    vm = v.rolling(vol_win, min_periods=vol_win).mean().shift(1)
    vs = v.rolling(vol_win, min_periods=vol_win).std(ddof=1).shift(1)

    trig = (rd < m - ret_sig * s) & (v > vm + vol_sig * vs)
    exit_dates = [d.index[i + 1] for i in np.flatnonzero(trig.values) if i < len(d) - 1]
    return pd.Series(True, index=pd.DatetimeIndex(exit_dates))


def workaround2(raw: dict[str, pd.DataFrame], dec: dict[str, pd.DataFrame],
                windows=(20, 60, 120)) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, pv = [], []
    for sym in raw:
        r = dec[sym]
        base = night_book(r, pd.Series(True, index=r.index))
        rows.append({"symbol": sym, **trade_stats(base, "every night (control)")})
        for w in windows:
            ent = washout_entries(raw[sym], ret_win=w)
            bk = night_book(r, ent)
            bk["r_night_all"] = r["r_night"]
            st = trade_stats(bk, f"washout ret_win={w}")
            rows.append({"symbol": sym, **st})
            if st.get("trades", 0) >= 10:
                pv.append({"symbol": sym, "filter": f"washout ret_win={w}",
                           "trades": st["trades"], "gross_bp/trade": st["gross_bp/trade"],
                           "random_night_bp": 1e4 * float(r["r_night"].mean()),
                           "perm_p": permutation_p(bk, st["trades"])})
    return (pd.DataFrame(rows).set_index(["symbol", "filter"]),
            pd.DataFrame(pv).set_index(["symbol", "filter"]))


# ------------------------------------------------- holdout for the surviving candidate

def fomc_holdout(start: str = "1996-01-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    """The decisive test for FOMC eve: 22 years of data the filter was not chosen on.

    A calendar filter is the rare case where a clean holdout is actually available. The filter was
    selected by looking at 2018-present, so 2018-present is in-sample by construction no matter how
    unfitted the rule feels. The Fed's historical pages reach back to 1996, giving ~175 meetings of
    untouched data. If pre-event drift into FOMC decisions is a real feature of how the market
    positions, it should be visible there too; if it only exists after 2018, it was selection.
    """
    fomc = ec.fomc_dates()
    raw = load_ohlc_div(SYMBOLS, start=start)
    rows, blocks = [], []
    for sym, d in raw.items():
        r = decompose(d)
        ent_all = pd.Series(True, index=r.index.intersection(pd.DatetimeIndex(fomc)))
        for lab, sl in (("HOLDOUT pre-2018", r.index < EVENT_START),
                        ("in-sample 2018+", r.index >= EVENT_START),
                        ("full history", r.index == r.index)):
            w = r[sl]
            if len(w) < 250:
                continue
            bk = night_book(w, ent_all)
            bk["r_night_all"] = w["r_night"]
            st = trade_stats(bk, lab)
            if st.get("trades", 0) >= 10:
                st["perm_p"] = permutation_p(bk, st["trades"])
                st["net_bp_ex_best2"] = 1e4 * float(
                    bk.loc[bk["act"] > 0, "net"].sort_values().iloc[:-2].mean())
            rows.append({"symbol": sym, "period": f"{w.index[0].year}-{w.index[-1].year}", **st})

        # Five-year blocks: a real effect should not live in one regime.
        for lo in range(1996, 2026, 5):
            w = r[(r.index >= f"{lo}-01-01") & (r.index < f"{lo + 5}-01-01")]
            if len(w) < 250:
                continue
            bk = night_book(w, ent_all)
            tr = bk.loc[bk["act"] > 0]
            blocks.append({"symbol": sym, "block": f"{lo}-{lo + 4}", "trades": len(tr),
                           "net_bp/trade": 1e4 * tr["net"].mean() if len(tr) else np.nan,
                           "hit_%": 100 * float((tr["net"] > 0).mean()) if len(tr) else np.nan})
    keep = ["period", "trades", "net_bp/trade", "net_bp_ex_best2", "sd_bp/trade", "hit_%",
            "sharpe_ann", "maxdd_%", "t_hac", "perm_p", "years_needed"]
    t = pd.DataFrame(rows).set_index(["symbol", "filter"])
    return t[[c for c in keep if c in t.columns]], pd.DataFrame(blocks).pivot(
        index="block", columns="symbol", values="net_bp/trade")


def washout_stability(raw: dict[str, pd.DataFrame], dec: dict[str, pd.DataFrame],
                      ret_win: int = 60) -> pd.DataFrame:
    """Same split applied to the washout filter, so the two workarounds are judged alike.

    Note the asymmetry in what a split can prove here. FOMC eve has no fitted parameter, so pre-2018
    is a true holdout. The washout has two sigma thresholds chosen while looking at the whole series,
    so a split shows stability but cannot establish out-of-sample validity.
    """
    rows = []
    for sym, d in raw.items():
        r = dec[sym]
        ent = washout_entries(d, ret_win=ret_win)
        for lab, sl in (("pre-2018", r.index < EVENT_START), ("2018+", r.index >= EVENT_START)):
            w = r[sl]
            bk = night_book(w, ent)
            tr = bk.loc[bk["act"] > 0]
            rows.append({"symbol": sym, "period": lab, "trades": len(tr),
                         "net_bp/trade": 1e4 * tr["net"].mean() if len(tr) else np.nan,
                         "hit_%": 100 * float((tr["net"] > 0).mean()) if len(tr) else np.nan})
    return pd.DataFrame(rows).pivot(index="symbol", columns="period",
                                    values=["trades", "net_bp/trade", "hit_%"])


def washout_grid(raw: pd.DataFrame, r: pd.DataFrame) -> pd.DataFrame:
    """Is the washout result a plateau or a spike? A real effect should be insensitive to the exact
    sigma cut; a result that only exists at one corner of the grid is a fitted artifact."""
    rows = []
    for rs in (1.5, 2.0, 2.5, 3.0):
        for vs in (1.0, 2.0, 3.0):
            bk = night_book(r, washout_entries(raw, 60, ret_sig=rs, vol_sig=vs))
            tr = bk.loc[bk["act"] > 0]
            rows.append({"ret_sigma": rs, "vol_sigma": vs, "trades": len(tr),
                         "net_bp/trade": 1e4 * tr["net"].mean() if len(tr) else np.nan})
    return pd.DataFrame(rows).pivot(index="ret_sigma", columns="vol_sigma",
                                    values="net_bp/trade")


# -------------------------------------------------------------------------------- main

def fmt(df: pd.DataFrame) -> str:
    return df.to_string(float_format=lambda x: f"{x:,.2f}")


def main() -> int:
    cals = ec.build(EVENT_START, "2026-12-31")
    print("=" * 112)
    print("CALENDAR PROVENANCE AND VALIDATION")
    print("=" * 112)
    for k, v in cals.items():
        print(f"  {k:5} n={len(v):4}  {v[0].date()} .. {v[-1].date()}")
    print("\n  FOMC scraped from federalreserve.gov; NFP/OpEx are calendar rules; CPI hardcoded")
    print("  (bls.gov returns 403). Signature test: 08:30 releases should move the OVERNIGHT leg,")
    print("  14:00 decisions the INTRADAY leg. `expected=True` rows are the ones that must show a")
    print("  ratio above 1 — if they do not, the dates are wrong.\n")

    raw_ev = load_ohlc_div(SYMBOLS, start=EVENT_START)
    dec_ev = {s: decompose(d) for s, d in raw_ev.items()}
    print(fmt(ec.validate(dec_ev["SPY"], cals).set_index(["calendar", "session"])))

    print("\n" + "=" * 112)
    print(f"WORKAROUND 1 — MACRO CATALYST FILTER, {EVENT_START[:4]}-present, "
          f"{SLIP_BP}bp/side + IBKR $0.0035/sh")
    print("=" * 112)
    t1, p1 = workaround1(dec_ev, cals)
    print(fmt(t1))
    print("\n  --- permutation test vs random nights of the same count (5,000 draws) ---")
    print(fmt(p1))

    print("\n" + "=" * 112)
    print(f"WORKAROUND 2 — INTRADAY WASHOUT (daily proxy), {WASHOUT_START[:4]}-present")
    print("=" * 112)
    raw_w = load_ohlc_div(SYMBOLS, start=WASHOUT_START)
    dec_w = {s: decompose(d) for s, d in raw_w.items()}
    t2, p2 = workaround2(raw_w, dec_w)
    print(fmt(t2))
    print("\n  --- permutation test vs random nights of the same count (5,000 draws) ---")
    print(fmt(p2))

    print("\n  --- subperiod stability, ret_win=60 ---")
    print(fmt(washout_stability(raw_w, dec_w)))

    print("\n  --- parameter sensitivity, net bp/trade, IWM, ret_win=60 ---")
    print("  (rows = intraday-return sigma cut, cols = volume sigma cut)")
    print(fmt(washout_grid(raw_w["IWM"], dec_w["IWM"])))

    print("\n" + "=" * 112)
    print("HOLDOUT — FOMC EVE ON 1996-2017, DATA THE FILTER WAS NOT SELECTED ON")
    print("=" * 112)
    h, blk = fomc_holdout()
    print(fmt(h))
    print("\n  --- FOMC eve net bp/trade by 5-year block ---")
    print(fmt(blk))

    print("\n" + "=" * 112)
    print(f"VERDICT — does either filter clear {TARGET_BP:.0f}bp net per trade?")
    print("=" * 112)
    for nm, t in (("catalyst", t1), ("washout", t2)):
        w = t[t.index.get_level_values("filter") != "every night (control)"]
        w = w[w["net_bp/trade"].notna()]
        if w.empty:
            continue
        best = w["net_bp/trade"].idxmax()
        print(f"  {nm:9} best: {best[0]} / {best[1]:24} "
              f"{w.loc[best, 'net_bp/trade']:+7.2f}bp  n={int(w.loc[best, 'trades']):4}  "
              f"sharpe={w.loc[best, 'sharpe_ann']:.2f}  perm_p="
              f"{(p1 if nm == 'catalyst' else p2).get('perm_p', pd.Series()).get(best, float('nan')):.3f}")
        print(f"  {'':9} filters clearing {TARGET_BP:.0f}bp: "
              f"{int(w['beats_15bp'].sum())} of {len(w)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
