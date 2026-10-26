"""
spread_calc.py — what 99 shares of this actually pays, after the costs nobody puts in the spreadsheet.

THE FEE IS NOT A ROUNDING ERROR AT THIS SIZE, IT IS THE TRADE. A voluntary corporate action (which is
what tendering is) carries a broker fee that is FLAT, typically $0-$38 depending on the broker. On a
99-share position in a $40 stock — $3,960 of capital — a 3% gross spread is $119. A $35 fee takes 29% of
that. Flat costs against a capped position size is the defining economics of a capacity-constrained
trade, and any model that quotes gross spread is overstating the return by a third. So the headline
number here is NET.

A WIDE SPREAD IS A WARNING, NOT A PRIZE. This is the part that inverts naive screening. Once a tender is
announced, arbitrageurs compress the stock to just under the offer price within hours. A spread still
sitting at 8% two weeks later is not an opportunity the market missed; it is the market pricing a
probability that the offer does not complete — a financing condition, a minimum tender condition, a
regulatory hold, or a Dutch auction the market expects to clear at the bottom of its range. So
`risk_flags` gets populated from the parsed conditions, and spreads above `SUSPICIOUS_SPREAD` are
labelled as suspicious rather than promoted to the top of the report.

WHY ANNUALISING IS REPORTED BUT NOT RANKED ON. A 2% return over 9 days annualises to 122%, which is a
true statement about a number and a false statement about a strategy: you cannot redeploy into another
tender that does not exist yet. The binding constraint is deal FREQUENCY, exactly as it was for the
8-night FOMC trade. Annualised return is shown for comparison between simultaneous deals only.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

ODD_LOT_MAX = 99                 # the whole point: 99 shares, never 100
DEFAULT_FEE = 35.0               # broker voluntary corporate action fee; set to yours
SUSPICIOUS_SPREAD = 0.06         # above this, the spread is probably compensation for deal risk
MIN_DAYS = 5                     # need time to buy, settle (T+1) and submit the election
IMPLAUSIBLE_SPREAD = 0.30        # above this, assume the PARSER is wrong, not the market


@dataclass
class Economics:
    ticker: str
    company: str
    kind: str
    market_price: float
    tender_low: float | None
    tender_high: float | None
    shares: int
    capital: float
    gross_profit: float
    net_profit: float
    gross_spread: float
    net_spread: float
    days_left: int | None
    annualised: float | None
    fee: float
    risk_flags: list[str]
    verdict: str

    def row(self) -> dict:
        return {
            "ticker": self.ticker, "company": self.company[:28], "type": self.kind,
            "mkt": round(self.market_price, 2),
            "tender_lo": self.tender_low, "tender_hi": self.tender_high,
            "sh": self.shares, "capital": round(self.capital),
            "gross%": round(100 * self.gross_spread, 2),
            "NET%": round(100 * self.net_spread, 2),
            "net_$": round(self.net_profit, 2),
            "days": self.days_left,
            "ann%": round(100 * self.annualised, 1) if self.annualised is not None else None,
            "verdict": self.verdict,
            "flags": "; ".join(self.risk_flags) or "-",
        }


def _days_to(expiry: str | None) -> int | None:
    if not expiry:
        return None
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return (datetime.strptime(expiry, fmt).date() - date.today()).days
        except ValueError:
            continue
    return None


def _blank(ticker, company, terms, px, shares, capital, flags, verdict) -> Economics:
    return Economics(ticker, company, terms.kind, px or 0.0, terms.price_low, terms.price_high,
                     shares, capital, 0.0, 0.0, 0.0, 0.0, _days_to(terms.expiry), None,
                     DEFAULT_FEE, flags, verdict)


def evaluate(ticker: str, company: str, market_price: float, terms,
             fee: float = DEFAULT_FEE, max_capital: float = 5000.0) -> Economics | None:
    """Price the 99-share odd-lot trade against the CONSERVATIVE tender price.

    Ordered so that the disqualifying facts are checked BEFORE any return is computed. An earlier
    version computed the spread first and appended warnings after, which produced a table where
    "NET +1.70%" sat next to a flag saying the odd-lot provision did not exist. The number draws the
    eye; the flag does not. If the trade is not available, no return is reported at all.
    """
    # NaN slips through yfinance for halted/suspended tickers and is not caught by `<= 0`.
    if market_price is None or not (market_price > 0) or market_price != market_price:
        return None
    flags = list(terms.conditions)
    shares = min(ODD_LOT_MAX, int(max_capital // market_price))
    capital = shares * market_price

    # --- category disqualifiers: no return is computed for any of these -------------------
    if terms.nontraded:
        return _blank(ticker, company, terms, market_price, 0, 0.0,
                      ["non-traded interval fund / private BDC — routine NAV repurchase, "
                       "no secondary market to buy in"], "EXCLUDE not traded")
    if terms.odd_lot == "denied":
        return _blank(ticker, company, terms, market_price, 0, 0.0,
                      ["filing EXPLICITLY denies odd-lot priority — odd lots are prorated "
                       "with everyone else"], "REJECT odd-lot denied")
    if terms.odd_lot == "absent":
        return _blank(ticker, company, terms, market_price, 0, 0.0,
                      ["no odd-lot language in the filing"], "REJECT no odd-lot")
    if terms.odd_lot == "ambiguous":
        return _blank(ticker, company, terms, market_price, 0, 0.0,
                      flags + ["odd-lot language present but intent unclear — read the filing "
                               "before assuming priority"], "REVIEW odd-lot unclear")
    if terms.kind == "nav_linked":
        pct = f"{terms.nav_pct:.0f}% of NAV" if terms.nav_pct else "a % of NAV"
        return _blank(ticker, company, terms, market_price, shares, capital,
                      flags + [f"priced at {pct} struck on the expiration date — no knowable "
                               "price at trade time, so no spread can be computed"],
                      "REVIEW NAV-linked")
    if not terms.priceable:
        return _blank(ticker, company, terms, market_price, shares, capital,
                      flags + ["offer price not parsed — read the filing"], "REVIEW manually")
    if terms.expiry_stale:
        return _blank(ticker, company, terms, market_price, shares, capital,
                      flags + [f"latest expiration found ({terms.expiry}) is in the past — "
                               "stale amendment, offer likely closed"], "EXCLUDE expired")
    if shares < 1:
        return _blank(ticker, company, terms, market_price, 0, 0.0,
                      flags + [f"share price ${market_price:,.2f} exceeds "
                               f"${max_capital:,.0f} budget"], "REJECT capital")

    # --- now the economics ---------------------------------------------------------------
    target = terms.conservative_price
    gross = shares * (target - market_price)
    net = gross - fee
    gs, ns = gross / capital, net / capital
    days = _days_to(terms.expiry)
    ann = ((1 + ns) ** (365 / days) - 1) if (days and days > 0 and ns > -1) else None

    if abs(gs) > IMPLAUSIBLE_SPREAD:
        return _blank(ticker, company, terms, market_price, shares, capital,
                      flags + [f"parsed offer ${target:,.2f} vs market ${market_price:,.2f} implies "
                               f"{100*gs:+.0f}% — implausible, assume a parse error"],
                      "REVIEW parse suspect")

    if days is not None and days < MIN_DAYS:
        flags.append(f"only {days}d left — may not settle in time to tender")
    if gs > SUSPICIOUS_SPREAD:
        flags.append(f"spread {100*gs:.1f}% is suspiciously wide — price in deal-break risk")
    if terms.kind == "dutch":
        flags.append("Dutch auction: elect the price-determined option or risk not being filled")

    if ns <= 0:
        verdict = "REJECT negative net"
    elif days is not None and days < MIN_DAYS:
        verdict = "REJECT too late"
    elif gs > SUSPICIOUS_SPREAD:
        verdict = "CAUTION wide spread"
    else:
        verdict = "CANDIDATE"

    return Economics(ticker, company, terms.kind, market_price, terms.price_low, terms.price_high,
                     shares, capital, gross, net, gs, ns, days, ann, fee, flags, verdict)


def quote(ticker: str, asof: str | None = None) -> float | None:
    """Close price. `asof` (YYYY-MM-DD) returns the CONTEMPORANEOUS close instead of the latest.

    WHY `asof` EXISTS — the look-ahead bug it kills. Pricing a closed offer against today's quote is
    not a backtest, it is a comparison between an old offer price and an unrelated later price. WIX
    filed its $80-$92 Dutch auction on 2026-03-05 with the stock at $92.94, i.e. ABOVE the entire
    range: tendering at the $80 floor was a guaranteed 13.9% loss. But WIX then fell to $77.53 over
    six months, and measuring the $80 floor against THAT produced a fictitious "+2.48% net" — the only
    qualifying deal the 365-day scan reported. The offer price is known as of the filing date, so the
    market price must be read as of the same date or the spread is meaningless.
    """
    try:
        import yfinance as yf
        if asof:
            from datetime import date, timedelta
            d = date.fromisoformat(asof)
            h = yf.Ticker(ticker).history(start=(d - timedelta(days=7)).isoformat(),
                                         end=(d + timedelta(days=2)).isoformat(),
                                         auto_adjust=False)
        else:
            h = yf.Ticker(ticker).history(period="5d", auto_adjust=False)
        if h.empty:
            return None
        s = h["Close"].dropna()
        return float(s.iloc[-1]) if len(s) else None
    except Exception:
        return None
