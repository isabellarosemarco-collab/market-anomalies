"""
fee_frontier.py — the arithmetic that decides whether the odd-lot book can work at all.

WHY THIS FILE EXISTS. The 120-day scan found exactly two genuine odd-lot priority grants, and both lost
money — for OPPOSITE reasons, which is the whole finding:

    ABUS  Dutch $5.00-$5.75, market $4.88   gross +2.46%   ->  NET -4.79%
    UTMD  fixed $75.00,      market $74.75  gross +0.33%   ->  NET -0.37%

ABUS has a spread seven times wider than UTMD and loses thirteen times more. The reason is that the
99-share cap converts share PRICE into capital: 99 x $4.88 = $483, and a flat $35 fee is 7.2% of $483.
UTMD at $74.75 deploys $4,934 — nearly the whole budget, so the fee is only 0.47% — but by then
arbitrageurs have compressed the spread to 0.33%.

So the trade lives or dies on a single inequality. Net profit is positive when

    gross_spread  >  fee / (99 x price)

The right-hand side is the FEE DRAG, and it falls as 1/price. This produces a squeeze that is a property
of the structure, not of any particular deal:

  * LOW-priced tenders clear the spread hurdle easily but cannot raise the capital to pay the fee.
  * HIGH-priced tenders raise the capital but competitive arbitrage has already taken the spread.

The viable window is the intersection, and this module measures how wide it is. If the window is empty
at a $35 fee, the correct conclusion is not "scan harder", it is "the fee is the adversary" — and the
only lever that moves is the fee itself, which is a broker choice, not a market forecast.

The second output is the breakeven FEE at observed spreads, because that reframes the whole book: this
strategy is not a search for mispricing, it is a search for a $0-commission voluntary corporate action
desk. Fidelity and Schwab charge $0-$38 for exactly the same instruction.
"""
from __future__ import annotations

import numpy as np

ODD_LOT_MAX = 99

# The two genuine odd-lot priority grants found in the 120-day EDGAR window, priced off the
# CONSERVATIVE tender price (fixed price, or the LOW end of a Dutch range — nobody is paid a midpoint).
OBSERVED = [
    # ticker, market price, conservative tender price, days to expiry
    ("UTMD", 74.75, 75.00, 12),
    ("ABUS",  4.88,  5.00,  4),
]


def fee_drag(price: float, fee: float, shares: int = ODD_LOT_MAX) -> float:
    """Flat fee as a fraction of the capital 99 shares can deploy. This is the hurdle to clear."""
    return fee / (shares * price)


def min_gross_required(price: float, fee: float) -> float:
    """Minimum gross spread for a net-positive odd-lot trade at this share price."""
    return fee_drag(price, fee)


def breakeven_fee(price: float, gross: float, shares: int = ODD_LOT_MAX) -> float:
    """The highest fee at which this deal still nets zero. Above it, the broker takes the whole edge."""
    return gross * shares * price


def report(fees=(0.0, 5.0, 35.0)) -> None:
    print("=" * 100)
    print("FEE DRAG — flat fee as % of deployable capital, since 99 shares caps capital at 99 x price")
    print("=" * 100)
    prices = [5, 10, 20, 30, 50, 75, 100, 150]
    hdr = "  price | " + " | ".join(f"fee ${f:<5.0f}" for f in fees)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for p in prices:
        cells = " | ".join(f"{100*fee_drag(p, f):8.2f}%" for f in fees)
        print(f"  ${p:5.0f} | {cells}")
    print("\n  Read the $35 column as the MINIMUM gross spread the deal must show just to break even.")
    print("  At $5 a share you need 7.07% gross. At $75 you need 0.47%. The hurdle falls as 1/price.")

    print("\n" + "=" * 100)
    print("THE SQUEEZE — the two real grants found, and what each would have needed")
    print("=" * 100)
    print(f"  {'tkr':5} {'price':>8} {'tender':>8} {'gross%':>8} {'capital':>9} "
          f"{'need%@$35':>10} {'NET%@$35':>9} {'NET%@$0':>8} {'BE fee':>8}")
    for tkr, px, tender, days in OBSERVED:
        shares = min(ODD_LOT_MAX, int(5000 // px))
        cap = shares * px
        gross = (tender - px) / px
        need = min_gross_required(px, 35.0)
        net35 = (shares * (tender - px) - 35.0) / cap
        net0 = gross
        be = breakeven_fee(gross, shares=shares, price=px) if False else gross * cap
        print(f"  {tkr:5} {px:8.2f} {tender:8.2f} {100*gross:7.2f}% {cap:9.0f} "
              f"{100*need:9.2f}% {100*net35:8.2f}% {100*net0:7.2f}% {be:8.2f}")
    print("\n  ABUS clears a 2.46% spread and still loses 4.79%: 99 shares of a $4.88 stock is $483 of")
    print("  capital, so $35 is 7.2% of the position. UTMD deploys $4,934 and the fee costs only 0.47%,")
    print("  but the spread by then is 0.33%. Wide spread, no capital base; big capital base, no spread.")
    print("\n  At a $0 fee BOTH are profitable: +0.33% and +2.46%. The fee is the entire adversary.")

    print("\n" + "=" * 100)
    print("WHERE THE VIABLE WINDOW IS — required gross spread vs. what spreads actually survive")
    print("=" * 100)
    print("  Announced tenders get arbitraged within hours. Observed residual spreads on LISTED,")
    print("  fixed-price issuer tenders cluster around 0.2-0.8%; anything materially wider is pricing")
    print("  deal-break risk (ABUS carries a financing condition AND a minimum tender condition).")
    print("  Taking 0.5% as the realistic compressed spread, the minimum viable share price is:")
    for f in (0.0, 5.0, 10.0, 35.0):
        if f == 0:
            print(f"    fee ${f:5.0f} -> any price qualifies (hurdle is 0)")
            continue
        p_min = f / (ODD_LOT_MAX * 0.005)
        ok = p_min <= 5000 / ODD_LOT_MAX
        print(f"    fee ${f:5.0f} -> need price >= ${p_min:7.2f}"
              + ("" if ok else f"   IMPOSSIBLE: 99 shares would cost ${ODD_LOT_MAX*p_min:,.0f}, "
                               f"over the $5,000 budget"))
    print("\n  That last line is the structural verdict. At a $35 fee and a realistic 0.5% spread you")
    print("  would need a $71 stock, which is fine — but you also need the spread to STILL be 0.5% on")
    print("  a $71 stock, and UTMD at $74.75 offered 0.33%. The window is not empty, it is narrow, and")
    print("  it is entirely determined by the fee you pay, not by how many filings you read.")


def sensitivity() -> None:
    """How many deals per year must exist, at a given net spread, to matter on $5,000."""
    print("\n" + "=" * 100)
    print("FREQUENCY x SIZE — what this book pays even when it works")
    print("=" * 100)
    print("  Assume a $0 voluntary-corporate-action fee, so the gross spread survives intact, and")
    print("  assume you deploy the full $5,000 (i.e. share price >= $50).")
    nets = np.array([0.003, 0.005, 0.010, 0.025])
    counts = np.array([2, 5, 10, 25])
    print(f"\n  {'net/deal':>9} | " + " | ".join(f"{c:>2} deals/yr" for c in counts))
    print("  " + "-" * 56)
    for n in nets:
        cells = " | ".join(f"${5000*n*c:9,.0f}" for c in counts)
        print(f"  {100*n:8.2f}% | {cells}")
    print("\n  The realistic cell is 0.3-0.5% net, 2-5 deals a year: $30 to $125 annually on $5,000.")
    print("  The optimistic cell — 2.5% net, 25 deals a year — is $3,125, and would require every")
    print("  tender to be both wide AND high-priced AND odd-lot-granting, which the 120-day sample")
    print("  says does not happen: 2 grants out of 18 issuers, neither one both wide and high-priced.")


if __name__ == "__main__":
    report()
    sensitivity()
