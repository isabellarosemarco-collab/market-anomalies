"""
loophole.py — where the Kelly bound does NOT bind, computed rather than asserted.

The bound established earlier is correct: max annual log-growth = S^2/2 at ANY leverage, so 100x in a
year needs Sharpe 3.03. But that bound answers ONE question — "what leverage maximises my MEDIAN
outcome?" — and three assumptions inside it can each be attacked. This prices all three.

  L1  GOAL-MAXIMISING BET SIZING. Kelly maximises E[log W]. If the goal is FIXED at 100x and you do
      not care about the median, the right objective is P(reach 100x before ruin). That is a different
      optimisation, and its solution is deliberate overbetting. The bound never said 100x is
      impossible — it said 100x cannot be the EXPECTED outcome. Those are not the same claim, and the
      gap between them is a real, computable probability.
  L2  QUADRATURE. n uncorrelated strategies each at Sharpe s combine to s*sqrt(n), so the growth
      ceiling S^2/2 scales LINEARLY in n. This is the only loophole that raises the median, and it is
      what every multi-strategy fund is actually selling.
  L3  CAPACITY. Sharpe is not a property of a strategy; it is a property of a strategy AT A SIZE.
      Reasoned about in the writeup, not simulated — it is the one that matters most here.
"""
import numpy as np

rng = np.random.default_rng(20260924)
N, DAYS = 40_000, 252            # 40k paths -> SE ~0.05% on a 1% probability, plenty
TARGET, RUIN = 100.0, 0.30       # ruin = down 70%: a margin call, or the point a human stops

def p_goal(sharpe: float, vol: float) -> tuple[float, float, float]:
    """P(reach 100x before ruin) inside one year. GBM, daily rebalanced to constant vol."""
    dt = 1.0 / DAYS
    lr = ((sharpe * vol - 0.5 * vol**2) * dt
          + vol * np.sqrt(dt) * rng.standard_normal((N, DAYS)))
    lw = np.cumsum(lr, axis=1)
    up, dn = lw >= np.log(TARGET), lw <= np.log(RUIN)
    got, died = up.any(axis=1), dn.any(axis=1)
    # argmax gives first True; only compare when both happened
    win = got & (~died | (up.argmax(axis=1) < dn.argmax(axis=1)))
    return win.mean(), float(np.exp(np.median(lw[:, -1]))), died.mean()

print("=" * 98)
print("L1  THE MAXIMUM ACHIEVABLE P($5K -> $500K IN ONE YEAR), searched over every leverage choice")
print("=" * 98)
print(f"  ruin = down {100*(1-RUIN):.0f}%.  vol is the ANNUALISED vol you choose to run; Kelly vol = Sharpe.")
grid = np.arange(0.25, 8.01, 0.25)
for s in (0.78, 0.88, 1.5, 2.0, 3.03):
    res = [(p, v, m, d) for v in grid for p, m, d in [p_goal(s, v)]]
    p, v, m, d = max(res)
    kelly_p, kelly_m, kelly_d = p_goal(s, s)
    print(f"\n  Sharpe {s:4.2f}")
    print(f"    goal-max sizing: vol {v:4.2f} = {v/s:4.1f}x Kelly   "
          f"P(100x) {100*p:5.2f}%   P(ruin) {100*d:5.1f}%   median ${5000*m:>9,.0f}")
    print(f"    Kelly sizing:    vol {s:4.2f} = 1.0x Kelly   "
          f"P(100x) {100*kelly_p:5.2f}%   P(ruin) {100*kelly_d:5.1f}%   median ${5000*kelly_m:>9,.0f}")

print("\n" + "=" * 98)
print("L2  QUADRATURE — the ONLY loophole that moves the median. n uncorrelated edges at Sharpe 0.8")
print("=" * 98)
for n in (1, 2, 3, 5, 10, 20, 30, 40):
    s = 0.8 * np.sqrt(n)
    g = np.exp(s**2 / 2)
    print(f"  n={n:<3} combined Sharpe {s:5.2f}   growth ceiling {g:9.2f}x/yr   "
          f"${5000*g:>14,.0f}" + ("   <-- clears 100x" if g >= 100 else ""))
print("\n  Correlation is what kills this: at pairwise rho, combined Sharpe -> s*sqrt(n/(1+(n-1)rho)).")
for rho in (0.0, 0.1, 0.3, 0.6, 1.0):
    s10 = 0.8 * np.sqrt(10 / (1 + 9 * rho))
    print(f"    n=10, rho={rho:4.2f}  ->  Sharpe {s10:5.2f}   ceiling {np.exp(s10**2/2):7.2f}x/yr")
