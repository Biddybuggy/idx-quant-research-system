# Does anything work when the market is weak?

**Date:** 2026-07-25 · **Data:** 21 IDX large caps, 2010-01-04 → 2026-07-10 (4,061 trading days)
**Risk-off definition:** JCI below its 200-day SMA (2% hysteresis) — 1,315 days, 32% of the sample.

## Why this study exists

The momentum engine goes to cash whenever JCI is below its long-term trend. That is
correct for momentum and useless as a product: for a third of all trading days the
dashboard has nothing to say, and those are precisely the days a reader wants help.

An earlier attempt at filling that gap — a "short-term opportunity" panel that ranked
stocks *outperforming* the index — was backtested and lost 20.7%/yr gross during
risk-off periods against JCI's −6.4%. It was removed. This study asks whether anything
else fills the gap, starting from the observation that a signal reliably pointing the
wrong way is still a signal.

## Phase 1 — Which signals predict anything?

Cross-sectional rank IC (Spearman) of signal at *t* against forward excess return
(stock return minus equal-weight universe return), overlapping windows, t-stats
haircut by effective sample size *n/h*. Reproduce with
`.venv/bin/python scripts/research/ic_sweep.py`; the ranking that mattered:

| Signal | IC (5d, all) | t | IC (5d, risk-off) | t |
|---|---|---|---|---|
| **−rs5** (buy 5-day laggards) | **+0.070** | 7.3 | **+0.080** | 4.5 |
| −rsi14 (buy oversold) | +0.046 | 4.5 | +0.066 | 3.5 |
| −rs20 | +0.031 | 3.0 | +0.050 | 2.7 |
| 12-1 momentum (live system) | +0.019 | 1.8 | −0.004 | −0.2 |
| **rs20 (the removed panel)** | **−0.031** | −3.0 | **−0.050** | −2.7 |

At the 20-day horizon the ordering shifts toward slower signals: **−rs60** is the best
risk-off predictor (IC +0.092, t=2.3) while being *nothing* in risk-on (+0.008). The
whole reversal family is positive; the whole momentum/trend family is negative in
risk-off. The removed panel sits at the bottom, which is the same finding as before
with the sign made explicit.

## Phase 2 — Which of those survive costs?

Everything below runs through the real engine: 0.15%/0.25% commissions, 0.10%
half-spread, 0.10% slippage (~0.5% round trip), 100-share lots, the ADV liquidity
floor and participation cap, and the 20% drawdown halt.

Risk-off annualised return:

| Strategy | turnover/yr | 2010–18 | 2019–26 | full |
|---|---|---|---|---|
| **−rs60, top 5, monthly, risk-off only** | 6.5× | **+18.2%** | **+19.6%** | **+17.2%** |
| −rs60, top 5, monthly, always on | 11.2× | −4.4% | +27.9% | +8.4% |
| −rsi14, top 5, monthly | 15.7× | −0.5% | +15.0% | +6.6% |
| −rs5, top 5, **weekly** | 72.2× | −5.7% | −23.6% | −14.9% |
| rs20 top 5 weekly *(the removed panel)* | 65.2× | −28.7% | −21.7% | −20.4% |
| JCI buy & hold | — | +5.0% | −16.8% | −6.0% |
| BBCA buy & hold | — | +15.2% | −0.4% | +7.4% |

**The headline is the −rs5 row.** It has by far the strongest raw signal in the study
(IC 0.070, t=7.3) and is the worst strategy in the study after costs. Weekly re-ranking
turns over 72× a year against 0.5% friction. Statistical significance and profitability
are different questions, and on IDX retail costs the second one is decided by turnover.

## Phase 3 — Four attempts to kill the survivor

**A. Parameter surface.** lookback {40,60,90,120} × top_n {3,5,8} × rebalance {10,21,42}:
**36 of 36 cells positive**, median +11.7%, all beating JCI. The neighbourhood works,
not one cell. Defaults are set to the centre of the surface, not its best cell.

**B. Per-episode.** 12 contiguous risk-off stretches ≥20 days. Positive in 7, beat JCI
in 9. Mean excess **+8.7pp per episode**, bootstrap 95% CI [+1.3, +18.3],
P(mean ≤ 0) = 0.007. Sign test on 9-of-12 is p=0.073 — marginal on its own.
Dropping the single best episode (2020, +34.3% vs JCI −16.6%) still leaves +4.9pp.

**C. Cost stress.** +17.2% at 1× friction, +11.0% at 2×, +5.6% at 3×.

**D. No-lookahead.** Verified by truncation, pinned in `tests/test_reversal.py`.

## The qualification that matters

Our universe is 21 names chosen because they are liquid **today**. A buy-the-losers rule
is exactly the rule that, in a real point-in-time universe, would also have bought the
names that kept falling to zero — and this backtest can never buy those, because they
are not in the list. Dropping the four commodity cyclicals (ANTM/INCO/ADRO/PTBA), whose
survival was least assured:

| Universe | risk-off ann, full | risk-off ann, **out-of-sample** |
|---|---|---|
| all 21 | +17.2% | **+19.6%** |
| ex-commodity (17) | +13.4% | **+4.1%** |
| banks + staples (9) | +14.1% | **+0.5%** |

The full-period edge survives everywhere. **The out-of-sample edge does not** — it
concentrates almost entirely in the four names most exposed to the bias. That makes
this promising and far better evidenced than what it replaces; it does not make it
proven.

## What shipped

- `idxquant/strategies/reversal.py` — `RiskOffReversal`, the strategy above.
- `idxquant/strategies/regime_switch.py` — momentum when risk-on, reversal when
  risk-off. Highest CAGR tested (+9.1% full, +11.6% OOS) **and the deepest drawdown
  tested (−62.2%)**, because the momentum leg is weak on this universe (+3.3%/yr,
  −50% drawdown on its own) and liquidates into the regime flip.
- `idxquant/research/riskoff.py` + a dashboard panel that renders **only** in risk-off,
  reads the same `rank_frame()` the strategy trades, and recomputes the live episode
  through the costed engine on every render.
- Both new strategies are registered in the factory but **`momentum` remains the
  default**. Switching the live paper portfolio is a config decision for the operator.

## Reproducing

Every number above regenerates from the committed scripts (a few minutes total):

```
.venv/bin/python scripts/research/ic_sweep.py      # phase 1, the IC table
.venv/bin/python scripts/research/bt_reversal.py   # phase 2, costed backtests
.venv/bin/python scripts/research/bt_robust.py     # phase 3, attacks A-C
.venv/bin/python scripts/research/bt_combine.py    # the combined-portfolio diagnosis
.venv/bin/python scripts/research/bt_survivor.py   # survivorship probe + bootstrap
```

## Open threads

1. **The momentum leg needs its own review.** +3.3%/yr with a 50% drawdown is worse
   than the index, and it is what the paper portfolio trades today. Not caused by the
   drawdown halt or by `top_n` — checked both.
2. **Survivorship.** The honest fix is a point-in-time universe with delisted names.
   Until then, treat live results as the test.
3. The sign test (p=0.073) wants more episodes. Only time supplies those.
