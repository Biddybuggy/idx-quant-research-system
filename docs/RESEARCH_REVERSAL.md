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
| **−rs60, top 5, monthly, risk-off only** | 6.5× | **+18.2%** | **+21.1%** | **+17.9%** |
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

## Addendum, same day: a bug, and the live strategy switch

**A mark-to-market bug.** While testing an always-invested composite, an
"own every liquid name" portfolio showed a −51.9% loss over Oct 2018 → Jun 2019,
a window in which JCI rose +6.6%. That is not a strategy result, it is a defect.

Cause: on **2019-06-19**, ten of the 21 names had no close in the vendor data —
the only such day in 4,061. `step()` marked to market by summing positions whose
close was present, i.e. valuing the rest at **zero**. Equity fell 39% for one
day, tripped the 20% drawdown halt, liquidated the book at the next open and
blocked re-entry for 20 trading days. Equity was normal again the next day: the
loss never happened, the forced liquidation did.

Fixed in `build_frames` (bounded forward-fill of the marking frame only — never
of `opens`, since you cannot fill an order at a price that never printed) plus an
entry-price floor inside `step()` for gaps too long to bridge. Pinned by
`tests/test_missing_price_mark.py`. **The reversal results above were unaffected**
— the strategy was in cash on that day. Momentum was affected, by ~5pp of drawdown.

**Research approximations are not the shipped strategy.** A second correction:
`bt_composite.py` rebalanced momentum on a fixed 21-day grid, where the real
`CrossSectionalMomentum` uses calendar month-ends plus an absolute-momentum gate.
That gap alone moved max drawdown from −38.1% to −44.7% and flipped the
composite's 2×-cost verdict from fail to pass. Anything quoted in a docstring,
in this document, or on the dashboard now comes from
`scripts/research/bt_live_config.py`, which runs what `make_strategy()` returns.

**The live strategy changed** from `momentum` to `regime_switch` (momentum top 8
when risk-on, reversal top 5 when risk-off), measured as shipped:

| full 2010–2026 | CAGR | Sharpe | maxDD | risk-off |
|---|---|---|---|---|
| momentum *(was live)* | +2.2% | 0.22 | −41.5% | −8.5% |
| reversal only | +7.8% | 0.55 | **−29.1%** | **+17.9%** |
| **regime_switch** *(now live)* | **+13.0%** | **0.66** | −44.7% | +6.4% |
| JCI buy & hold | +5.2% | 0.39 | −41.5% | −6.0% |
| BBCA buy & hold | +13.8% | 0.63 | −51.8% | +7.4% |

Out-of-sample the ordering is closer: regime_switch +9.7%/0.52/−46.7% against
reversal +7.7%/**0.54**/**−24.8%**. Reversal has the better out-of-sample
risk-adjusted return and half the drawdown; regime_switch has the higher return
and is invested 89% of the time instead of 28%. The composite is live because
the product needs a portfolio that is doing something in both regimes — a
product decision, not a claim that it is the better strategy.

Why the composite beats the momentum it replaced: **concentration, not selection.**
In risk-on periods, top 3 (+9.4%/yr), top 8 (+9.6%) and owning every liquid name
(+9.6%) are indistinguishable. Holding only 3 names took idiosyncratic risk for
no return. `switch_top_n=8` is not a clean plateau either (Sharpe by top_n:
0.60 / 0.43 / 0.66 / 0.71 / 0.61 at 3 / 5 / 8 / 10 / 13); 8 was written to config
before that table existed and left there rather than moved to the best cell.

## What shipped

- `idxquant/strategies/reversal.py` — `RiskOffReversal`, the strategy above.
- `idxquant/strategies/regime_switch.py` — momentum top 8 when risk-on, reversal
  top 5 when risk-off. **Now the live strategy**; figures in the addendum below.
- `idxquant/research/riskoff.py` + a dashboard panel that renders **only** in risk-off,
  reads the same `rank_frame()` the strategy trades, and recomputes the live episode
  through the costed engine on every render.
- `idxquant/backtest/engine.py` — the mark-to-market fix, plus
  `tests/test_missing_price_mark.py`.
- `scripts/research/bt_live_config.py` — the authority for any quoted number,
  because it measures the strategy the factory actually builds.

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

1. **Survivorship.** The honest fix is a point-in-time universe including delisted
   names. Until then, treat live results as the test.
2. The sign test (p=0.073) wants more episodes. Only time supplies those.
3. **`switch_top_n` is noisy** — Sharpe dips to 0.43 at top_n=5, between 0.60 at 3
   and 0.66 at 8. Worth understanding rather than tuning around.
4. **Nothing reads data-quality at backtest time.** The 2019-06-19 gap was invisible
   until it corrupted a result. Ingest already writes `data_quality` rows; a run that
   marks positions on carried-forward prices should say so out loud.
5. **Momentum standalone** measured +2.2%/yr with a −41.5% drawdown. It is no longer
   live, but it is still registered and still unexplained.
