"""Risk-off mean reversion: buy the biggest 60-day LOSERS while JCI is weak.

This is the strategy the failed tactical screen implies once you read its sign.
That screen bought short-term relative-strength LEADERS and lost 20.7%/yr gross
in risk-off periods. The cross-sectional rank IC of 20-day relative strength vs
forward excess return was NEGATIVE (-0.047 risk-off). A signal that reliably
points the wrong way is a signal; you just have to turn it around.

Rules, all computed at the close of day t (the engine applies the entry lag):
  - signal   = -(60d stock return - 60d JCI return)  -> highest = worst laggard
  - universe = names passing the 20d ADV liquidity floor
  - hold     = the top `top_n` by that signal, re-ranked every `rebalance_days`
  - regime   = hold ONLY while JCI is below its regime SMA; flat when risk-on

The regime condition is not a filter bolted on afterwards, it is the hypothesis.
Reversal is a risk-premium story: in a falling market, forced sellers and
liquidity demanders push good names below fair value and the discount unwinds.
In a rising market the same names are just losers, and the IC confirms it
(+0.008 risk-on vs +0.092 risk-off at the 20-day horizon).

WHY 60 DAYS AND NOT 5. The 5-day version has a much stronger raw signal
(IC 0.070, t=7.3, vs 0.008 for 60d over all regimes) and is the WORST strategy
in the whole study after costs: -10.7%/yr, because weekly re-ranking turns over
72x a year against ~0.5% round-trip friction. The 60-day version turns over 6.5x.
Whatever survives here is chosen for surviving costs, not for signal strength.

EVIDENCE (21 names, 2010-01 to 2026-07, real engine, real costs, lots, liquidity
caps and the 20% drawdown halt; risk-off = JCI below its 200d SMA):

                                 in-sample   out-of-sample   full
                                  2010-18       2019-26     2010-26
    this strategy, risk-off ann    +18.2%        +19.6%      +17.2%
    JCI buy & hold,  risk-off       +5.0%        -16.8%       -6.0%
    BBCA buy & hold, risk-off      +15.2%         -0.4%       +7.4%
    full-period CAGR / Sharpe / maxDD:           +7.7% / 0.54 / -30.1%
    (JCI +5.2% / 0.39 / -41.5%,  BBCA +13.8% / 0.63 / -51.8%)

Robustness, all run before this file was written:
  - parameter surface: 36 of 36 cells (lookback 40/60/90/120 x top 3/5/8 x
    rebalance 10/21/42) have positive risk-off returns; median +11.7%.
    The neighbourhood works, not one lucky cell.
  - cost stress: +17.2% at 1x friction, +11.0% at 2x, +5.6% at 3x.
  - episodes: 12 contiguous risk-off stretches >= 20 days. Beat JCI in 9,
    positive in 7, mean excess +8.7pp/episode, bootstrap 95% CI [+1.3, +18.3],
    P(mean excess <= 0) = 0.007. Dropping the best episode (2020) leaves +4.9pp.
  - no-lookahead verified by truncation (tests/test_reversal.py).

THE QUALIFICATION THAT MATTERS, and the reason this is registered but not the
default strategy. Our universe is 21 names picked because they are liquid TODAY.
A buy-the-losers rule is precisely the rule that, in a real point-in-time
universe, would also have bought the names that kept falling to zero — and this
backtest can never buy those, because they are not in the list. Probing it by
dropping the four commodity cyclicals (ANTM/INCO/ADRO/PTBA), whose survival was
least assured and whose drawdowns are deepest:

    risk-off annualised        full 2010-26    OOS 2019-26
      all 21 names                +17.2%          +19.6%
      ex-commodity (17)           +13.4%           +4.1%
      banks + staples (9)         +14.1%           +0.5%

The full-period edge survives everywhere. The OUT-OF-SAMPLE edge does not: it
concentrates almost entirely in the four names most exposed to the bias. Read
honestly, that makes this promising and much better evidenced than what it
replaces — not proven. Treat the live numbers as the test.
"""
from __future__ import annotations

import pandas as pd

from ..features import indicators as ind

# Defaults are the centre of the tested parameter surface, not its best cell.
LOOKBACK = 60
TOP_N = 5
REBALANCE_DAYS = 21


class RiskOffReversal:
    def __init__(self, lookback: int = LOOKBACK, top_n: int = TOP_N,
                 rebalance_days: int = REBALANCE_DAYS, regime_sma: int = 200,
                 min_adv_idr: float = 10.0e9):
        self.lookback = lookback
        self.top_n = top_n
        self.rebalance_days = rebalance_days
        self.regime_sma = regime_sma
        self.min_adv_idr = min_adv_idr
        self.name = f"revoff_{lookback}_top{top_n}_rb{rebalance_days}"

    def rank_frame(self, prices: dict[str, pd.DataFrame],
                   index_close: pd.Series) -> pd.DataFrame:
        """The scoring frame: higher = further below the index over `lookback`
        days. Illiquid names are NaN so they can never be ranked. Shared with
        the dashboard so the screen and the backtest can never diverge."""
        closes = pd.DataFrame({t: df["Close"] for t, df in prices.items()}).sort_index()
        idx = index_close.reindex(closes.index).ffill()
        rel = closes.pct_change(self.lookback).sub(idx.pct_change(self.lookback), axis=0)
        adv = pd.DataFrame({t: ind.avg_daily_value(df) for t, df in prices.items()}) \
            .reindex(closes.index)
        return (-rel).where(adv >= self.min_adv_idr)

    def signals(self, prices: dict[str, pd.DataFrame],
                index_close: pd.Series) -> pd.DataFrame:
        score = self.rank_frame(prices, index_close)
        risk_off = ~ind.regime_filter(index_close, self.regime_sma) \
            .reindex(score.index).ffill().fillna(False).astype(bool)

        target = pd.DataFrame(0, index=score.index, columns=score.columns)
        held: list[str] = []
        for i, date in enumerate(score.index):
            if i % self.rebalance_days == 0:
                row = score.loc[date].dropna()
                held = list(row.nlargest(self.top_n).index) if len(row) >= self.top_n else []
            if held and risk_off.loc[date]:
                target.loc[date, held] = 1
        return target.astype(int)

    def signal_context(self, ticker: str, prices: dict[str, pd.DataFrame],
                       index_close: pd.Series) -> dict:
        """Per-ticker fields for the signal file, computed at the latest close."""
        score = self.rank_frame(prices, index_close)
        row = score.iloc[-1].dropna()
        ranks = row.rank(ascending=False)
        risk_off = not bool(ind.regime_filter(index_close, self.regime_sma).iloc[-1])
        s = float(row.get(ticker, float("nan")))
        rank = int(ranks[ticker]) if ticker in ranks.index else None
        active = risk_off and rank is not None and rank <= self.top_n
        return {
            "confidence": "medium" if active else "low",
            "reasoning": (
                f"{self.lookback}-day return vs JCI is {-s:+.1%}, rank {rank} of "
                f"{len(row)} from the weakest end; market regime is "
                f"{'risk-off' if risk_off else 'risk-on'}. This strategy only "
                f"holds while the regime is risk-off."
            ),
            "invalidation": (
                f"Exit at the next {self.rebalance_days}-day rebalance if the name "
                f"leaves the bottom {self.top_n} by {self.lookback}-day relative "
                f"return; exit immediately once JCI closes back above its "
                f"{self.regime_sma}-day SMA."
            ),
            "extra_risk": {f"rel_{self.lookback}d": round(-s, 4) if s == s else None,
                           "laggard_rank": rank},
        }
