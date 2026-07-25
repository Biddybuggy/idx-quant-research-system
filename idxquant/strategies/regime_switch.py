"""Always-invested composite: 12-1 momentum when JCI is risk-on, 60-day
reversal when it is risk-off.

This is the strategy shape the product asks for — something to say every day,
not only in bull markets. The two legs are deliberately opposite: momentum buys
what has been strong, reversal buys what has been weak, and the regime decides
which question is the right one. Neither leg is new; both are composed from the
existing modules so there is one implementation of each.

This is the LIVE strategy as of 2026-07-25 (config/settings.yaml). It replaced
standalone momentum, which measured below the index.

EVIDENCE. Every number below comes from scripts/research/bt_live_config.py,
which runs exactly what make_strategy() returns — not a research approximation
of it. That distinction is not pedantic: an earlier draft of this docstring
quoted -38.1% max drawdown from a script that rebalanced momentum on a fixed
21-day grid, where the real strategy uses calendar month-ends plus an
absolute-momentum gate. The real figure is -44.7%.

`switch_top_n=8`, 2010-01 to 2026-07, real costs, lots, liquidity caps, halt:

                                    CAGR   Sharpe   maxDD   risk-off ann
    full 2010-2026
      momentum only (was live)      +2.2%   0.22    -41.5%     -8.5%
      reversal only (risk-off)      +7.8%   0.55    -29.1%    +17.9%
      THIS composite               +13.0%   0.66    -44.7%     +6.4%
      JCI buy & hold                +5.2%   0.39    -41.5%     -6.0%
      BBCA buy & hold              +13.8%   0.63    -51.8%     +7.4%
    out-of-sample 2019-2026
      momentum only                 +0.2%   0.11    -30.1%     -5.0%
      reversal only                 +7.7%   0.54    -24.8%    +21.1%
      THIS composite                +9.7%   0.52    -46.7%     +6.2%
      JCI buy & hold                -0.6%   0.05    -41.5%    -16.8%
      BBCA buy & hold               +5.1%   0.33    -51.8%     -0.4%

Cost stress, full period:
      1x  CAGR +13.0% Sharpe 0.66  |  2x  +7.8% / 0.45  |  3x  +1.9% / 0.20
It clears the project's 2x-cost bar. (An earlier approximation suggested it did
not; that was the same grid artifact.)

WHY IT BEATS THE MOMENTUM IT REPLACED. Concentration, not stock selection, was
the problem. In risk-on periods, holding the top 3 by momentum (+9.4%/yr), the
top 8 (+9.6%) and simply owning every liquid name (+9.6%) are indistinguishable
— selection adds nothing there. Holding only 3 names added idiosyncratic risk
for no return. The gain here comes from being less concentrated in risk-on and
from having something to do in risk-off, not from picking better.

HONEST READING OF THE TRADE-OFF vs `reversal` ALONE:

  - `switch_top_n` is NOT a clean plateau. Full-period Sharpe by top_n reads
    0.60 (3), 0.43 (5), 0.66 (8), 0.71 (10), 0.61 (13). The dip at 5 says this
    surface carries real noise. 8 was chosen and written into config BEFORE this
    table was produced, and is deliberately left alone rather than moved to 10:
    picking the best cell of a bumpy surface is how backtests get flattered.
  - Out-of-sample, `reversal` alone is arguably the better strategy: Sharpe 0.54
    vs 0.52, and a -24.8% drawdown against -46.7%, for 2pp less CAGR. This
    composite's edge is strongest in-sample.
  - It is configured live anyway because the product needs a portfolio that is
    doing something in both regimes; `reversal` sits in cash 72% of the time.
    That is a product decision, not a claim that it is the better strategy.
    If drawdown matters more than activity, `reversal` is one config line away.

Numbers dated before 2026-07-25 differ: they were measured with a mark-to-market
bug that valued positions with a missing close at zero (see backtest/engine.py
build_frames). The reversal leg was unaffected — it was in cash on the only
affected day.

See idxquant/strategies/reversal.py for the survivorship-bias qualification,
which applies to the risk-off leg here too.
"""
from __future__ import annotations

import pandas as pd

from ..features import indicators as ind
from .momentum import CrossSectionalMomentum
from .reversal import RiskOffReversal


class RegimeSwitch:
    def __init__(self, momentum: CrossSectionalMomentum, reversal: RiskOffReversal,
                 regime_sma: int = 200):
        self.momentum = momentum
        self.reversal = reversal
        self.regime_sma = regime_sma
        self.name = f"switch__{momentum.name}__{reversal.name}"

    def signals(self, prices: dict[str, pd.DataFrame],
                index_close: pd.Series) -> pd.DataFrame:
        mom = self.momentum.signals(prices, index_close)
        rev = self.reversal.signals(prices, index_close)
        # Each leg already gates itself on the regime, so they cannot both be
        # active on the same day; the clip is belt-and-braces against a future
        # leg that forgets to.
        return mom.add(rev, fill_value=0).clip(0, 1).astype(int)

    def signal_context(self, ticker: str, prices: dict[str, pd.DataFrame],
                       index_close: pd.Series) -> dict:
        risk_on = bool(ind.regime_filter(index_close, self.regime_sma).iloc[-1])
        leg = self.momentum if risk_on else self.reversal
        ctx = leg.signal_context(ticker, prices, index_close)
        ctx["reasoning"] = (f"[{'momentum' if risk_on else 'reversal'} leg] "
                            + ctx["reasoning"])
        return ctx
