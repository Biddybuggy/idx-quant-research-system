"""Always-invested composite: 12-1 momentum when JCI is risk-on, 60-day
reversal when it is risk-off.

This is the strategy shape the product asks for — something to say every day,
not only in bull markets. The two legs are deliberately opposite: momentum buys
what has been strong, reversal buys what has been weak, and the regime decides
which question is the right one. Neither leg is new; both are composed from the
existing modules so there is one implementation of each.

EVIDENCE (real engine, real costs, 2010-01 to 2026-07):

                                    CAGR   Sharpe   maxDD   risk-off ann
    full 2010-2026
      momentum only (live today)    +3.3%   0.27    -50.0%     -8.0%
      reversal only (risk-off)      +7.7%   0.54    -30.1%    +17.2%
      THIS composite                +9.1%   0.48    -62.2%     +0.9%
      JCI buy & hold                +5.2%   0.39    -41.5%     -6.0%
      BBCA buy & hold              +13.8%   0.63    -51.8%     +7.4%
    out-of-sample 2019-2026
      momentum only                 +3.2%   0.25    -39.0%     -5.0%
      reversal only                 +7.6%   0.53    -24.8%    +19.6%
      THIS composite               +11.6%   0.53    -41.4%    +16.2%

READ THE DRAWDOWN COLUMN BEFORE THE CAGR COLUMN. The composite earns the
highest CAGR of anything tested and also the deepest drawdown of anything
tested — deeper than either leg alone and deeper than simply holding the index.
Two reasons, both real:

  1. The momentum leg is weak on this universe. On its own it returns +3.3%/yr
     with a 50% drawdown, worse than the index. That is not a costs artifact and
     not the drawdown halt: disabling the halt and varying top_n (3/5/8) leaves
     it in the same place. It needs its own review.
  2. The legs share one equity curve and one 20% drawdown halt. When the regime
     flips to risk-off the momentum leg liquidates into the fall, and those
     losses land on exactly the risk-off days the reversal leg is measured over.
     That is why the composite's full-period risk-off number (+0.9%) is so much
     worse than the reversal leg alone (+17.2%) despite holding the same names.

So this is registered as an option, not made the default. `reversal` alone has
the better risk-adjusted record and far more robustness evidence behind it; this
one wins on raw return by leaning on the leg with the weaker case. Switching the
live paper portfolio is a decision for the operator, made in config, once.

See idxquant/strategies/reversal.py for the survivorship-bias qualification that
applies to the risk-off leg here too.
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
