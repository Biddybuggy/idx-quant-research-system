"""Baseline: SMA crossover with JCI regime filter.

Hypothesis: liquid IDX large caps trend on multi-week horizons, driven by
persistent foreign flows and slow information diffusion. The regime filter
(JCI above its long SMA) avoids fighting broad de-risking episodes, which on
IDX are strongly foreign-outflow driven.

Long when SMA(fast) > SMA(slow) AND JCI > SMA(regime). Flat otherwise.
"""
from __future__ import annotations

import pandas as pd

from ..features import indicators as ind


class SmaCrossover:
    def __init__(self, fast: int = 20, slow: int = 100, regime_sma: int = 200):
        assert fast < slow, "fast window must be shorter than slow"
        self.fast, self.slow, self.regime_sma = fast, slow, regime_sma
        self.name = f"sma_{fast}_{slow}_regime{regime_sma}"

    def signals(self, prices: dict[str, pd.DataFrame], index_close: pd.Series) -> pd.DataFrame:
        regime_ok = ind.regime_filter(index_close, self.regime_sma)
        cols = {}
        for ticker, df in prices.items():
            close = df["Close"]
            raw = (ind.sma(close, self.fast) > ind.sma(close, self.slow)).astype(int)
            regime = regime_ok.reindex(raw.index).ffill().fillna(False)
            cols[ticker] = raw.where(regime, 0)
        return pd.DataFrame(cols).fillna(0).astype(int)

    def signal_context(self, ticker: str, prices: dict[str, pd.DataFrame],
                       index_close: pd.Series) -> dict:
        """Per-ticker fields for the signal file, computed at the latest close."""
        close = prices[ticker]["Close"]
        fast_ma = ind.sma(close, self.fast).iloc[-1]
        slow_ma = ind.sma(close, self.slow).iloc[-1]
        spread = float(fast_ma / slow_ma - 1) if slow_ma else float("nan")
        regime_ok = bool(ind.regime_filter(index_close, self.regime_sma).iloc[-1])
        if not regime_ok or spread <= 0.01:
            confidence = "low"
        elif spread > 0.03:
            confidence = "high"
        else:
            confidence = "medium"
        return {
            "confidence": confidence,
            "reasoning": (
                f"SMA({self.fast}) is {spread:+.1%} vs SMA({self.slow}); market regime is "
                f"{'risk-on' if regime_ok else 'risk-off'}."
            ),
            "invalidation": (
                f"Exit if SMA({self.fast}) closes below SMA({self.slow}), or if JCI "
                f"closes below its {self.regime_sma}-day SMA (regime turns risk-off)."
            ),
            "extra_risk": {"trend_strength_pct": round(spread, 4)},
        }
