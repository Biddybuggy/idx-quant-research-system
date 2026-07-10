"""Cross-sectional 12-1 momentum, monthly rebalance, JCI regime filter.

Hypothesis: relative strength persists on multi-month horizons in IDX large
caps (slow-moving foreign flows, index-inclusion effects, analyst herding).
The classic construction skips the most recent month to avoid short-term
reversal. Low turnover (~monthly) is deliberate: IDX retail costs kill
fast strategies.

Rules, all computed at the close of each month's last trading day:
  - momentum = total return from t-252 to t-21 trading days
  - hold the top `top_n` names whose momentum is positive
  - only while JCI is above its regime SMA (checked daily, exits mid-month
    if the regime breaks — crash protection outranks turnover)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import indicators as ind


class CrossSectionalMomentum:
    def __init__(self, lookback: int = 252, skip: int = 21, top_n: int = 3,
                 regime_sma: int = 200):
        assert skip < lookback
        self.lookback, self.skip, self.top_n = lookback, skip, top_n
        self.regime_sma = regime_sma
        self.name = f"xmom_{lookback}_{skip}_top{top_n}_regime{regime_sma}"

    def _momentum(self, closes: pd.DataFrame) -> pd.DataFrame:
        # return over [t-lookback, t-skip]; shift makes it known at close t
        return closes.pct_change(self.lookback - self.skip).shift(self.skip)

    def signals(self, prices: dict[str, pd.DataFrame], index_close: pd.Series) -> pd.DataFrame:
        closes = pd.DataFrame({t: df["Close"] for t, df in prices.items()}).sort_index()
        mom = self._momentum(closes)
        regime_ok = ind.regime_filter(index_close, self.regime_sma) \
            .reindex(closes.index).ffill().fillna(False)

        # last trading day of each month = rebalance decision point
        month_ends = set(closes.index.to_series().groupby(closes.index.to_period("M")).max())

        target = pd.DataFrame(0, index=closes.index, columns=closes.columns)
        held: list[str] = []
        for date in closes.index:
            if date in month_ends:
                row = mom.loc[date].dropna()
                row = row[row > 0]  # absolute-momentum gate
                held = list(row.nlargest(self.top_n).index)
            if held and regime_ok.loc[date]:
                target.loc[date, held] = 1
        return target.astype(int)

    def signal_context(self, ticker: str, prices: dict[str, pd.DataFrame],
                       index_close: pd.Series) -> dict:
        """Per-ticker fields for the signal file, computed at the latest close."""
        closes = pd.DataFrame({t: df["Close"] for t, df in prices.items()}).sort_index()
        mom = self._momentum(closes).iloc[-1]
        ranks = mom.rank(ascending=False)
        regime_ok = bool(ind.regime_filter(index_close, self.regime_sma).iloc[-1])
        m = float(mom.get(ticker, np.nan))
        rank = int(ranks.get(ticker)) if not np.isnan(m) else None
        if not regime_ok:
            confidence = "low"
        elif rank is not None and rank == 1 and m > 0.15:
            confidence = "high"
        elif rank is not None and rank <= self.top_n and m > 0:
            confidence = "medium"
        else:
            confidence = "low"
        return {
            "confidence": confidence,
            "reasoning": (
                f"12-month momentum (skipping the last month) is {m:+.1%}, "
                f"rank {rank} of {mom.notna().sum()} in the watchlist; market regime is "
                f"{'risk-on' if regime_ok else 'risk-off'}."
            ),
            "invalidation": (
                f"Exit at the next monthly rebalance if the stock drops out of the "
                f"top {self.top_n} by momentum or momentum turns negative; exit "
                f"immediately if JCI closes below its {self.regime_sma}-day SMA."
            ),
            "extra_risk": {"momentum_12_1": round(m, 4) if m == m else None,
                           "watchlist_rank": rank},
        }
