"""Strategy contract.

A strategy maps price history to a *desired exposure frame*:
DataFrame indexed by date, one column per ticker, values in {0, 1}
(long-only; fractional exposures can come later).

The value at date t must be computable from data up to and including the
CLOSE of t. The engine — not the strategy — lags it one day for execution.
Keeping the lag in exactly one place is the no-lookahead guarantee.
"""
from __future__ import annotations

from typing import Protocol

import pandas as pd


class Strategy(Protocol):
    name: str

    def signals(
        self,
        prices: dict[str, pd.DataFrame],
        index_close: pd.Series,
    ) -> pd.DataFrame:
        """Return desired exposure frame (dates x tickers, values 0/1)."""
        ...
