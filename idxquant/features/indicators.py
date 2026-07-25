"""Indicator library. All functions take Series/DataFrames and return Series
aligned to the input index. Nothing here may peek forward: rolling windows only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window).mean()


def momentum(close: pd.Series, days: int) -> pd.Series:
    """Total return over `days` trading days (63≈3m, 126≈6m, 252≈12m)."""
    return close.pct_change(days)


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def bollinger_z(close: pd.Series, window: int = 20) -> pd.Series:
    mean = close.rolling(window).mean()
    std = close.rolling(window).std()
    return (close - mean) / std


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range; df needs High/Low/Close."""
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [df["High"] - df["Low"],
         (df["High"] - prev_close).abs(),
         (df["Low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False).mean()


def realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    """Annualized close-to-close volatility."""
    return close.pct_change().rolling(window).std() * np.sqrt(252)


def regime_filter(index_close: pd.Series, window: int, band: float = 0.02) -> pd.Series:
    """Risk-on/off state with hysteresis to prevent whipsaw at the SMA line.

    Turns ON when index closes above its SMA; turns OFF only when it closes
    below SMA*(1-band); holds the previous state in between.
    """
    sma_v = index_close.rolling(window).mean()
    raw = np.where(index_close > sma_v, 1.0,
                   np.where(index_close < sma_v * (1 - band), 0.0, np.nan))
    return pd.Series(raw, index=index_close.index).ffill().fillna(0).astype(bool)


def relative_strength(close: pd.Series, index_close: pd.Series, window: int) -> pd.Series:
    """Excess return of a stock over the index across `window` trading days.

    Positive means the stock is beating the market — the signal that survives
    even a weak overall regime (a leader on the way down is still a leader).
    The index is reindexed onto the stock's calendar so the two align.
    """
    idx = index_close.reindex(close.index).ffill()
    return close.pct_change(window) - idx.pct_change(window)


def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


def avg_daily_value(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """20-day average traded value (IDR) — the liquidity yardstick."""
    return (df["Close"] * df["Volume"]).rolling(window).mean()
