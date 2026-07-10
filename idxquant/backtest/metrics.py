"""Performance and trade statistics. Risk-free rate assumed 0 for simplicity;
swap in an Indonesian short-rate series later if you want excess-return Sharpe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def cagr(equity: pd.Series) -> float:
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0 or equity.iloc[0] <= 0:
        return np.nan
    return (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1


def sharpe(daily_returns: pd.Series) -> float:
    sd = daily_returns.std()
    return np.nan if sd == 0 else daily_returns.mean() / sd * np.sqrt(TRADING_DAYS)


def sortino(daily_returns: pd.Series) -> float:
    downside = daily_returns[daily_returns < 0]
    dd = downside.std()
    return np.nan if (dd == 0 or np.isnan(dd)) else daily_returns.mean() / dd * np.sqrt(TRADING_DAYS)


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def turnover_per_year(trades: pd.DataFrame, equity: pd.Series) -> float:
    """Round-trip notional traded / average equity / years."""
    if trades.empty:
        return 0.0
    traded = (trades["shares"] * trades["entry_price"]).sum() + \
             (trades["shares"] * trades["exit_price"]).sum()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    return float(traded / equity.mean() / max(years, 1e-9))


def trade_stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n_trades": 0}
    closed = trades[trades["open_flag"] == 0]
    ref = closed if len(closed) else trades
    wins = ref[ref["pnl"] > 0]
    losses = ref[ref["pnl"] <= 0]
    gross_win = wins["pnl"].sum()
    gross_loss = -losses["pnl"].sum()
    return {
        "n_trades": int(len(ref)),
        "n_open": int(trades["open_flag"].sum()),
        "win_rate": float(len(wins) / len(ref)) if len(ref) else np.nan,
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else np.inf,
        "avg_holding_days": float(ref["holding_days"].mean()),
        "median_holding_days": float(ref["holding_days"].median()),
        "avg_return_pct": float(ref["return_pct"].mean()),
        "total_costs_idr": float(trades["costs"].sum()),
    }


def worst_trades(trades: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    if trades.empty:
        return trades
    cols = ["ticker", "entry_date", "exit_date", "return_pct", "pnl", "holding_days"]
    return trades.nsmallest(n, "return_pct")[cols]


def buy_and_hold_return(close: pd.Series, start, end) -> float:
    s = close.loc[start:end].dropna()
    return float(s.iloc[-1] / s.iloc[0] - 1) if len(s) > 1 else np.nan


def summarize(result, benchmarks: dict[str, pd.Series]) -> dict:
    """benchmarks: name -> close series (adjusted), for comparison rows."""
    eq, rets = result.equity, result.daily_returns
    start, end = eq.index[0], eq.index[-1]
    out = {
        "start": str(start.date()), "end": str(end.date()),
        "final_equity_idr": float(eq.iloc[-1]),
        "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1),
        "cagr": float(cagr(eq)),
        "sharpe": float(sharpe(rets)),
        "sortino": float(sortino(rets)),
        "max_drawdown": max_drawdown(eq),
        "avg_exposure": float(result.exposure.mean()),
        "turnover_per_year": turnover_per_year(result.trades, eq),
        **trade_stats(result.trades),
    }
    for name, close in benchmarks.items():
        s = close.loc[start:end].dropna()
        if len(s) > 1:
            bh_eq = s / s.iloc[0]
            out[f"bh_{name}_total_return"] = float(bh_eq.iloc[-1] - 1)
            out[f"bh_{name}_cagr"] = float(cagr(bh_eq))
            out[f"bh_{name}_sharpe"] = float(sharpe(bh_eq.pct_change().dropna()))
            out[f"bh_{name}_max_drawdown"] = max_drawdown(bh_eq)
    return out
