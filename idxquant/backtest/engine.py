"""Daily-bar portfolio simulator for IDX long-only strategies.

Execution model (no lookahead, in ONE place):
  - Strategy emits desired exposure at the CLOSE of day t.
  - The engine lags it: orders derived from day t's signal fill at the OPEN of t+1
    (the shift lives in `build_frames`).
  - Buy fills at open*(1 + half_spread + slippage) plus buy commission.
  - Sell fills at open*(1 - half_spread - slippage) minus sell commission
    (sell commission includes the 0.1% IDX sales tax via config).

IDX realism:
  - 100-share round lots.
  - Liquidity: entries skipped if 20d avg traded value < min_adv_idr; position
    value capped at max_adv_participation * ADV (both computed on lagged data).
  - Long-only, no leverage: buys capped by available cash.

Risk overlay:
  - If portfolio drawdown breaches max_drawdown_halt at a close, all positions
    are liquidated at the NEXT open and entries are blocked for
    halt_cooloff_days trading days; the equity peak then resets so the halt
    doesn't retrigger instantly.

The single-day transition lives in `step()`. The backtester loops `step()` over
history; the paper executor (idxquant/paper/executor.py) calls the SAME
`step()` on each new trading day. One code path = no backtest/paper drift.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from ..config import Config
from ..features import indicators as ind


@dataclass
class Position:
    shares: int
    entry_date: pd.Timestamp
    entry_price: float  # effective fill incl. friction
    entry_costs: float  # commissions paid at entry


@dataclass
class PortfolioState:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    peak: float = 0.0
    halted: bool = False
    cooloff: int = 0
    prev_equity: float = 0.0

    def __post_init__(self):
        if self.peak == 0.0:
            self.peak = self.cash
        if self.prev_equity == 0.0:
            self.prev_equity = self.cash


@dataclass
class BacktestResult:
    equity: pd.Series
    cash: pd.Series
    exposure: pd.Series
    trades: pd.DataFrame
    daily_returns: pd.Series = field(init=False)

    def __post_init__(self):
        self.daily_returns = self.equity.pct_change().fillna(0.0)


def _lots(value_idr: float, price: float, lot_size: int) -> int:
    lots = math.floor(value_idr / (price * lot_size))
    return max(lots, 0) * lot_size


def build_frames(prices: dict[str, pd.DataFrame], signals: pd.DataFrame):
    """Aligned frames for simulation. THE no-lookahead lag is applied here:
    today's orders come from yesterday's signal; liquidity uses lagged ADV."""
    tickers = list(signals.columns)
    dates = signals.index
    target = signals.shift(1).fillna(0).astype(int)
    adv = pd.DataFrame(
        {t: ind.avg_daily_value(prices[t]).shift(1).reindex(dates) for t in tickers}
    )
    opens = pd.DataFrame({t: prices[t]["Open"].reindex(dates) for t in tickers})
    closes = pd.DataFrame({t: prices[t]["Close"].reindex(dates) for t in tickers})
    return target, opens, closes, adv


def step(
    state: PortfolioState,
    date: pd.Timestamp,
    row_target: pd.Series,
    row_open: pd.Series,
    row_close: pd.Series,
    row_adv: pd.Series,
    cfg: Config,
) -> tuple[list[dict], float, float]:
    """One trading day: sells and buys at the open, mark-to-market at the close,
    then the drawdown-halt state machine. Mutates `state`.
    Returns (closed_trades, equity, exposure)."""
    c = cfg.costs
    closed: list[dict] = []

    # --- 1. SELLS at today's open (signal exits, or forced by halt) ---
    for t in list(state.positions):
        want_out = (row_target.get(t, 0) == 0) or state.halted
        px = row_open.get(t)
        if want_out and px is not None and not pd.isna(px):
            pos = state.positions.pop(t)
            fill = px * (1 - c.sell_friction)
            gross = pos.shares * fill
            commission = gross * c.sell_commission
            state.cash += gross - commission
            pnl = gross - commission - pos.shares * pos.entry_price - pos.entry_costs
            basis = pos.shares * pos.entry_price + pos.entry_costs
            closed.append({
                "ticker": t, "side": "long",
                "entry_date": pos.entry_date, "entry_price": pos.entry_price,
                "exit_date": date, "exit_price": fill,
                "shares": pos.shares, "costs": pos.entry_costs + commission,
                "pnl": pnl, "return_pct": pnl / basis,
                "holding_days": int((date - pos.entry_date).days),
                "open_flag": 0,
            })

    # --- 2. BUYS at today's open ---
    if not state.halted:
        n_active = int(row_target.sum())
        wanted = [t for t in row_target.index
                  if row_target[t] == 1 and t not in state.positions
                  and not pd.isna(row_open.get(t))]
        if wanted and n_active > 0:
            weight = min(cfg.max_weight, 1.0 / n_active)
            for t in wanted:
                adv_t = row_adv.get(t)
                if adv_t is None or pd.isna(adv_t) or adv_t < cfg.min_adv_idr:
                    continue  # liquidity filter
                budget = min(state.prev_equity * weight,
                             adv_t * cfg.max_adv_participation,
                             state.cash / (1 + c.buy_commission))
                fill = row_open[t] * (1 + c.buy_friction)
                shares = _lots(budget, fill, cfg.lot_size)
                if shares == 0:
                    continue
                notional = shares * fill
                commission = notional * c.buy_commission
                state.cash -= notional + commission
                state.positions[t] = Position(shares, date, fill, commission)

    # --- 3. Mark to market at today's close ---
    stock_value = sum(pos.shares * row_close[t]
                      for t, pos in state.positions.items()
                      if not pd.isna(row_close.get(t)))
    equity = state.cash + stock_value
    exposure = stock_value / equity if equity > 0 else 0.0
    state.prev_equity = equity

    # --- 4. Drawdown-halt state machine (evaluated at close, acts next open) ---
    state.peak = max(state.peak, equity)
    if not state.halted and equity < state.peak * (1 - cfg.max_drawdown_halt):
        state.halted = True
        state.cooloff = cfg.halt_cooloff_days
    elif state.halted and not state.positions:  # count down only once flat
        state.cooloff -= 1
        if state.cooloff <= 0:
            state.halted = False
            state.peak = equity  # reset so halt doesn't retrigger immediately

    return closed, equity, exposure


def mark_open_trades(state: PortfolioState, row_close: pd.Series,
                     date: pd.Timestamp) -> list[dict]:
    """Snapshot still-open positions as open trades at the given close."""
    out = []
    for t, pos in state.positions.items():
        px = row_close.get(t)
        if px is None or pd.isna(px):
            continue
        pnl = pos.shares * px - pos.shares * pos.entry_price - pos.entry_costs
        basis = pos.shares * pos.entry_price + pos.entry_costs
        out.append({
            "ticker": t, "side": "long",
            "entry_date": pos.entry_date, "entry_price": pos.entry_price,
            "exit_date": date, "exit_price": float(px),
            "shares": pos.shares, "costs": pos.entry_costs,
            "pnl": pnl, "return_pct": pnl / basis,
            "holding_days": int((date - pos.entry_date).days),
            "open_flag": 1,
        })
    return out


def run_backtest(
    prices: dict[str, pd.DataFrame],
    signals: pd.DataFrame,
    cfg: Config,
) -> BacktestResult:
    dates = signals.index
    target, opens, closes, adv = build_frames(prices, signals)

    state = PortfolioState(cash=cfg.initial_cash)
    trades: list[dict] = []
    equity_hist, cash_hist, expo_hist = [], [], []

    for date in dates:
        closed, equity, exposure = step(
            state, date, target.loc[date], opens.loc[date],
            closes.loc[date], adv.loc[date], cfg,
        )
        trades.extend(closed)
        equity_hist.append(equity)
        cash_hist.append(state.cash)
        expo_hist.append(exposure)

    trades.extend(mark_open_trades(state, closes.loc[dates[-1]], dates[-1]))

    return BacktestResult(
        equity=pd.Series(equity_hist, index=dates, name="equity"),
        cash=pd.Series(cash_hist, index=dates, name="cash"),
        exposure=pd.Series(expo_hist, index=dates, name="exposure"),
        trades=pd.DataFrame(trades),
    )
