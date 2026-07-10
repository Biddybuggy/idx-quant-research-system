"""Paper-trading executor.

Runs the SAME `step()` as the backtester, one new trading day at a time,
persisting portfolio state in SQLite between runs. This is deliberate: the
Phase 5 reconciliation requirement ("paper equals what the engine would say")
is satisfied by construction because there is only one fill code path.

Behavior:
  - First run initializes a flat portfolio at the latest close. With
    `backfill_days=N` it instead replays the last N trading days first, so the
    dashboard starts with a curve; the replayed span is recorded in paper_meta
    ('backfill_until') and must be labeled as simulated wherever displayed.
  - Each subsequent run processes every not-yet-processed trading day in order
    (catches up automatically if the scheduler missed days).
  - Trades land in `trades` (mode='paper'), the curve in `equity_curve`
    (mode='paper', run_id='paper-live').
"""
from __future__ import annotations

import sqlite3

import pandas as pd

from ..backtest.engine import PortfolioState, Position, build_frames, step
from ..config import Config
from ..data import db
from ..strategies.factory import make_strategy

RUN_ID = "paper-live"


def _meta_get(con: sqlite3.Connection, key: str) -> str | None:
    row = con.execute("SELECT value FROM paper_meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else None


def _meta_set(con: sqlite3.Connection, key: str, value) -> None:
    con.execute("INSERT INTO paper_meta (key, value) VALUES (?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, str(value)))


def _load_state(con: sqlite3.Connection) -> PortfolioState:
    state = PortfolioState(
        cash=float(_meta_get(con, "cash")),
        peak=float(_meta_get(con, "peak")),
        halted=_meta_get(con, "halted") == "1",
        cooloff=int(_meta_get(con, "cooloff")),
        prev_equity=float(_meta_get(con, "prev_equity")),
    )
    for t, sh, ed, ep, ec in con.execute(
            "SELECT ticker, shares, entry_date, entry_price, entry_costs FROM paper_positions"):
        state.positions[t] = Position(sh, pd.Timestamp(ed), ep, ec)
    return state


def _save_state(con: sqlite3.Connection, state: PortfolioState) -> None:
    _meta_set(con, "cash", state.cash)
    _meta_set(con, "peak", state.peak)
    _meta_set(con, "halted", "1" if state.halted else "0")
    _meta_set(con, "cooloff", state.cooloff)
    _meta_set(con, "prev_equity", state.prev_equity)
    con.execute("DELETE FROM paper_positions")
    con.executemany(
        "INSERT INTO paper_positions (ticker, shares, entry_date, entry_price, entry_costs) "
        "VALUES (?,?,?,?,?)",
        [(t, p.shares, p.entry_date.strftime("%Y-%m-%d"), p.entry_price, p.entry_costs)
         for t, p in state.positions.items()])


def _record_trade(con: sqlite3.Connection, tr: dict) -> None:
    con.execute(
        "INSERT INTO trades (mode, run_id, ticker, side, entry_date, entry_price, "
        "exit_date, exit_price, shares, costs, pnl, return_pct, holding_days, open_flag) "
        "VALUES ('paper',?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (RUN_ID, tr["ticker"], tr["side"],
         tr["entry_date"].strftime("%Y-%m-%d"), tr["entry_price"],
         tr["exit_date"].strftime("%Y-%m-%d"), tr["exit_price"],
         tr["shares"], tr["costs"], tr["pnl"], tr["return_pct"],
         tr["holding_days"], tr["open_flag"]))


def process(cfg: Config, backfill_days: int = 0) -> dict:
    """Process all new trading days. Returns a summary of what happened."""
    con = db.connect(cfg.db_path)
    prices = {t: db.load_prices(con, t) for t in cfg.tickers}
    index_close = db.load_prices(con, cfg.index_ticker)["Close"]
    strategy = make_strategy(cfg)
    signals = strategy.signals(prices, index_close)
    target, opens, closes, adv = build_frames(prices, signals)
    dates = signals.index

    last = _meta_get(con, "last_processed")
    if last is None:  # ---- first run: initialize ----
        anchor = max(len(dates) - 1 - backfill_days, 1)
        init_date = dates[anchor - 1]
        state = PortfolioState(cash=cfg.initial_cash)
        _meta_set(con, "started_at", init_date.strftime("%Y-%m-%d"))
        _meta_set(con, "strategy", strategy.name)
        if backfill_days:
            _meta_set(con, "backfill_until", dates[-1].strftime("%Y-%m-%d"))
        con.execute(
            "INSERT OR REPLACE INTO equity_curve (mode, run_id, date, equity, cash, exposure, drawdown) "
            "VALUES ('paper',?,?,?,?,0,0)",
            (RUN_ID, init_date.strftime("%Y-%m-%d"), cfg.initial_cash, cfg.initial_cash))
        last = init_date.strftime("%Y-%m-%d")
    else:
        state = _load_state(con)

    todo = dates[dates > pd.Timestamp(last)]
    n_trades = 0
    for date in todo:
        closed, equity, exposure = step(
            state, date, target.loc[date], opens.loc[date],
            closes.loc[date], adv.loc[date], cfg,
        )
        for tr in closed:
            _record_trade(con, tr)
        n_trades += len(closed)
        dd = equity / state.peak - 1 if state.peak > 0 else 0.0
        con.execute(
            "INSERT OR REPLACE INTO equity_curve (mode, run_id, date, equity, cash, exposure, drawdown) "
            "VALUES ('paper',?,?,?,?,?,?)",
            (RUN_ID, date.strftime("%Y-%m-%d"), equity, state.cash, exposure, dd))

    if len(todo):
        _meta_set(con, "last_processed", todo[-1].strftime("%Y-%m-%d"))
    _save_state(con, state)
    con.commit()

    positions = [
        {"ticker": t, "shares": p.shares,
         "entry_date": p.entry_date.strftime("%Y-%m-%d"),
         "entry_price": round(p.entry_price, 2),
         "last_close": float(closes.iloc[-1].get(t)),
         "unrealized_pnl": round(p.shares * (closes.iloc[-1].get(t) - p.entry_price)
                                 - p.entry_costs, 2)}
        for t, p in state.positions.items()
    ]
    summary = {
        "processed_days": [d.strftime("%Y-%m-%d") for d in todo],
        "closed_trades": n_trades,
        "equity": state.prev_equity,
        "cash": state.cash,
        "halted": state.halted,
        "positions": positions,
    }
    con.close()
    return summary
