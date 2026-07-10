"""SQLite schema and access helpers."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    source TEXT DEFAULT 'yfinance',
    ingested_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (ticker, date)
);
CREATE TABLE IF NOT EXISTS data_quality (
    ticker TEXT, date TEXT, issue TEXT, detail TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT, ticker TEXT, strategy TEXT, action TEXT,
    confidence TEXT, reasoning TEXT, expected_holding_days REAL,
    invalidation TEXT, suggested_weight REAL, risk_json TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mode TEXT NOT NULL,
    run_id TEXT, ticker TEXT, side TEXT,
    entry_date TEXT, entry_price REAL, exit_date TEXT, exit_price REAL,
    shares INTEGER, costs REAL, pnl REAL, return_pct REAL,
    holding_days INTEGER, open_flag INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS equity_curve (
    mode TEXT, run_id TEXT, date TEXT,
    equity REAL, cash REAL, exposure REAL, drawdown REAL,
    PRIMARY KEY (mode, run_id, date)
);
CREATE TABLE IF NOT EXISTS paper_positions (
    ticker TEXT PRIMARY KEY,
    shares INTEGER, entry_date TEXT, entry_price REAL, entry_costs REAL
);
CREATE TABLE IF NOT EXISTS paper_meta (
    key TEXT PRIMARY KEY, value TEXT
);
CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id TEXT PRIMARY KEY,
    strategy TEXT, params_json TEXT, start TEXT, end TEXT,
    metrics_json TEXT, config_hash TEXT, code_version TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    return con


def upsert_prices(con: sqlite3.Connection, ticker: str, df: pd.DataFrame) -> int:
    """df: index=DatetimeIndex, columns Open/High/Low/Close/Volume."""
    rows = [
        (ticker, d.strftime("%Y-%m-%d"), r.Open, r.High, r.Low, r.Close, r.Volume)
        for d, r in df.iterrows()
    ]
    con.executemany(
        """INSERT INTO prices (ticker, date, open, high, low, close, volume)
           VALUES (?,?,?,?,?,?,?)
           ON CONFLICT(ticker, date) DO UPDATE SET
             open=excluded.open, high=excluded.high, low=excluded.low,
             close=excluded.close, volume=excluded.volume,
             ingested_at=datetime('now')""",
        rows,
    )
    con.commit()
    return len(rows)


def load_prices(con: sqlite3.Connection, ticker: str) -> pd.DataFrame:
    df = pd.read_sql(
        "SELECT date, open AS Open, high AS High, low AS Low, close AS Close, volume AS Volume "
        "FROM prices WHERE ticker=? ORDER BY date",
        con, params=(ticker,), parse_dates=["date"], index_col="date",
    )
    return df


def log_quality_issue(con: sqlite3.Connection, ticker: str, date: str, issue: str, detail: str) -> None:
    con.execute(
        "INSERT INTO data_quality (ticker, date, issue, detail) VALUES (?,?,?,?)",
        (ticker, date, issue, detail),
    )
    con.commit()
