"""Download daily OHLCV from Yahoo Finance and validate before storing.

Prices are stored *adjusted* (auto_adjust=True): splits and dividends folded in,
so backtest returns are total returns. Caveat (documented in ARCHITECTURE.md):
paper/live order sizing must use raw traded prices — a separate code path later.
"""
from __future__ import annotations

import sqlite3
import time

import pandas as pd
import yfinance as yf

from ..config import Config
from . import db


def download_ticker(ticker: str, start: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):  # yfinance >=0.2 returns MultiIndex
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def validate(ticker: str, df: pd.DataFrame, cfg: Config, con: sqlite3.Connection) -> None:
    """Flag anomalies; never silently 'fix' data."""
    is_index = ticker.startswith("^")
    ret = df["Close"].pct_change()
    for d in ret[ret.abs() > cfg.max_daily_move].index:
        db.log_quality_issue(con, ticker, d.strftime("%Y-%m-%d"), "extreme_move",
                             f"daily return {ret[d]:.1%} beyond auto-rejection band")
    bad_price = df[(df["Close"] <= 0) | ((df["Close"] < cfg.min_price) & ~is_index)]
    for d in bad_price.index:
        db.log_quality_issue(con, ticker, d.strftime("%Y-%m-%d"), "bad_price",
                             f"close={df.loc[d, 'Close']}")
    if not is_index:
        zero_vol = df[df["Volume"] == 0]
        if len(zero_vol) > 0:
            db.log_quality_issue(con, ticker, zero_vol.index[-1].strftime("%Y-%m-%d"),
                                 "zero_volume", f"{len(zero_vol)} zero-volume days in history")
    stale = (pd.Timestamp.now() - df.index[-1]).days
    if stale > 7:
        db.log_quality_issue(con, ticker, df.index[-1].strftime("%Y-%m-%d"),
                             "stale_series", f"last bar is {stale} days old")


def run_ingest(cfg: Config) -> dict[str, int]:
    """Download all tickers. A transient failure on one stock is logged and
    skipped (yesterday's stored data remains usable); only a failure of the
    index series or of most of the universe aborts the run."""
    con = db.connect(cfg.db_path)
    counts: dict[str, int] = {}
    failed: list[str] = []
    for ticker in cfg.tickers + [cfg.index_ticker]:
        df = None
        for attempt in (1, 2):
            try:
                df = download_ticker(ticker, cfg.start)
                break
            except Exception as err:
                if attempt == 1:
                    time.sleep(5)
                else:
                    failed.append(ticker)
                    db.log_quality_issue(con, ticker, pd.Timestamp.now().strftime("%Y-%m-%d"),
                                         "ingest_failed", str(err))
        if df is None:
            continue
        validate(ticker, df, cfg, con)
        counts[ticker] = db.upsert_prices(con, ticker, df)
    con.close()
    if cfg.index_ticker in failed:
        raise RuntimeError(f"Index series {cfg.index_ticker} failed to download")
    if len(failed) > len(cfg.tickers) // 2:
        raise RuntimeError(f"Majority of universe failed to download: {failed}")
    if failed:
        print(f"[ingest] WARNING: skipped (stale data in use): {failed}")
    return counts
