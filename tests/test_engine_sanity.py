"""Engine sanity check: an always-long signal on a single ticker must reproduce
buy-and-hold total return minus entry frictions and lot-rounding cash drag.

Run:  .venv/bin/python tests/test_engine_sanity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from idxquant.backtest.engine import run_backtest
from idxquant.config import load_config
from idxquant.data import db


def main():
    cfg = load_config()
    con = db.connect(cfg.db_path)
    ticker = cfg.benchmark_stock
    df = db.load_prices(con, ticker)

    # Full-weight single name, huge halt threshold so the risk overlay is inert.
    cfg.max_weight = 1.0
    cfg.max_drawdown_halt = 0.99
    cfg.max_adv_participation = 1.0

    signals = pd.DataFrame({ticker: 1}, index=df.index)
    result = run_backtest({ticker: df}, signals, cfg)

    # Expected entry: the first day the engine is ALLOWED to buy — signals fill
    # next open, and the 20d ADV liquidity yardstick needs history first.
    from idxquant.features.indicators import avg_daily_value
    adv = avg_daily_value(df).shift(1)
    allowed = df.index[(adv.notna()) & (adv >= cfg.min_adv_idr)]
    entry_date = allowed[allowed >= df.index[1]][0]
    entry_open = df.loc[entry_date, "Open"]
    fill = entry_open * (1 + cfg.costs.buy_friction)
    shares_exact = cfg.initial_cash / (fill * (1 + cfg.costs.buy_commission))
    expected_ret = df["Close"].iloc[-1] / fill - 1

    engine_ret = result.equity.iloc[-1] / cfg.initial_cash - 1
    n_trades = len(result.trades)
    still_open = int(result.trades["open_flag"].sum())

    # Lot rounding leaves a little cash idle -> engine return slightly below exact.
    invested_frac = (int(shares_exact // cfg.lot_size) * cfg.lot_size) / shares_exact
    expected_engine = expected_ret * invested_frac

    print(f"buy-and-hold arithmetic (net of entry friction): {expected_ret:+.2%}")
    print(f"engine result:                                   {engine_ret:+.2%}")
    print(f"expected after lot-rounding cash drag:           {expected_engine:+.2%}")
    print(f"trades: {n_trades} (open: {still_open})")

    assert n_trades == 1 and still_open == 1, "expected exactly one open trade"
    tol = 0.02 * max(abs(expected_engine), 1.0)  # 2% relative tolerance
    assert abs(engine_ret - expected_engine) < tol, \
        f"engine {engine_ret:.4f} vs expected {expected_engine:.4f}"
    print("PASS")


if __name__ == "__main__":
    main()
