"""A gap in the price data must not be able to invent a loss.

Regression test for a real bug. On 2019-06-19 ten of the 21 watchlist names had
no close in the vendor data. `step()` marked to market by summing only the
positions whose close was present, which values the rest at zero. Equity fell
39% for one day, the 20% drawdown halt fired, the book was liquidated at the
next open and entries were blocked for 20 trading days. Equity was back to
normal the next day: the loss never happened, the forced liquidation did.

One bad day in 4,061 changed the measured record of every strategy that held
those names. The same gap during live paper trading would liquidate the real
paper book, so this is pinned in both directions: a short gap must be bridged,
and a long one must still not be readable as a wipeout.

Run:  .venv/bin/python tests/test_missing_price_mark.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from idxquant.backtest.engine import (MAX_STALE_MARK_DAYS, PortfolioState,
                                      Position, build_frames, run_backtest, step)
from idxquant.config import load_config


def _prices(n=120, gap_at=None, gap_len=1, tickers=("A.JK", "B.JK", "C.JK", "D.JK")):
    dates = pd.bdate_range("2020-01-01", periods=n)
    out = {}
    for i, t in enumerate(tickers):
        close = pd.Series(np.linspace(1000, 1200 + i * 10, n), index=dates)
        df = pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99,
                           "Close": close, "Volume": 5e8})
        if gap_at is not None and i < len(tickers) - 1:   # all but the last name
            df.iloc[gap_at:gap_at + gap_len, df.columns.get_loc("Close")] = np.nan
        out[t] = df
    return out, dates


def main():
    cfg = load_config()
    cfg.min_adv_idr = 0.0            # keep the liquidity filter out of this test
    cfg.max_weight = 0.25

    GAP = 80
    tickers = ["A.JK", "B.JK", "C.JK", "D.JK"]

    # --- 1. a one-day gap must not move equity ------------------------------
    clean, dates = _prices()
    holey, _ = _prices(gap_at=GAP)
    signals = pd.DataFrame(1, index=dates, columns=tickers)

    eq_clean = run_backtest(clean, signals, cfg).equity
    eq_holey = run_backtest(holey, signals, cfg).equity

    drop_clean = float(eq_clean.pct_change().iloc[GAP])
    drop_holey = float(eq_holey.pct_change().iloc[GAP])
    err = abs(drop_holey - drop_clean)
    print(f"1. gap day: clean {drop_clean:+.4%}  with-gap {drop_holey:+.4%}  "
          f"stale-by {err:.4%}")
    # A carried mark is YESTERDAY's price, so equity on the gap day is one day
    # stale — understated here by roughly one day of drift on the three gapped
    # names. That is the honest answer: we do not know that day's price. What
    # must not happen is a move on the order of the position's whole value.
    one_day = abs(drop_clean) * len(tickers)
    assert err <= one_day + 1e-9, (
        f"missing close moved equity {err:.2%}, more than one day of drift "
        f"({one_day:.2%}) — the mark is being fabricated, not carried")

    # ...and the staleness must be temporary: once prices print again the curve
    # has to rejoin the clean one exactly, with no permanent step.
    after = eq_holey.index[GAP + 1:]
    pd.testing.assert_series_equal(
        eq_clean.loc[after], eq_holey.loc[after], check_exact=False, rtol=1e-9,
        obj="equity curve after the gap closes")
    print("   curve rejoins exactly once prices resume  PASS")

    # --- 2. the halt must not fire off a data gap ---------------------------
    res = run_backtest(holey, signals, cfg)
    dd = float((res.equity / res.equity.cummax() - 1).min())
    assert dd > -cfg.max_drawdown_halt, \
        f"data gap produced a {dd:.1%} drawdown, enough to trip the {cfg.max_drawdown_halt:.0%} halt"
    print(f"2. worst drawdown on a rising series with a gap: {dd:+.2%}  PASS")

    # --- 3. a gap longer than the ffill limit still cannot read as a wipeout -
    long_gap, _ = _prices(gap_at=GAP, gap_len=MAX_STALE_MARK_DAYS + 6)
    res_long = run_backtest(long_gap, signals, cfg)
    dd_long = float((res_long.equity / res_long.equity.cummax() - 1).min())
    print(f"3. gap of {MAX_STALE_MARK_DAYS + 6} days (> ffill limit "
          f"{MAX_STALE_MARK_DAYS}): worst drawdown {dd_long:+.2%}")
    assert dd_long > -0.25, \
        f"an unbridgeable gap still read as a {dd_long:.0%} loss — entry-price fallback missing"

    # --- 4. step() directly: a NaN close marks at entry price, not zero ------
    target, opens, closes, adv = build_frames(clean, signals)
    st = PortfolioState(cash=cfg.initial_cash)
    st.positions["A.JK"] = Position(shares=1000, entry_date=dates[0],
                                    entry_price=1000.0, entry_costs=0.0)
    st.cash = 0.0
    row = pd.Series({t: np.nan for t in tickers})
    _, equity, _ = step(st, dates[50], pd.Series({t: 1 for t in tickers}),
                        pd.Series({t: np.nan for t in tickers}), row,
                        adv.loc[dates[50]], cfg)
    assert abs(equity - 1000 * 1000.0) < 1e-6, \
        f"position with no close marked at {equity:,.0f}, expected entry value 1,000,000"
    print(f"4. step() marks an unpriced position at entry value: {equity:,.0f}  PASS")

    # --- 5. open-trade snapshots use the same fallback ----------------------
    from idxquant.backtest.engine import mark_open_trades
    rows = mark_open_trades(st, row, dates[50])
    assert len(rows) == 1, "an unpriced open position vanished from the trade snapshot"
    assert abs(rows[0]["return_pct"]) < 1e-9
    print("5. mark_open_trades keeps unpriced positions at flat P&L  PASS")

    print("PASS")


if __name__ == "__main__":
    main()
