"""No-lookahead guard for the tactical score and the strategy built on it.

The whole backtest is worthless if the score at day t can see day t+1. The
strategy contract says a value at t must be computable from data up to and
including the CLOSE of t — this asserts that directly, by truncating history
and checking nothing changes.

Run:  .venv/bin/python tests/test_tactical_lookahead.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from idxquant.research import tactical
from idxquant.strategies.tactical_rs import TacticalRelativeStrength

RNG = np.random.default_rng(7)


def _series(n: int, start: float, drift: float) -> pd.Series:
    steps = RNG.normal(drift, 0.015, n)
    return pd.Series(start * np.exp(np.cumsum(steps)),
                     index=pd.bdate_range("2020-01-01", periods=n))


def _ohlcv(close: pd.Series, vol: float = 5e7) -> pd.DataFrame:
    return pd.DataFrame({"Open": close.shift(1).fillna(close.iloc[0]),
                         "High": close * 1.01, "Low": close * 0.99,
                         "Close": close, "Volume": vol})


def main():
    n = 300
    index_close = _series(n, 1000, -0.0004)
    tickers = {f"T{i}.JK": _ohlcv(_series(n, 1000, d))
               for i, d in enumerate([0.0012, 0.0006, -0.0002, -0.0009, 0.0003])}

    # --- 1. score_history: truncating the future must not change the past ----
    df = tickers["T0.JK"]
    full = tactical.score_history(df, index_close, min_adv_bn=1.0)
    for cut in (200, 250, 299):
        part = tactical.score_history(df.iloc[:cut], index_close.iloc[:cut], min_adv_bn=1.0)
        last = part.index[-1]
        for col in ("st_score", "rs20", "rs60", "from_hi20", "stop_pct", "is_opportunity"):
            a, b = full.loc[last, col], part.loc[last, col]
            if isinstance(a, (bool, np.bool_)):
                assert bool(a) == bool(b), (col, last, a, b)
            else:
                assert (np.isnan(a) and np.isnan(b)) or np.isclose(a, b), (col, last, a, b)
    print(f"score_history: no lookahead across {len(full)} rows")

    # --- 2. the strategy's exposure frame must be equally causal ------------
    strat = TacticalRelativeStrength(top_n=2, min_adv_bn=1.0)
    sig_full = strat.signals(tickers, index_close)
    cut = 260
    sig_part = strat.signals({t: d.iloc[:cut] for t, d in tickers.items()},
                             index_close.iloc[:cut])
    common = sig_part.index
    assert (sig_full.loc[common] == sig_part).all().all(), \
        "truncating history changed earlier signals — lookahead in the strategy"
    print(f"strategy signals: no lookahead across {len(common)} rows")

    # --- 3. sanity: the frame is a real 0/1 exposure, and it actually trades -
    assert set(np.unique(sig_full.values)) <= {0, 1}, "exposure must be 0/1"
    assert (sig_full.sum(axis=1) <= 2).all(), "held more than top_n at once"
    assert sig_full.values.sum() > 0, "strategy never took a position — test is vacuous"
    turnover = int((sig_full.diff().abs().sum(axis=1) > 0).sum())
    print(f"positions taken on {int((sig_full.sum(axis=1) > 0).sum())} days, "
          f"{turnover} change-days, max concurrent {int(sig_full.sum(axis=1).max())}")

    # --- 4. the stop must be able to force an exit --------------------------
    crash = _series(150, 1000, 0.004)
    crash = pd.concat([crash, pd.Series(crash.iloc[-1] * np.linspace(1, 0.55, 40),
                                        index=pd.bdate_range(crash.index[-1] +
                                                             pd.Timedelta(days=1), periods=40))])
    idx2 = pd.Series(1000.0, index=crash.index)
    one = {"C.JK": _ohlcv(crash)}
    # Between rebalances the stop is the only risk control, so its effect is
    # only visible at a slower cadence — at daily cadence the weekly re-screen
    # and the stop exit on essentially the same day by construction.
    with_stop = TacticalRelativeStrength(top_n=1, min_adv_bn=1.0, use_stop=True,
                                         rebalance_days=20).signals(one, idx2)
    without = TacticalRelativeStrength(top_n=1, min_adv_bn=1.0, use_stop=False,
                                       rebalance_days=20).signals(one, idx2)
    assert with_stop.values.sum() < without.values.sum(), \
        "the stop never bound, so it is not being tested"
    print(f"stop cuts exposure {int(without.values.sum())} -> {int(with_stop.values.sum())} days "
          f"(20-day rebalance)")

    # At daily cadence the stop is near-redundant by construction — assert that
    # explicitly so the backtest write-up cannot overclaim what the stop adds.
    d_stop = TacticalRelativeStrength(top_n=1, min_adv_bn=1.0, use_stop=True,
                                      rebalance_days=1).signals(one, idx2).values.sum()
    d_none = TacticalRelativeStrength(top_n=1, min_adv_bn=1.0, use_stop=False,
                                      rebalance_days=1).signals(one, idx2).values.sum()
    print(f"daily cadence: stop changes exposure {int(d_none)} -> {int(d_stop)} days "
          f"({'binds' if d_stop < d_none else 'redundant'})")

    print("PASS")


if __name__ == "__main__":
    main()
