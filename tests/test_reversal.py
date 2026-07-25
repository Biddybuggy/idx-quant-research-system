"""Guards for the risk-off reversal strategy and the panel that displays it.

Three things must hold, and each one is a mistake this project has already made
once:

  1. No lookahead. The whole backtest is worthless if the signal at day t can
     see day t+1. Asserted by truncating history and checking nothing changed.
  2. The strategy is regime-gated. It was only ever validated while JCI is below
     its trend (risk-on IC is +0.008 — nothing). If it can hold in a rising
     market, it is trading a hypothesis nobody tested.
  3. The panel cannot present itself as advice, cannot drop its caveats, and
     cannot quote a return that did not come through the costed engine. The
     retired tactical panel showed a score that had never been backtested; this
     asserts the replacement cannot repeat that.

Run:  .venv/bin/python tests/test_reversal.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from idxquant.config import load_config
from idxquant.features import indicators as ind
from idxquant.research import riskoff
from idxquant.strategies.reversal import RiskOffReversal

RNG = np.random.default_rng(11)


def _series(n, start, drift, index=None):
    steps = RNG.normal(drift, 0.014, n)
    idx = index if index is not None else pd.bdate_range("2019-01-01", periods=n)
    return pd.Series(start * np.exp(np.cumsum(steps)), index=idx)


def _ohlcv(close, vol=5e7):
    return pd.DataFrame({"Open": close.shift(1).fillna(close.iloc[0]),
                         "High": close * 1.01, "Low": close * 0.99,
                         "Close": close, "Volume": vol})


def _universe(n=600):
    """A falling index (so the regime is risk-off for most of the sample) plus
    eight names spread from strong to weak."""
    index_close = _series(n, 1000, -0.0006)
    drifts = [0.0014, 0.0010, 0.0006, 0.0002, -0.0002, -0.0006, -0.0011, -0.0016]
    prices = {f"T{i}.JK": _ohlcv(_series(n, 1000, d, index_close.index))
              for i, d in enumerate(drifts)}
    return prices, index_close


def main():
    prices, index_close = _universe()
    strat = RiskOffReversal(lookback=60, top_n=3, rebalance_days=21,
                            regime_sma=200, min_adv_idr=1e9)

    # --- 1. no lookahead: truncate the future, the past must not move --------
    full_rank = strat.rank_frame(prices, index_close)
    full_sig = strat.signals(prices, index_close)
    for cut in (400, 500, 599):
        part_prices = {t: df.iloc[:cut] for t, df in prices.items()}
        part_idx = index_close.iloc[:cut]
        r = strat.rank_frame(part_prices, part_idx)
        pd.testing.assert_frame_equal(r, full_rank.iloc[:cut], check_exact=False,
                                      rtol=1e-10, obj=f"rank_frame cut={cut}")
        # signals re-rank on an index-position grid, so truncation only has to
        # preserve the prefix that shares the same rebalance days
        s = strat.signals(part_prices, part_idx)
        assert (s.values == full_sig.iloc[:cut].values).all(), \
            f"signals changed after truncating at {cut}"
    print("1. no-lookahead: rank_frame and signals are prefix-stable  PASS")

    # --- 2. regime gate: never long while the market is risk-on -------------
    risk_on = ind.regime_filter(index_close, 200).reindex(full_sig.index) \
        .ffill().fillna(False).astype(bool)
    assert risk_on.any(), "test universe must contain some risk-on days to be meaningful"
    assert int(full_sig[risk_on].to_numpy().sum()) == 0, \
        "strategy held a position while the regime was risk-on — it was never tested there"
    assert int(full_sig[~risk_on].to_numpy().sum()) > 0, "strategy never held anything at all"
    print(f"2. regime gate: 0 long-days in {int(risk_on.sum())} risk-on days  PASS")

    # --- 3. it really is buying the laggards, not the leaders ---------------
    # The score must BE the negated relative return: that sign is the entire
    # difference between this strategy and the one that lost 20.7%/yr.
    closes = pd.DataFrame({t: df["Close"] for t, df in prices.items()}).sort_index()
    rel = closes.pct_change(60).sub(index_close.reindex(closes.index).ffill()
                                    .pct_change(60), axis=0)
    pd.testing.assert_frame_equal(full_rank, -rel, check_exact=False, rtol=1e-10,
                                  obj="rank_frame must be the NEGATED relative return")

    # And the holdings on a rebalance day must be that score's top names. Picks
    # are only refreshed every `rebalance_days`, so check a rebalance date, not
    # the last date — the holdings on any other day are deliberately stale.
    rebals = [d for i, d in enumerate(full_sig.index)
              if i % strat.rebalance_days == 0 and not risk_on.loc[d]]
    assert rebals, "test universe produced no risk-off rebalance days"
    d = rebals[-1]
    picked = sorted(t for t in full_sig.columns if full_sig.loc[d, t] == 1)
    weakest = sorted(rel.loc[d].nsmallest(strat.top_n).index)
    assert picked == weakest, \
        f"on {d.date()} picked {picked} but the weakest were {weakest} — sign inverted"
    print(f"3. sign: score is -relative-return; holds weakest {picked}  PASS")

    # --- 4. the panel's honesty requirements --------------------------------
    cfg = load_config()
    ctx = riskoff.screen(prices, index_close, cfg)
    if ctx is None:
        raise AssertionError("screen() returned None on a risk-off sample")

    # 4a. the live-episode number must be engine-measured, i.e. after costs.
    #     A gross number is a different, flattering number and must not appear.
    for lang, word in (("episode_id", "setelah biaya"), ("episode_en", "after costs")):
        assert word in ctx[lang], f"{lang} must state the figure is after costs: {ctx[lang]}"
    assert "sebelum biaya" not in ctx["episode_id"] and "before costs" not in ctx["episode_en"], \
        "the panel must never quote a gross return"

    # 4b. both caveats are mandatory and cannot be quietly dropped.
    for lang, must in (("caveat_id", ("21 saham", "komoditas", "bukan rekomendasi")),
                       ("caveat_en", ("21 stocks", "commodity", "not a buy recommendation"))):
        for m in must:
            assert m in ctx[lang], f"{lang} lost its caveat about {m!r}"

    # 4c. no advice wording anywhere in the copy the reader sees.
    banned = ["rekomendasi beli", "sinyal beli", "wajib beli", "pasti naik",
              "dijamin", "guaranteed", "sure thing", "you should buy"]
    copy = " ".join(str(ctx[k]) for k in
                    ("intro_id", "intro_en", "record_id", "record_en",
                     "episode_id", "episode_en")).lower()
    for w in banned:
        assert w not in copy, f"panel copy reads as advice ({w!r})"
    # the disclaimers are the one place "bukan rekomendasi beli" is required
    assert "bukan rekomendasi beli" in ctx["caveat_id"].lower()

    # 4d. the record shown must match the strategy module, the source of truth.
    src = (Path(__file__).resolve().parent.parent /
           "idxquant" / "strategies" / "reversal.py").read_text()
    assert f"{riskoff.RECORD['riskoff_ann']*100:+.1f}%" in src, \
        "riskoff.RECORD drifted from the evidence block in strategies/reversal.py"

    # 4e. `losing` must agree with the number actually printed.
    assert ctx["losing"] == (ctx["ep_strat"] < 0)
    print("4. panel: after-costs figure, caveats intact, no advice wording  PASS")

    # --- 5. risk-on renders nothing at all ----------------------------------
    rising = _series(600, 1000, 0.0012)
    rising_prices = {t: _ohlcv(_series(600, 1000, 0.0004, rising.index))
                     for t in prices}
    assert riskoff.screen(rising_prices, rising, cfg) is None, \
        "the panel must be absent in a risk-on market, not empty"
    print("5. risk-on: panel absent  PASS")

    print("PASS")


if __name__ == "__main__":
    main()
