"""What should the portfolio do while the market is RISK-ON?

The reversal leg settled the risk-off half. The risk-on half is currently 12-1
momentum, which measured at +3.3%/yr with a -50% drawdown -- below the index.
This asks what else could occupy that slot.

The candidate worth taking seriously is the boring one: stop picking. An
equal-weight basket of the whole liquid universe returned +12.5%/yr with the
best Sharpe of anything measured so far. If stock selection cannot beat "own
them all", then selection is destroying value and the honest thing is to say so.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from idxquant.backtest.engine import run_backtest
from idxquant.backtest import metrics as mt
from idxquant.config import load_config
from idxquant.data import db
from idxquant.features import indicators as ind

cfg = load_config()
con = db.connect(cfg.db_path)
prices = {t: db.load_prices(con, t) for t in cfg.tickers}
index_close = db.load_prices(con, cfg.index_ticker)["Close"]
con.close()

closes = pd.DataFrame({t: d["Close"] for t, d in prices.items()}).sort_index()
idx = index_close.reindex(closes.index).ffill()
adv = pd.DataFrame({t: ind.avg_daily_value(prices[t]) for t in cfg.tickers}).reindex(closes.index)
liquid = adv >= cfg.min_adv_idr
regime_on = ind.regime_filter(idx, cfg.regime_sma).reindex(closes.index).ffill().fillna(False).astype(bool)
risk_off = ~regime_on
rsi14 = pd.DataFrame({t: ind.rsi(closes[t]) for t in cfg.tickers})
vol60 = pd.DataFrame({t: ind.realized_vol(closes[t], 60) for t in cfg.tickers})


def rs(w):
    return closes.pct_change(w).sub(idx.pct_change(w), axis=0)


def rank_signals(sig, top_n, rb, allow):
    target = pd.DataFrame(0, index=closes.index, columns=closes.columns)
    s = sig.where(liquid)
    held: list[str] = []
    for i, d in enumerate(closes.index):
        if i % rb == 0:
            row = s.loc[d].dropna()
            held = list(row.nlargest(top_n).index) if len(row) >= top_n else []
        if allow.loc[d] and held:
            target.loc[d, held] = 1
    return target.astype(int)


def hold_all(allow):
    """Own every liquid name -- no selection at all."""
    t = liquid.astype(int).copy()
    t[~allow] = 0
    return t.astype(int)


def riskoff_ann(rets):
    sub = rets[risk_off.reindex(rets.index).fillna(False)]
    return (1 + sub).prod() ** (252 / len(sub)) - 1 if len(sub) >= 20 else np.nan


def riskon_ann(rets):
    sub = rets[regime_on.reindex(rets.index).fillna(False)]
    return (1 + sub).prod() ** (252 / len(sub)) - 1 if len(sub) >= 20 else np.nan


rev_t = rank_signals(-rs(60), 5, 21, risk_off)
mom = closes.pct_change(231).shift(21)

RISKON_LEGS = {
    "momentum 12-1 top3 (current)": rank_signals(mom, 3, 21, regime_on),
    "momentum 12-1 top8":           rank_signals(mom, 8, 21, regime_on),
    "own everything liquid":        hold_all(regime_on),
    "low vol top8":                 rank_signals(-vol60, 8, 21, regime_on),
    "low vol top5":                 rank_signals(-vol60, 5, 21, regime_on),
    "reversal -rs60 top5":          rank_signals(-rs(60), 5, 21, regime_on),
    "not-overbought top8 (-rsi)":   rank_signals(-rsi14, 8, 21, regime_on),
}

PERIODS = {"full 2010-2026": ("2010-01-01", "2026-12-31"),
           "OOS  2019-2026": ("2019-01-01", "2026-12-31")}


def run(t, a, b):
    px = {k: v.loc[a:b] for k, v in prices.items()}
    return run_backtest(px, t.loc[a:b], cfg)


def line(label, r):
    print(f"  {label:34s} CAGR {mt.cagr(r.equity)*100:+6.1f}%  Sharpe {mt.sharpe(r.daily_returns):5.2f}  "
          f"maxDD {mt.max_drawdown(r.equity)*100:6.1f}%  risk-on {riskon_ann(r.daily_returns)*100:+6.1f}%  "
          f"expo {r.exposure.mean()*100:3.0f}%  turn {mt.turnover_per_year(r.trades, r.equity):4.1f}x")


print("=" * 118)
print("1. RISK-ON LEG ALONE (flat while risk-off, so 'risk-on' column is what matters)")
print("=" * 118)
for pname, (a, b) in PERIODS.items():
    print(f"\n--- {pname} ---")
    for label, t in RISKON_LEGS.items():
        line(label, run(t, a, b))

print()
print("=" * 118)
print("2. FULL COMPOSITE: each risk-on leg + the SAME reversal leg when risk-off")
print("=" * 118)
for pname, (a, b) in PERIODS.items():
    print(f"\n--- {pname} ---")
    rows = []
    for label, t in RISKON_LEGS.items():
        combo = t.add(rev_t, fill_value=0).clip(0, 1).astype(int)
        r = run(combo, a, b)
        rows.append((label, r))
    for label, r in sorted(rows, key=lambda x: -mt.sharpe(x[1].daily_returns)):
        print(f"  {label:34s} CAGR {mt.cagr(r.equity)*100:+6.1f}%  Sharpe {mt.sharpe(r.daily_returns):5.2f}  "
              f"maxDD {mt.max_drawdown(r.equity)*100:6.1f}%  "
              f"risk-off {riskoff_ann(r.daily_returns)*100:+6.1f}%  "
              f"expo {r.exposure.mean()*100:3.0f}%  turn {mt.turnover_per_year(r.trades, r.equity):4.1f}x")
    for bl, s in (("JCI buy & hold", idx), ("BBCA buy & hold", closes["BBCA.JK"])):
        ss = s.loc[a:b].dropna()
        rr = ss.pct_change().fillna(0.0)
        print(f"  {bl:34s} CAGR {mt.cagr(ss)*100:+6.1f}%  Sharpe {mt.sharpe(rr):5.2f}  "
              f"maxDD {mt.max_drawdown(ss)*100:6.1f}%  risk-off {riskoff_ann(rr)*100:+6.1f}%")
