"""Measure the strategies AS THE FACTORY BUILDS THEM, not as research approximates them.

bt_composite.py approximated the momentum leg with a fixed 21-trading-day
rebalance grid. The real CrossSectionalMomentum rebalances on calendar month-ends
and applies an absolute-momentum gate (only names with mom > 0). Those are not
the same strategy, and the difference showed up as -38.1% vs -44.7% max drawdown.

Research may approximate. The numbers quoted in docstrings, docs and on the
dashboard may not. This script is the authority for those: it runs exactly what
make_strategy() returns for a given config name.
"""
from __future__ import annotations

import copy
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
from idxquant.strategies.factory import make_strategy

cfg = load_config()
con = db.connect(cfg.db_path)
prices = {t: db.load_prices(con, t) for t in cfg.tickers}
index_close = db.load_prices(con, cfg.index_ticker)["Close"]
con.close()

closes = pd.DataFrame({t: d["Close"] for t, d in prices.items()}).sort_index()
idx = index_close.reindex(closes.index).ffill()
regime_on = ind.regime_filter(idx, cfg.regime_sma).reindex(closes.index).ffill().fillna(False).astype(bool)
risk_off = ~regime_on

PERIODS = {"full 2010-2026": "2010-01-01", "OOS  2019-2026": "2019-01-01"}
NAMES = ["regime_switch", "reversal", "momentum"]


def riskoff_ann(rets):
    sub = rets[risk_off.reindex(rets.index).fillna(False)]
    return (1 + sub).prod() ** (252 / len(sub)) - 1 if len(sub) >= 20 else np.nan


def run(sig, a, c=cfg):
    px = {k: v.loc[a:] for k, v in prices.items()}
    return run_backtest(px, sig.loc[a:], c)


sigs = {}
for n in NAMES:
    s = make_strategy(cfg, name=n)
    sigs[n] = (s.name, s.signals(prices, index_close))

for pname, a in PERIODS.items():
    print(f"\n=== {pname} ===")
    print(f"  {'strategy':30s} {'CAGR':>7} {'Sharpe':>7} {'maxDD':>8} {'risk-off':>9} {'expo':>5} {'turn':>6}")
    for n in NAMES:
        label, sig = sigs[n]
        r = run(sig, a)
        print(f"  {n:30s} {mt.cagr(r.equity)*100:+6.1f}% {mt.sharpe(r.daily_returns):7.2f} "
              f"{mt.max_drawdown(r.equity)*100:7.1f}% {riskoff_ann(r.daily_returns)*100:+8.1f}% "
              f"{r.exposure.mean()*100:4.0f}% {mt.turnover_per_year(r.trades, r.equity):5.1f}x")
    for bl, ser in (("JCI buy & hold", idx), ("BBCA buy & hold", closes["BBCA.JK"])):
        ss = ser.loc[a:].dropna()
        rr = ss.pct_change().fillna(0.0)
        print(f"  {bl:30s} {mt.cagr(ss)*100:+6.1f}% {mt.sharpe(rr):7.2f} "
              f"{mt.max_drawdown(ss)*100:7.1f}% {riskoff_ann(rr)*100:+8.1f}%")

print("\n=== COST STRESS (full period) ===")
for n in NAMES:
    label, sig = sigs[n]
    out = []
    for mult in (1, 2, 3):
        c2 = copy.deepcopy(cfg)
        for f in ("buy_commission", "sell_commission", "half_spread", "slippage"):
            setattr(c2.costs, f, getattr(cfg.costs, f) * mult)
        r = run(sig, "2010-01-01", c2)
        out.append(f"{mult}x: CAGR {mt.cagr(r.equity)*100:+5.1f}% Sharpe {mt.sharpe(r.daily_returns):4.2f}")
    print(f"  {n:16s} " + "  |  ".join(out))

print("\n=== switch_top_n PLATEAU (regime_switch, real factory) ===")
print(f"  {'switch_top_n':>12} | {'full CAGR':>9} {'Sharpe':>7} {'maxDD':>8} | {'OOS CAGR':>9} {'Sharpe':>7} {'maxDD':>8}")
for n in (3, 5, 8, 10, 13):
    s = make_strategy(cfg, name="regime_switch", switch_top_n=n)
    sig = s.signals(prices, index_close)
    rf = run(sig, "2010-01-01")
    ro = run(sig, "2019-01-01")
    print(f"  {n:>12} | {mt.cagr(rf.equity)*100:+8.1f}% {mt.sharpe(rf.daily_returns):7.2f} "
          f"{mt.max_drawdown(rf.equity)*100:7.1f}% | {mt.cagr(ro.equity)*100:+8.1f}% "
          f"{mt.sharpe(ro.daily_returns):7.2f} {mt.max_drawdown(ro.equity)*100:7.1f}%")
