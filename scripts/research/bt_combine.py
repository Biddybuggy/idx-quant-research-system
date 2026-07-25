"""Phase 4: why is COMBINED worse than its own risk-off leg, and what is the
right way to run both? Suspects: (a) the momentum leg is simply weak, (b) the
20% drawdown halt is a shared resource -- the momentum leg spends it and the
reversal leg gets blocked from trading during the very episodes it is good at.
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


def riskoff_ann(rets):
    sub = rets[risk_off.reindex(rets.index).fillna(False)]
    return (1 + sub).prod() ** (252 / len(sub)) - 1 if len(sub) >= 20 else np.nan


def show(label, t, c=cfg, a=None, b=None):
    px = prices if a is None else {k: v.loc[a:b] for k, v in prices.items()}
    sl = t if a is None else t.loc[a:b]
    r = run_backtest(px, sl, c)
    print(f"  {label:38s} CAGR {mt.cagr(r.equity)*100:+6.1f}%  Sharpe {mt.sharpe(r.daily_returns):5.2f}  "
          f"maxDD {mt.max_drawdown(r.equity)*100:6.1f}%  risk-off {riskoff_ann(r.daily_returns)*100:+6.1f}%  "
          f"expo {r.exposure.mean()*100:4.0f}%")
    return r


mom = closes.pct_change(231).shift(21)
rev = -rs(60)
rev_t = rank_signals(rev, 5, 21, risk_off)

no_halt = copy.deepcopy(cfg)
no_halt.max_drawdown_halt = 0.99   # effectively off

print("=" * 100)
print("1. IS THE MOMENTUM LEG ITSELF WEAK?  (risk-on days only, full period)")
print("=" * 100)
for tn in (3, 5, 8):
    show(f"momentum top{tn} monthly", rank_signals(mom, tn, 21, regime_on))
print("  -- same, with the 20% drawdown halt disabled --")
for tn in (3, 5, 8):
    show(f"momentum top{tn} monthly (no halt)", rank_signals(mom, tn, 21, regime_on), no_halt)

print()
print("=" * 100)
print("2. IS THE HALT THE PROBLEM FOR THE COMBINED PORTFOLIO?")
print("=" * 100)
mom3 = rank_signals(mom, 3, 21, regime_on)
mom5 = rank_signals(mom, 5, 21, regime_on)
combo3 = (mom3 + rev_t).clip(0, 1)
combo5 = (mom5 + rev_t).clip(0, 1)
show("reversal only", rev_t)
show("reversal only (no halt)", rev_t, no_halt)
show("combined mom3+rev", combo3)
show("combined mom3+rev (no halt)", combo3, no_halt)
show("combined mom5+rev", combo5)
show("combined mom5+rev (no halt)", combo5, no_halt)

print()
print("=" * 100)
print("3. THE SAME TABLE, OUT-OF-SAMPLE 2019-2026 ONLY")
print("=" * 100)
for lbl, t in (("reversal only", rev_t), ("combined mom3+rev", combo3),
               ("combined mom5+rev", combo5)):
    show(lbl, t, cfg, "2019-01-01", "2026-12-31")
    show(lbl + " (no halt)", t, no_halt, "2019-01-01", "2026-12-31")
print("  benchmarks OOS:")
for label, s in (("JCI", idx), ("BBCA", closes["BBCA.JK"])):
    ss = s.loc["2019-01-01":].dropna()
    r = ss.pct_change().fillna(0.0)
    print(f"  {label:38s} CAGR {mt.cagr(ss)*100:+6.1f}%  Sharpe {mt.sharpe(r):5.2f}  "
          f"maxDD {mt.max_drawdown(ss)*100:6.1f}%  risk-off {riskoff_ann(r)*100:+6.1f}%")

print()
print("=" * 100)
print("4. NO-LOOKAHEAD CHECK: truncating history must not change past signals")
print("=" * 100)
cut = closes.index[-300]
full = rank_signals(rev, 5, 21, risk_off).loc[:cut]
sub_closes = closes.loc[:cut]
sub_idx = idx.loc[:cut]
sub_rev = -(sub_closes.pct_change(60).sub(sub_idx.pct_change(60), axis=0))
sub_liquid = liquid.loc[:cut]
sub_off = risk_off.loc[:cut]
t2 = pd.DataFrame(0, index=sub_closes.index, columns=sub_closes.columns)
held = []
s = sub_rev.where(sub_liquid)
for i, d in enumerate(sub_closes.index):
    if i % 21 == 0:
        row = s.loc[d].dropna()
        held = list(row.nlargest(5).index) if len(row) >= 5 else []
    if sub_off.loc[d] and held:
        t2.loc[d, held] = 1
diff = int((full.values != t2.astype(int).values).sum())
print(f"  signal cells differing after truncation: {diff}  "
      f"({'PASS - no lookahead' if diff == 0 else 'FAIL'})")
