"""Phase 3: try to KILL the one survivor (-rs60, top5, monthly, risk-off only).

A result that only works at one parameter setting, or comes from one lucky
episode, or dies when costs double, is not an edge -- it is a coincidence I
happened to stop searching at. Four attacks:

  A. parameter surface   -- is the neighbourhood good, or just this cell?
  B. per-episode         -- does it beat cash in most risk-off episodes, or one?
  C. cost stress         -- 1x, 2x, 3x friction
  D. combined product    -- momentum when regime on + reversal when regime off,
                            which is the strategy the dashboard would actually run
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
    if len(sub) < 20:
        return np.nan
    return (1 + sub).prod() ** (252 / len(sub)) - 1


def run(sig_target, c=cfg):
    return run_backtest(prices, sig_target, c)


print("=" * 78)
print("A. PARAMETER SURFACE  (risk-off annualised, full 2010-2026)")
print("   baseline cell = lookback 60, top 5, rebalance 21")
print("=" * 78)
surface = {}
for lb in (40, 60, 90, 120):
    for top_n in (3, 5, 8):
        for rb in (10, 21, 42):
            t = rank_signals(-rs(lb), top_n, rb, risk_off)
            r = run(t)
            surface[(lb, top_n, rb)] = riskoff_ann(r.daily_returns)
srf = pd.Series(surface).unstack(level=2)
srf.index.names = ["lookback", "top_n"]
srf.columns.name = "rebal_days"
print((srf * 100).round(1).to_string())
vals = srf.values.flatten()
print(f"\n  cells tested {len(vals)} | positive {np.sum(vals > 0)} | "
      f"beating JCI risk-off (-6.0%) {np.sum(vals > -0.060)} | "
      f"median {np.median(vals)*100:.1f}%")

print()
print("=" * 78)
print("B. PER-EPISODE  (each contiguous risk-off stretch >= 20 trading days)")
print("=" * 78)
base = rank_signals(-rs(60), 5, 21, risk_off)
res = run(base)
strat_r = res.daily_returns
jci_r = idx.pct_change().fillna(0.0)
bbca_r = closes["BBCA.JK"].pct_change().fillna(0.0)

blocks, start = [], None
for d, off in risk_off.items():
    if off and start is None:
        start = d
    elif not off and start is not None:
        blocks.append((start, prev))
        start = None
    prev = d
if start is not None:
    blocks.append((start, risk_off.index[-1]))
blocks = [(a, b) for a, b in blocks if len(closes.loc[a:b]) >= 20]

erows = []
for a, b in blocks:
    def tot(r):
        s = r.loc[a:b]
        return float((1 + s).prod() - 1)
    erows.append({"start": a.date(), "end": b.date(), "days": len(closes.loc[a:b]),
                  "strategy": tot(strat_r), "JCI": tot(jci_r), "BBCA": tot(bbca_r)})
ep = pd.DataFrame(erows)
ep["beat_JCI"] = ep.strategy > ep.JCI
ep["positive"] = ep.strategy > 0
print((ep.assign(**{c: (ep[c] * 100).round(1) for c in ("strategy", "JCI", "BBCA")})).to_string(index=False))
print(f"\n  episodes {len(ep)} | strategy positive in {ep.positive.sum()} "
      f"| beat JCI in {ep.beat_JCI.sum()} | median strat {ep.strategy.median()*100:+.1f}% "
      f"vs JCI {ep.JCI.median()*100:+.1f}%")

print()
print("=" * 78)
print("C. COST STRESS")
print("=" * 78)
import copy
for mult in (1, 2, 3):
    c2 = copy.deepcopy(cfg)
    for f in ("buy_commission", "sell_commission", "half_spread", "slippage"):
        setattr(c2.costs, f, getattr(cfg.costs, f) * mult)
    r = run_backtest(prices, base, c2)
    print(f"  {mult}x friction -> risk-off ann {riskoff_ann(r.daily_returns)*100:+6.1f}%  "
          f"| full CAGR {mt.cagr(r.equity)*100:+6.1f}%  | Sharpe {mt.sharpe(r.daily_returns):.2f}")

print()
print("=" * 78)
print("D. COMBINED PRODUCT: 12-1 momentum when regime ON, -rs60 reversal when OFF")
print("=" * 78)
mom = closes.pct_change(231).shift(21)
mom_t = rank_signals(mom, 3, 21, regime_on)
rev_t = rank_signals(-rs(60), 5, 21, risk_off)
combo = (mom_t + rev_t).clip(0, 1)

for label, t in (("momentum only (current system)", mom_t),
                 ("reversal only (risk-off)", rev_t),
                 ("COMBINED", combo)):
    r = run(t)
    print(f"  {label:34s} CAGR {mt.cagr(r.equity)*100:+6.1f}%  Sharpe {mt.sharpe(r.daily_returns):5.2f}  "
          f"maxDD {mt.max_drawdown(r.equity)*100:6.1f}%  risk-off ann {riskoff_ann(r.daily_returns)*100:+6.1f}%")
for label, s in (("JCI buy&hold", idx), ("BBCA buy&hold", closes["BBCA.JK"])):
    r = s.pct_change().fillna(0.0)
    print(f"  {label:34s} CAGR {mt.cagr(s.dropna())*100:+6.1f}%  Sharpe {mt.sharpe(r):5.2f}  "
          f"maxDD {mt.max_drawdown(s.dropna())*100:6.1f}%  risk-off ann {riskoff_ann(r)*100:+6.1f}%")

# same, out-of-sample only
print("\n  --- out-of-sample 2019-2026 only ---")
for label, t in (("momentum only (current system)", mom_t),
                 ("reversal only (risk-off)", rev_t),
                 ("COMBINED", combo)):
    sl = t.loc["2019-01-01":]
    px = {k: v.loc["2019-01-01":] for k, v in prices.items()}
    r = run_backtest(px, sl, cfg)
    print(f"  {label:34s} CAGR {mt.cagr(r.equity)*100:+6.1f}%  Sharpe {mt.sharpe(r.daily_returns):5.2f}  "
          f"maxDD {mt.max_drawdown(r.equity)*100:6.1f}%  risk-off ann {riskoff_ann(r.daily_returns)*100:+6.1f}%")
