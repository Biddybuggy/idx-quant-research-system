"""Robustness for the composite that would actually go live:
  risk-on  -> 12-1 momentum, top N
  risk-off -> 60-day reversal, top 5, monthly

Section 1 of bt_riskon.py showed selection adds ~nothing in risk-on (top3 +9.4%,
top8 +9.6%, owning everything +9.6%). So the top_n choice must be justified by a
surface, not by one winning cell -- otherwise this is just the largest number in
a table I searched.
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
mom = closes.pct_change(231).shift(21)


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


def rs(w):
    return closes.pct_change(w).sub(idx.pct_change(w), axis=0)


def composite(mom_top, rev_top=5, rev_lb=60, rb=21):
    a = rank_signals(mom, mom_top, rb, regime_on)
    b = rank_signals(-rs(rev_lb), rev_top, rb, risk_off)
    return a.add(b, fill_value=0).clip(0, 1).astype(int)


def run(t, a="2010-01-01", b="2026-12-31", c=cfg):
    px = {k: v.loc[a:b] for k, v in prices.items()}
    return run_backtest(px, t.loc[a:b], c)


def stats(r):
    return (mt.cagr(r.equity), mt.sharpe(r.daily_returns), mt.max_drawdown(r.equity))


print("=" * 96)
print("1. TOP_N SURFACE for the risk-on leg (risk-off leg fixed at top5/60d/monthly)")
print("=" * 96)
print(f"  {'mom_top_n':>10} | {'full CAGR':>10} {'Sharpe':>7} {'maxDD':>8} | {'OOS CAGR':>9} {'Sharpe':>7} {'maxDD':>8}")
for n in (3, 5, 8, 10, 13, 21):
    t = composite(n)
    cf, sf, df = stats(run(t))
    co, so, do = stats(run(t, "2019-01-01"))
    print(f"  {n:>10} | {cf*100:+9.1f}% {sf:7.2f} {df*100:7.1f}% | "
          f"{co*100:+8.1f}% {so:7.2f} {do*100:7.1f}%")

print()
print("=" * 96)
print("2. THE CHOSEN CELL (mom top8) UNDER STRESS")
print("=" * 96)
best = composite(8)
for mult in (1, 2, 3):
    c2 = copy.deepcopy(cfg)
    for f in ("buy_commission", "sell_commission", "half_spread", "slippage"):
        setattr(c2.costs, f, getattr(cfg.costs, f) * mult)
    c, s, d = stats(run(best, c=c2))
    print(f"  {mult}x friction        CAGR {c*100:+6.1f}%  Sharpe {s:5.2f}  maxDD {d*100:6.1f}%")

# reversal-leg params must not be load-bearing either
print()
for lb in (40, 60, 90, 120):
    for rt in (3, 5, 8):
        c, s, d = stats(run(composite(8, rev_top=rt, rev_lb=lb)))
        print(f"  rev lb={lb:3d} top={rt}   CAGR {c*100:+6.1f}%  Sharpe {s:5.2f}  maxDD {d*100:6.1f}%")

print()
print("=" * 96)
print("3. WHERE DOES THE -49% DRAWDOWN COME FROM?  (halt is ON at 20%)")
print("=" * 96)
r = run(best)
dd = r.equity / r.equity.cummax() - 1
worst = dd.idxmin()
peak = r.equity.loc[:worst].idxmax()
rec = r.equity.loc[worst:]
recovered = rec[rec >= r.equity.loc[peak]]
print(f"  worst drawdown {dd.min()*100:.1f}%  peak {peak.date()} -> trough {worst.date()}")
print(f"  recovered on   {recovered.index[0].date() if len(recovered) else 'not yet'}")
print(f"  exposure during that window: {r.exposure.loc[peak:worst].mean()*100:.0f}%")
print(f"  JCI over the same window:    {idx.loc[worst]/idx.loc[peak]*100-100:+.1f}%")
print(f"  BBCA over the same window:   {closes['BBCA.JK'].loc[worst]/closes['BBCA.JK'].loc[peak]*100-100:+.1f}%")
print("\n  drawdowns worse than 20% (the halt threshold), by year:")
under = (dd < -0.20)
for y, grp in dd[under].groupby(dd[under].index.year):
    print(f"    {y}: {len(grp):3d} days below -20%, worst {grp.min()*100:.1f}%")

print()
print("=" * 96)
print("4. FINAL COMPARISON, both periods")
print("=" * 96)
for pname, a in (("full 2010-2026", "2010-01-01"), ("OOS  2019-2026", "2019-01-01")):
    print(f"\n--- {pname} ---")
    rows = [("COMPOSITE mom8 + reversal", run(best, a)),
            ("current: mom3 only", run(rank_signals(mom, 3, 21, regime_on), a)),
            ("reversal only", run(rank_signals(-rs(60), 5, 21, risk_off), a))]
    for lbl, rr in rows:
        c, s, d = stats(rr)
        print(f"  {lbl:28s} CAGR {c*100:+6.1f}%  Sharpe {s:5.2f}  maxDD {d*100:6.1f}%  "
              f"expo {rr.exposure.mean()*100:3.0f}%")
    for bl, ser in (("JCI buy & hold", idx), ("BBCA buy & hold", closes["BBCA.JK"])):
        ss = ser.loc[a:].dropna()
        rr = ss.pct_change().fillna(0.0)
        print(f"  {bl:28s} CAGR {mt.cagr(ss)*100:+6.1f}%  Sharpe {mt.sharpe(rr):5.2f}  "
              f"maxDD {mt.max_drawdown(ss)*100:6.1f}%")
