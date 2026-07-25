"""Phase 5: the strongest objection to a buy-the-losers strategy.

Our universe is 21 names chosen because they are liquid TODAY. A strategy that
buys whatever fell hardest is exactly the strategy that, in a real universe,
would have bought the ones that kept falling to zero. Our backtest can never
buy those, because they are not in the list.

Probes:
  1. drop the commodity cyclicals (ANTM/INCO/ADRO/PTBA) -- the names whose
     drawdowns are deepest and whose survival was least assured
  2. banks + staples only -- the subset whose survival was never in doubt
  3. block bootstrap on episode returns -- is beating cash luck?
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
ALL = list(cfg.tickers)
prices_all = {t: db.load_prices(con, t) for t in ALL}
index_close = db.load_prices(con, cfg.index_ticker)["Close"]
con.close()

closes_all = pd.DataFrame({t: d["Close"] for t, d in prices_all.items()}).sort_index()
idx = index_close.reindex(closes_all.index).ffill()
regime_on = ind.regime_filter(idx, cfg.regime_sma).reindex(closes_all.index).ffill().fillna(False).astype(bool)
risk_off = ~regime_on

CYCLICALS = ["ANTM.JK", "INCO.JK", "ADRO.JK", "PTBA.JK"]
SAFE = ["BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "TLKM.JK",
        "INDF.JK", "ICBP.JK", "KLBF.JK", "UNVR.JK"]

SUBSETS = {
    "all 21 (baseline)": ALL,
    "ex-commodity (17)": [t for t in ALL if t not in CYCLICALS],
    "banks+staples (9)": SAFE,
}


def riskoff_ann(rets):
    sub = rets[risk_off.reindex(rets.index).fillna(False)]
    return (1 + sub).prod() ** (252 / len(sub)) - 1 if len(sub) >= 20 else np.nan


def build(tickers, top_n=5, rb=21, lb=60):
    px = {t: prices_all[t] for t in tickers}
    closes = pd.DataFrame({t: d["Close"] for t, d in px.items()}).sort_index()
    adv = pd.DataFrame({t: ind.avg_daily_value(px[t]) for t in tickers}).reindex(closes.index)
    liquid = adv >= cfg.min_adv_idr
    sig = -(closes.pct_change(lb).sub(idx.reindex(closes.index).pct_change(lb), axis=0))
    s = sig.where(liquid)
    target = pd.DataFrame(0, index=closes.index, columns=closes.columns)
    held = []
    for i, d in enumerate(closes.index):
        if i % rb == 0:
            row = s.loc[d].dropna()
            held = list(row.nlargest(top_n).index) if len(row) >= top_n else []
        if risk_off.loc[d] and held:
            target.loc[d, held] = 1
    return px, target.astype(int)


print("=" * 92)
print("1+2. DOES THE EDGE SURVIVE WITHOUT THE NAMES MOST EXPOSED TO SURVIVORSHIP BIAS?")
print("=" * 92)
for label, tickers in SUBSETS.items():
    px, t = build(tickers)
    for pname, (a, b) in (("full  2010-2026", ("2010-01-01", "2026-12-31")),
                          ("OOS   2019-2026", ("2019-01-01", "2026-12-31"))):
        pxs = {k: v.loc[a:b] for k, v in px.items()}
        r = run_backtest(pxs, t.loc[a:b], cfg)
        print(f"  {label:22s} {pname}  CAGR {mt.cagr(r.equity)*100:+6.1f}%  "
              f"Sharpe {mt.sharpe(r.daily_returns):5.2f}  maxDD {mt.max_drawdown(r.equity)*100:6.1f}%  "
              f"risk-off {riskoff_ann(r.daily_returns)*100:+6.1f}%")
    print()

print("=" * 92)
print("3. BLOCK BOOTSTRAP: is 'beat JCI in 9 of 12 episodes' distinguishable from luck?")
print("=" * 92)
px, t = build(ALL)
res = run_backtest(px, t, cfg)
strat_r = res.daily_returns
jci_r = idx.pct_change().fillna(0.0).reindex(strat_r.index).fillna(0.0)

blocks, start, prev = [], None, risk_off.index[0]
for d, off in risk_off.items():
    if off and start is None:
        start = d
    elif not off and start is not None:
        blocks.append((start, prev)); start = None
    prev = d
if start is not None:
    blocks.append((start, risk_off.index[-1]))
blocks = [(a, b) for a, b in blocks if len(closes_all.loc[a:b]) >= 20]

diffs = np.array([float((1 + strat_r.loc[a:b]).prod() - (1 + jci_r.loc[a:b]).prod())
                  for a, b in blocks])
obs = diffs.mean()
rng = np.random.default_rng(0)
draws = np.array([rng.choice(diffs, len(diffs), replace=True).mean() for _ in range(20000)])
print(f"  episodes: {len(diffs)}")
print(f"  mean excess over JCI per episode: {obs*100:+.1f}pp")
print(f"  bootstrap 95% CI: [{np.percentile(draws,2.5)*100:+.1f}pp, {np.percentile(draws,97.5)*100:+.1f}pp]")
print(f"  P(mean excess <= 0) = {float((draws <= 0).mean()):.3f}")
# sign test: 9 of 12 under a fair coin
from math import comb
p_sign = sum(comb(12, k) for k in range(9, 13)) / 2 ** 12
print(f"  sign test, >=9 of 12 episodes beating JCI by chance: p = {p_sign:.3f}")
print(f"  drop the single best episode (2020): mean excess "
      f"{np.sort(diffs)[:-1].mean()*100:+.1f}pp")
