"""Phase 1: which cross-sectional signals, if any, predict forward excess return
on this 21-name IDX universe -- and do any of them survive in RISK-OFF periods?

Cheap screen before committing engine time. IC = cross-sectional Spearman rank
correlation between signal at t and forward excess return over the next h days,
averaged over days. Excess = stock return minus equal-weight universe return, so
this measures selection skill, not market beta.

No lookahead: signal uses data up to and including t, forward return starts at t.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from idxquant.config import load_config
from idxquant.data import db
from idxquant.features import indicators as ind

cfg = load_config()
con = db.connect(cfg.db_path)
prices = {t: db.load_prices(con, t) for t in cfg.tickers}
index_close = db.load_prices(con, cfg.index_ticker)["Close"]
con.close()

closes = pd.DataFrame({t: d["Close"] for t, d in prices.items()}).sort_index()
highs = pd.DataFrame({t: d["High"] for t, d in prices.items()}).reindex(closes.index)
lows = pd.DataFrame({t: d["Low"] for t, d in prices.items()}).reindex(closes.index)
vols = pd.DataFrame({t: d["Volume"] for t, d in prices.items()}).reindex(closes.index)
idx = index_close.reindex(closes.index).ffill()

print(f"universe {closes.shape[1]} names | {closes.index[0].date()} -> {closes.index[-1].date()}"
      f" | {len(closes)} trading days")

# --- liquidity mask: only rank names tradeable on that day ---
adv = pd.DataFrame({t: ind.avg_daily_value(prices[t]) for t in cfg.tickers}).reindex(closes.index)
liquid = adv >= cfg.min_adv_idr

# --- regime ---
regime_on = ind.regime_filter(idx, cfg.regime_sma).reindex(closes.index).ffill().fillna(False).astype(bool)

# ---------------------------------------------------------------- signals
def rs(w):
    return closes.pct_change(w).sub(idx.pct_change(w), axis=0)

rsi14 = pd.DataFrame({t: ind.rsi(closes[t]) for t in cfg.tickers})
vol20 = pd.DataFrame({t: ind.realized_vol(closes[t]) for t in cfg.tickers})
vol60 = pd.DataFrame({t: ind.realized_vol(closes[t], 60) for t in cfg.tickers})
hi52 = closes.rolling(252, min_periods=60).max()
sma200 = closes.rolling(200, min_periods=100).mean()
sma50 = closes.rolling(50, min_periods=25).mean()
turnover = (closes * vols).rolling(20).mean() / (closes * vols).rolling(250).mean()

signals = {
    # the shipped-and-failed screen
    "rs20 (the removed panel)":   rs(20),
    # reversal family -- the inverse hypothesis
    "REVERSAL: -rs20":            -rs(20),
    "REVERSAL: -rs5":             -rs(5),
    "REVERSAL: -rs10":            -rs(10),
    "REVERSAL: -rs60":            -rs(60),
    "REVERSAL: -ret1d":           -closes.pct_change(1),
    "REVERSAL: -rsi14":           -rsi14,
    # momentum family
    "MOM 12-1 (current system)":  closes.pct_change(231).shift(21),
    "MOM 6-1":                    closes.pct_change(105).shift(21),
    "MOM 3-1":                    closes.pct_change(42).shift(21),
    "MOM 12-0":                   closes.pct_change(252),
    # trend / structure
    "TREND: above 200d sma":      (closes / sma200 - 1),
    "TREND: above 50d sma":       (closes / sma50 - 1),
    "TREND: near 52w high":       (closes / hi52 - 1),
    "VALUE-ish: far from 52w hi": -(closes / hi52 - 1),
    # risk
    "LOW VOL: -vol20":            -vol20,
    "LOW VOL: -vol60":            -vol60,
    "VOLUME: turnover surge":     turnover,
    "VOLUME: -turnover surge":    -turnover,
}

# ---------------------------------------------------------------- IC engine
def forward_excess(h):
    fwd = closes.shift(-h) / closes - 1
    ew = fwd.mean(axis=1)                     # equal-weight universe return
    return fwd.sub(ew, axis=0)

def ic_series(sig, fwd, mask):
    s = sig.where(mask)
    f = fwd.where(mask)
    out = {}
    for d in s.index:
        a, b = s.loc[d], f.loc[d]
        ok = a.notna() & b.notna()
        if ok.sum() >= 8:
            out[d] = a[ok].rank().corr(b[ok].rank())
    return pd.Series(out)

HORIZONS = [5, 20, 60]
fwds = {h: forward_excess(h) for h in HORIZONS}

rows = []
for name, sig in signals.items():
    row = {"signal": name}
    for h in HORIZONS:
        ic = ic_series(sig, fwds[h], liquid)
        # overlapping windows -> Newey-West-ish haircut: effective n = n/h
        for label, sub in (("all", ic), ("off", ic[~regime_on.reindex(ic.index).fillna(False)])):
            if len(sub) < 50:
                row[f"ic{h}_{label}"] = np.nan
                row[f"t{h}_{label}"] = np.nan
                continue
            m, sd = sub.mean(), sub.std()
            n_eff = len(sub) / h
            row[f"ic{h}_{label}"] = m
            row[f"t{h}_{label}"] = m / sd * np.sqrt(n_eff) if sd > 0 else np.nan
        row[f"n{h}"] = len(ic)
    rows.append(row)

res = pd.DataFrame(rows).set_index("signal")

n_off = int((~regime_on).sum())
print(f"risk-off days: {n_off} of {len(regime_on)} ({n_off/len(regime_on):.0%})\n")

pd.set_option("display.width", 200, "display.max_columns", 50)
for h in HORIZONS:
    print(f"===== forward {h}d excess return =====")
    sub = res[[f"ic{h}_all", f"t{h}_all", f"ic{h}_off", f"t{h}_off"]].copy()
    sub.columns = ["IC_all", "t_all", "IC_riskoff", "t_riskoff"]
    print(sub.sort_values("IC_riskoff", ascending=False).round(3).to_string())
    print()
