"""Phase 2: run the surviving IC candidates through the REAL engine, with the
real costs, lot sizes, liquidity caps and drawdown halt.

IC says a signal points the right way. It says nothing about whether the edge is
bigger than 0.5% round-trip friction. This does.

Split: in-sample 2010-2018, out-of-sample 2019-2026. Parameters are chosen ONLY
on the IC sweep, never on the OOS equity curve.
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
rsi14 = pd.DataFrame({t: ind.rsi(closes[t]) for t in cfg.tickers})


def rs(w):
    return closes.pct_change(w).sub(idx.pct_change(w), axis=0)


class RankStrategy:
    """Long the top_n names by `sig` (already sign-oriented so higher = better),
    re-ranked every `rebalance_days`. `regime` picks which days may hold:
    'always' | 'off' (risk-off only) | 'on'."""

    def __init__(self, name, sig, top_n=5, rebalance_days=5, regime="always"):
        self.name = name
        self.sig = sig
        self.top_n = top_n
        self.rebalance_days = rebalance_days
        self.regime = regime

    def signals(self, prices, index_close):
        s = self.sig.where(liquid)
        target = pd.DataFrame(0, index=closes.index, columns=closes.columns)
        allow = {"always": pd.Series(True, index=closes.index),
                 "off": ~regime_on, "on": regime_on}[self.regime]
        held: list[str] = []
        for i, d in enumerate(closes.index):
            if i % self.rebalance_days == 0:
                row = s.loc[d].dropna()
                held = list(row.nlargest(self.top_n).index) if len(row) >= self.top_n else []
            if allow.loc[d] and held:
                target.loc[d, held] = 1
        return target.astype(int)


def riskoff_ann(rets: pd.Series) -> float:
    """Annualised compound return counting ONLY risk-off days."""
    sub = rets[~regime_on.reindex(rets.index).fillna(False)]
    if len(sub) < 20:
        return np.nan
    return (1 + sub).prod() ** (252 / len(sub)) - 1


def bench_riskoff(close: pd.Series) -> float:
    return riskoff_ann(close.reindex(closes.index).ffill().pct_change().fillna(0.0))


PERIODS = {
    "IN-SAMPLE 2010-2018": ("2010-01-01", "2018-12-31"),
    "OUT-OF-SAMPLE 2019-2026": ("2019-01-01", "2026-12-31"),
    "FULL 2010-2026": ("2010-01-01", "2026-12-31"),
}

CANDIDATES = [
    # name,                        signal,        top_n, rebal, regime
    ("REV -rs5   top5  weekly",    -rs(5),            5,   5, "always"),
    ("REV -rs5   top3  weekly",    -rs(5),            3,   5, "always"),
    ("REV -rs20  top5  monthly",   -rs(20),           5,  21, "always"),
    ("REV -rs60  top5  monthly",   -rs(60),           5,  21, "always"),
    ("REV -rs60  top5  monthly RISKOFF-ONLY", -rs(60), 5,  21, "off"),
    ("REV -rsi14 top5  weekly",    -rsi14,            5,   5, "always"),
    ("REV -rsi14 top5  monthly",   -rsi14,            5,  21, "always"),
    ("REV composite top5 monthly", -(rs(20).rank(axis=1) + rs(60).rank(axis=1)
                                     + rsi14.rank(axis=1)), 5, 21, "always"),
    # the failed screen, as the control
    ("CONTROL rs20 top5 weekly",   rs(5),             5,   5, "always"),
]

rows = []
for label, sig, top_n, rb, reg in CANDIDATES:
    strat = RankStrategy(label, sig, top_n, rb, reg)
    full_signals = strat.signals(prices, index_close)
    for pname, (a, b) in PERIODS.items():
        sl = full_signals.loc[a:b]
        if len(sl) < 250:
            continue
        px = {t: prices[t].loc[a:b] for t in cfg.tickers}
        res = run_backtest(px, sl, cfg)
        eq = res.equity
        rows.append({
            "strategy": label, "period": pname,
            "cagr": mt.cagr(eq), "sharpe": mt.sharpe(res.daily_returns),
            "maxdd": mt.max_drawdown(eq),
            "riskoff_ann": riskoff_ann(res.daily_returns),
            "n_trades": len(res.trades),
            "turnover": mt.turnover_per_year(res.trades, eq),
        })
        print(f"  ran {label:42s} {pname}")

out = pd.DataFrame(rows)

# benchmarks
bench_rows = []
for pname, (a, b) in PERIODS.items():
    for bl, series in (("JCI buy&hold", idx), ("BBCA buy&hold", closes["BBCA.JK"]),
                       ("equal-weight 21", closes.pct_change().mean(axis=1).add(1).cumprod())):
        s = series.loc[a:b].dropna()
        r = s.pct_change().fillna(0.0)
        bench_rows.append({"strategy": bl, "period": pname,
                           "cagr": mt.cagr(s), "sharpe": mt.sharpe(r),
                           "maxdd": mt.max_drawdown(s),
                           "riskoff_ann": riskoff_ann(r),
                           "n_trades": 0, "turnover": 0.0})

allr = pd.concat([out, pd.DataFrame(bench_rows)])
pd.set_option("display.width", 220, "display.max_columns", 50)
for pname in PERIODS:
    print(f"\n================ {pname} ================")
    sub = allr[allr.period == pname].drop(columns="period").set_index("strategy")
    sub = sub.sort_values("riskoff_ann", ascending=False)
    print((sub * 1).round(3).to_string())

allr.to_csv("bt_reversal.csv", index=False)
