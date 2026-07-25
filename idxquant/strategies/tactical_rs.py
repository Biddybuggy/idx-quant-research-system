"""Tactical relative strength — the dashboard's short-term screen, as a strategy.

Hypothesis under test: names that outperform the index while holding a clean
short-term structure keep outperforming for days-to-weeks, INDEPENDENT of the
market regime. This is the claim the dashboard's "peluang jangka pendek" panel
makes, and it is the reason this strategy has no regime filter — removing it is
the experiment, not an oversight.

Scoring is imported from idxquant.research.tactical, not reimplemented, so the
backtest can only ever test what the dashboard actually shows.

Rules, all computed at the close of day t (the engine applies the entry lag):
  - candidates = tactical.is_opportunity (liquid, above 20d SMA, positive RS vs
    the index, RSI < 78, score >= threshold)
  - hold the `top_n` candidates with the highest score
  - exit when a name leaves the top `top_n`, stops qualifying, or closes below
    its entry stop (entry close x (1 - 1.5 x ATR%))

STOP FIDELITY: the engine is a daily-bar simulator that fills at the next open,
so this is a CLOSE-based stop — it exits the open after the close that breached,
not intraday at the stop price. A real broker stop would fill intraday, usually
better in a drift and worse in a gap. The tested rule is deliberately the one
the engine can honour honestly; it is not the rule the dashboard's stop implies.
"""
from __future__ import annotations

import pandas as pd

from ..research import tactical


class TacticalRelativeStrength:
    def __init__(self, top_n: int = 5, min_adv_bn: float = 10.0,
                 min_score: float = tactical.OPPORTUNITY_SCORE,
                 rebalance_days: int = 1, use_stop: bool = True):
        self.top_n = top_n
        self.min_adv_bn = min_adv_bn
        self.min_score = min_score
        self.rebalance_days = rebalance_days
        self.use_stop = use_stop
        self.name = (f"tactrs_top{top_n}_score{min_score:g}"
                     f"_rb{rebalance_days}{'_stop' if use_stop else ''}")

    def _frames(self, prices: dict[str, pd.DataFrame], index_close: pd.Series):
        hist = {t: tactical.score_history(df, index_close, self.min_adv_bn)
                for t, df in prices.items()}
        score = pd.DataFrame({t: h["st_score"] for t, h in hist.items()}).sort_index()
        opp = pd.DataFrame({t: h["is_opportunity"] for t, h in hist.items()}) \
                .reindex(score.index).fillna(False).astype(bool)
        stop = pd.DataFrame({t: h["stop_pct"] for t, h in hist.items()}).reindex(score.index)
        closes = pd.DataFrame({t: df["Close"] for t, df in prices.items()}).reindex(score.index)
        if self.min_score != tactical.OPPORTUNITY_SCORE:
            opp &= score >= self.min_score
        return score, opp, stop, closes

    def signals(self, prices: dict[str, pd.DataFrame],
                index_close: pd.Series) -> pd.DataFrame:
        score, opp, stop, closes = self._frames(prices, index_close)
        target = pd.DataFrame(0, index=score.index, columns=score.columns)

        held: dict[str, float] = {}      # ticker -> stop price fixed at entry
        for i, date in enumerate(score.index):
            close_row = closes.loc[date]

            # 1. stop-outs first: a breached stop exits regardless of rank.
            if self.use_stop:
                for t in [t for t, sp in held.items()
                          if close_row.get(t) is not None and close_row[t] < sp]:
                    held.pop(t)

            # 2. re-screen only on rebalance days. Between them the stop is the
            #    only risk control — otherwise "no longer qualifies" fires first
            #    on every drop and the stop can never bind at any cadence.
            if i % self.rebalance_days == 0:
                for t in [t for t in held if not opp.loc[date, t]]:
                    held.pop(t)

                ranked = score.loc[date][opp.loc[date]].nlargest(self.top_n)
                keep = set(ranked.index)
                for t in [t for t in held if t not in keep]:
                    held.pop(t)
                for t in ranked.index:
                    if t not in held:
                        sp = stop.loc[date, t]
                        held[t] = (float(close_row[t]) * (1 + sp)
                                   if self.use_stop and pd.notna(sp) else 0.0)

            if held:
                target.loc[date, list(held)] = 1
        return target.astype(int)
