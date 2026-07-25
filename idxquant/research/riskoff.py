"""The risk-off screen, for display: what the reversal strategy is looking at
while the market is weak.

This exists because the dashboard had nothing to say in exactly the conditions
the reader cares about most. The momentum engine goes to cash when JCI is below
its 200-day trend, which is correct for momentum and useless as a product — a
page that says "waiting in cash" for months is a page nobody opens.

The screen is NOT a second opinion invented for the panel. It reads the same
rank_frame() the backtested strategy trades, so what is displayed and what was
measured cannot drift apart. That was the failure of the retired tactical panel:
it showed a score nobody had ever run through the engine.

It renders ONLY while the regime is risk-off, because that is the only regime it
was validated in (risk-on IC is +0.008 — nothing). When JCI recovers, the panel
disappears and the momentum ranking is the whole story again.

Every number the panel prints about the strategy's record comes from RECORD
below, which is copied from the evidence block in strategies/reversal.py. The
live-episode figure is computed fresh on every render, gross of costs, so a bad
run shows up on the page instead of being quietly outlived by a static claim.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..features import indicators as ind
from ..strategies.reversal import RiskOffReversal

# Backtested record, 2010-01..2026-07, real engine and costs. Keep in sync with
# the docstring of strategies/reversal.py — that file is the source of truth.
RECORD = {
    "riskoff_ann": 0.172,        # strategy, annualised over risk-off days
    "riskoff_ann_jci": -0.060,   # JCI buy & hold, same days
    "episodes": 12,
    "episodes_beat_jci": 9,
    "oos_riskoff_ann": 0.196,    # out-of-sample 2019-2026
}


def _fmt_pct(v, dec=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return f"{v * 100:+.{dec}f}%"


def _episode_start(risk_off: pd.Series) -> pd.Timestamp | None:
    """First day of the risk-off stretch we are currently inside."""
    if not bool(risk_off.iloc[-1]):
        return None
    on = risk_off.to_numpy()
    i = len(on) - 1
    while i > 0 and on[i - 1]:
        i -= 1
    return risk_off.index[i]


def screen(prices: dict[str, pd.DataFrame], index_close: pd.Series,
           cfg: Config) -> dict | None:
    """Panel context, or None when the market is risk-on (panel hidden)."""
    strat = RiskOffReversal(
        lookback=int(cfg.strategy.get("rev_lookback", 60)),
        top_n=int(cfg.strategy.get("rev_top_n", 5)),
        rebalance_days=int(cfg.strategy.get("rev_rebalance_days", 21)),
        regime_sma=cfg.regime_sma,
        min_adv_idr=cfg.min_adv_idr)

    score = strat.rank_frame(prices, index_close)
    risk_off = ~ind.regime_filter(index_close, cfg.regime_sma) \
        .reindex(score.index).ffill().fillna(False).astype(bool)
    if not bool(risk_off.iloc[-1]):
        return None

    row = score.iloc[-1].dropna()
    if len(row) < strat.top_n:
        return None
    picks = row.nlargest(strat.top_n)

    closes = pd.DataFrame({t: df["Close"] for t, df in prices.items()}).sort_index()
    idx = index_close.reindex(closes.index).ffill()
    rel20 = closes.pct_change(20).sub(idx.pct_change(20), axis=0).iloc[-1]

    names = []
    for i, (t, s) in enumerate(picks.items(), start=1):
        names.append({
            "rank": i, "ticker": t, "name": t.replace(".JK", ""),
            "close_str": f"{float(closes[t].iloc[-1]):,.0f}".replace(",", "."),
            "rel_lb": -float(s),                      # return vs JCI over lookback
            "rel_lb_str": _fmt_pct(-float(s)),
            "rel20_str": _fmt_pct(float(rel20.get(t, np.nan))),
        })

    # --- live episode, recomputed every render, THROUGH THE REAL ENGINE ---
    # An equal-weighted paper sum of the picks would read -4.9% for the episode
    # running as this was written; the engine, which pays commissions and spread
    # and enforces the 20% drawdown halt, reads -21.2%. Both are "true"; only the
    # second is the number the strategy would actually have delivered, so that is
    # the one the panel prints. Costs about 0.4s, which the daily job can afford.
    from ..backtest.engine import run_backtest    # local: avoids an import cycle

    start = _episode_start(risk_off)
    sig = strat.signals(prices, index_close)
    equity = run_backtest(prices, sig, cfg).equity.loc[start:]
    ep_strat = float(equity.iloc[-1] / equity.iloc[0] - 1)
    ep_jci = float(idx.loc[start:].iloc[-1] / idx.loc[start:].iloc[0] - 1)
    ep_days = int(len(equity))

    d = start.date()
    lb = strat.lookback

    return {
        "lookback": lb,
        "top_n": strat.top_n,
        "rebalance_days": strat.rebalance_days,
        "names": names,
        "episode_start": str(d),
        "episode_days": ep_days,
        "ep_strat": ep_strat, "ep_strat_str": _fmt_pct(ep_strat),
        "ep_jci": ep_jci, "ep_jci_str": _fmt_pct(ep_jci),
        "record": RECORD,
        "record_ann_str": _fmt_pct(RECORD["riskoff_ann"], 0),
        "record_jci_str": _fmt_pct(RECORD["riskoff_ann_jci"], 0),
        # --- copy for the panel: descriptive, and caveated in the same breath ---
        "intro_id": (
            f"Saat IHSG di bawah tren jangka panjang, sistem menguji pendekatan "
            f"kebalikan: memantau {strat.top_n} saham likuid yang paling tertinggal "
            f"dari IHSG dalam {lb} hari terakhir, ditinjau ulang tiap "
            f"{strat.rebalance_days} hari bursa."),
        "intro_en": (
            f"While JCI is below its long-term trend, the system tests the opposite "
            f"approach: it tracks the {strat.top_n} liquid stocks that have fallen "
            f"furthest behind JCI over the last {lb} days, re-checked every "
            f"{strat.rebalance_days} trading days."),
        "record_id": (
            f"Diuji 2010–2026: pendekatan ini menghasilkan "
            f"{RECORD['riskoff_ann']*100:+.0f}% per tahun selama pasar lesu, "
            f"dibanding {RECORD['riskoff_ann_jci']*100:+.0f}% bila hanya mengikuti IHSG, "
            f"dan unggul di {RECORD['episodes_beat_jci']} dari "
            f"{RECORD['episodes']} periode lesu."),
        "record_en": (
            f"Tested 2010–2026: this approach returned "
            f"{RECORD['riskoff_ann']*100:+.0f}%/yr during weak markets versus "
            f"{RECORD['riskoff_ann_jci']*100:+.0f}% for simply tracking JCI, and beat "
            f"JCI in {RECORD['episodes_beat_jci']} of {RECORD['episodes']} weak periods."),
        "episode_id": (
            f"Periode lesu berjalan sejak {d} ({ep_days} hari bursa): pendekatan ini "
            f"{_fmt_pct(ep_strat)} setelah biaya, IHSG {_fmt_pct(ep_jci)}."),
        "episode_en": (
            f"Current weak period since {d} ({ep_days} trading days): this approach "
            f"{_fmt_pct(ep_strat)} after costs, JCI {_fmt_pct(ep_jci)}."),
        # True when the live episode is losing money. The panel uses this to lead
        # with the loss instead of the track record — a screen that is down must
        # not be able to present itself as if it were up.
        "losing": ep_strat < 0,
        # The two caveats a reader must not be able to miss.
        "caveat_id": (
            "Catatan penting: uji ini memakai 21 saham yang hari ini masih likuid, "
            "sehingga saham yang jatuh lalu tak pernah pulih tidak pernah ikut terbeli — "
            "hasil sesungguhnya kemungkinan lebih rendah. Di luar periode uji, "
            "keunggulannya banyak berasal dari saham komoditas yang sangat berfluktuasi. "
            "Ini portofolio latihan dan bahan riset, bukan rekomendasi beli."),
        "caveat_en": (
            "Important: this test uses 21 stocks that are still liquid today, so names "
            "that fell and never recovered could never be bought — the real result is "
            "likely lower. Out of sample, most of the edge came from highly volatile "
            "commodity stocks. This is a practice portfolio and research material, "
            "not a buy recommendation."),
    }
