"""Per-stock research readouts — the heart of the research-assistant product.

For every watchlist name, compute an evidence card: momentum rank, trend
state, RSI, 52-week position, volatility, liquidity, and what the system
thinks (with the reason). The reader decides; we show the data.

Everything is computed from daily closes already in the DB — no lookahead,
no external calls. Text is Bahasa Indonesia first (the primary audience).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..features import indicators as ind


def market_overview(index_close: pd.Series, cfg: Config) -> dict:
    """JCI context: level, recent change, regime state."""
    sma = ind.sma(index_close, cfg.regime_sma).iloc[-1]
    regime_on = bool(ind.regime_filter(index_close, cfg.regime_sma).iloc[-1])
    level = float(index_close.iloc[-1])
    chg_1d = float(index_close.pct_change(1).iloc[-1])
    chg_1m = float(index_close.pct_change(21).iloc[-1])
    vs_sma = level / sma - 1
    return {
        "level": level, "chg_1d": chg_1d, "chg_1m": chg_1m,
        "vs_sma": float(vs_sma),
        "regime_on": regime_on,
        "sma_window": cfg.regime_sma,
        "level_str": f"{level:,.0f}".replace(",", "."),
        "chg_1d_str": _pct(chg_1d, 2),
        "chg_1m_str": _pct(chg_1m),
        "vs_sma_str": _pct(vs_sma),
    }


def _trend(close: pd.Series) -> str:
    c = close.iloc[-1]
    s50, s200 = ind.sma(close, 50).iloc[-1], ind.sma(close, 200).iloc[-1]
    if np.isnan(s200):
        return "campuran"
    if c > s50 > s200:
        return "naik"
    if c < s50 < s200:
        return "turun"
    return "campuran"


def _view(ticker: str, rank: int | None, mom: float, regime_on: bool,
          held: set[str], top_n: int) -> tuple[str, str]:
    """(view label, one-sentence reason) — the system's opinion, ID-first."""
    if ticker in held:
        return ("dipegang",
                "Sedang dipegang portofolio latihan; keluar saat rebalance bulanan "
                "jika keluar dari peringkat atas, atau segera jika regime memburuk.")
    if mom is None or np.isnan(mom):
        return ("data kurang", "Riwayat harga belum cukup untuk menghitung momentum 12 bulan.")
    if rank is not None and rank <= top_n and mom > 0:
        if regime_on:
            return ("kandidat beli",
                    f"Peringkat {rank} momentum dengan tren positif {mom:+.1%}; "
                    "masuk daftar beli sistem pada rebalance bulan berikutnya.")
        return ("menunggu regime",
                f"Momentum kuat (peringkat {rank}, {mom:+.1%}), tetapi IHSG masih "
                "di bawah tren jangka panjang — sistem menunggu pasar membaik.")
    if mom < -0.10:
        return ("lemah",
                f"Momentum 12 bulan {mom:+.1%} — tren menurun; sistem menghindari.")
    return ("netral",
            f"Momentum {mom:+.1%}, belum masuk peringkat atas; sistem tidak tertarik saat ini.")


def stock_research(prices: dict[str, pd.DataFrame], index_close: pd.Series,
                   cfg: Config, held_tickers: set[str],
                   quality_flags: set[str] | None = None) -> list[dict]:
    """One evidence card per ticker, sorted by momentum rank."""
    closes = pd.DataFrame({t: df["Close"] for t, df in prices.items()}).sort_index()
    lookback = int(cfg.strategy.get("lookback", 252))
    skip = int(cfg.strategy.get("skip", 21))
    top_n = int(cfg.strategy.get("top_n", 3))
    mom = closes.pct_change(lookback - skip).shift(skip).iloc[-1]
    ranks = mom.rank(ascending=False)
    regime_on = bool(ind.regime_filter(index_close, cfg.regime_sma).iloc[-1])
    quality_flags = quality_flags or set()

    out = []
    for t, df in prices.items():
        close = df["Close"]
        c = float(close.iloc[-1])
        m = float(mom.get(t, np.nan))
        rank = int(ranks[t]) if not np.isnan(m) else None
        rsi = float(ind.rsi(close).iloc[-1])
        hi_52w = float(close.rolling(252, min_periods=60).max().iloc[-1])
        adv = float(ind.avg_daily_value(df).iloc[-1])
        view, reason = _view(t, rank, m, regime_on, held_tickers, top_n)
        out.append({
            "ticker": t, "name": t.replace(".JK", ""),
            "close": c,
            "chg_1d": float(close.pct_change(1).iloc[-1]),
            "chg_1m": float(close.pct_change(21).iloc[-1]),
            "mom_12_1": None if np.isnan(m) else m,
            "rank": rank, "n_ranked": int(mom.notna().sum()),
            "trend": _trend(close),
            "rsi": None if np.isnan(rsi) else rsi,
            "rsi_label": ("jenuh beli" if rsi > 70 else
                          "jenuh jual" if rsi < 30 else "netral") if not np.isnan(rsi) else "-",
            "pos_52w": c / hi_52w - 1 if hi_52w > 0 else None,
            "vol_20d": float(ind.realized_vol(close).iloc[-1]),
            "atr_pct": float(ind.atr(df).iloc[-1] / c),
            "adv_bn": adv / 1e9,
            "view": view, "view_reason": reason,
            "quality_flag": t in quality_flags,
        })
    out.sort(key=lambda r: r["rank"] if r["rank"] is not None else 999)
    for r in out:
        _add_display_fields(r)
    return out


_VIEW_META = {
    # view -> (english label, css class)
    "dipegang": ("held", "hold"),
    "kandidat beli": ("buy candidate", "buy"),
    "menunggu regime": ("awaiting regime", "wait"),
    "lemah": ("weak", "weak"),
    "netral": ("neutral", "muted"),
    "data kurang": ("insufficient data", "muted"),
}


def _pct(v: float | None, dec: int = 1) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return f"{v * 100:+.{dec}f}%"


def _add_display_fields(r: dict) -> None:
    r["close_str"] = f"{r['close']:,.0f}".replace(",", ".")
    r["chg_1d_str"] = _pct(r["chg_1d"])
    r["chg_1m_str"] = _pct(r["chg_1m"])
    r["mom_str"] = _pct(r["mom_12_1"])
    r["rsi_str"] = f"{r['rsi']:.0f}" if r["rsi"] is not None else "–"
    r["pos_52w_str"] = _pct(r["pos_52w"])
    r["vol_str"] = f"{r['vol_20d'] * 100:.0f}%" if not np.isnan(r["vol_20d"]) else "–"
    r["adv_str"] = f"{r['adv_bn']:,.0f}".replace(",", ".")
    r["trend_arrow"] = {"naik": "↗", "turun": "↘", "campuran": "→"}[r["trend"]]
    r["view_en"], r["view_css"] = _VIEW_META[r["view"]]


def top_movers_line(research: list[dict], n: int = 3) -> str:
    """One Telegram line: the strongest momentum names right now."""
    ranked = [r for r in research if r["rank"] is not None][:n]
    if not ranked:
        return ""
    parts = [f"{r['name']} {r['mom_12_1']:+.0%}" for r in ranked]
    return "🔎 Momentum teratas: " + " · ".join(parts)
