"""Short-term / tactical opportunity layer — leaders in a weak market.

The long-term momentum view is gated by the JCI regime filter: when the index
is below its 200-day trend, every long-term signal is suppressed and the whole
watchlist reads "menunggu regime". But traders (like the experienced reader this
is built for) still find names that rise while the market sags. That edge is
*relative strength* — a stock beating the index — plus a clean short-term
structure (above its 20-day average, near a 20-day high, not blown-off).

This module scores that setup PER STOCK, independent of the market regime, from
daily closes already in the DB. It is deliberately honest: these are swing /
short-term setups on *daily* data — not intraday day-trade timing, and not a
validated profitable edge. Every opportunity ships with a suggested stop so the
reader sizes the risk. The reader decides; we show the evidence.

Text is Bahasa Indonesia first (the primary audience).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import indicators as ind

# Score component weights (sum to 1.0). Relative strength dominates by design —
# it is the piece that actually answers "which names go up when the market is weak".
_W_RS = 0.45          # relative strength vs the index (20d + 60d)
_W_TREND = 0.20       # above a rising 20-day average
_W_BREAKOUT = 0.15    # proximity to the 20-day high
_W_RSI = 0.10         # healthy momentum, not blown-off or falling-knife
_W_VOL = 0.10         # recent volume confirming the move

# Gates a name must clear to be called a short-term opportunity (regime-agnostic).
OPPORTUNITY_SCORE = 55.0   # composite score, 0..100
_RSI_MAX = 78.0            # above this the move is over-extended for a fresh entry
_STOP_ATR_MULT = 1.5       # suggested stop = entry - 1.5 x ATR


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _rsi_score(rsi: float | None) -> float:
    """Peak reward for RSI in the 50-70 momentum band; taper outside it."""
    if rsi is None or np.isnan(rsi):
        return 0.5
    if 50.0 <= rsi <= 70.0:
        return 1.0
    if rsi < 50.0:
        return _clip01((rsi - 30.0) / 20.0)     # 30->0, 50->1
    return _clip01((80.0 - rsi) / 10.0)          # 70->1, 80->0


def _rsi_score_vec(rsi: pd.Series) -> pd.Series:
    """Vector form of _rsi_score; NaN scores 0.5 (no opinion), as in the scalar."""
    below = ((rsi - 30.0) / 20.0).clip(0.0, 1.0)
    above = ((80.0 - rsi) / 10.0).clip(0.0, 1.0)
    out = pd.Series(np.where(rsi < 50.0, below, above), index=rsi.index)
    out[(rsi >= 50.0) & (rsi <= 70.0)] = 1.0
    return out.where(rsi.notna(), 0.5)


def score_history(df: pd.DataFrame, index_close: pd.Series, min_adv_bn: float,
                  flagged: bool = False) -> pd.DataFrame:
    """The tactical score for one stock across its WHOLE history, one row per day.

    This is the single source of truth for the score. The dashboard reads the
    last row; the backtest reads every row. Two implementations of the same
    formula would drift apart silently — the same reason the backtester and the
    paper executor share one step().

    Every column is computable from data up to and including that day's close.
    The engine, not this function, applies the one-day execution lag.
    """
    close = df["Close"]
    idx = index_close.reindex(close.index).ffill()

    rs20 = close.pct_change(20) - idx.pct_change(20)
    rs60 = close.pct_change(60) - idx.pct_change(60)

    sma20 = close.rolling(20).mean()
    above_sma = (close > sma20).fillna(False)
    slope_up = (sma20 > sma20.shift(5)).fillna(False)

    hi20 = close.rolling(20).max()
    from_hi20 = (close / hi20 - 1).where(hi20 > 0)

    tv = close * df["Volume"]
    tv20 = tv.rolling(20).mean()
    vol_confirm = (tv.rolling(5).mean() / tv20).where(tv20 > 0)

    rsi = ind.rsi(close)
    atr_pct = ind.atr(df) / close
    adv_bn = ind.avg_daily_value(df) / 1e9

    # --- component sub-scores in [0,1] (NaN fallbacks match the scalar path) ---
    rs_sub = ((rs20.fillna(0.0) / 0.10).clip(0, 1) * 0.6
              + (rs60.fillna(0.0) / 0.20).clip(0, 1) * 0.4)
    trend_sub = pd.Series(np.where(above_sma & slope_up, 1.0,
                                   np.where(above_sma, 0.5, 0.0)), index=close.index)
    breakout_sub = (1 + from_hi20.fillna(-1.0) / 0.10).clip(0, 1)
    rsi_sub = _rsi_score_vec(rsi)
    vol_sub = ((vol_confirm.fillna(0.8) - 0.8) / 0.6).clip(0, 1)

    score = (100.0 * (_W_RS * rs_sub + _W_TREND * trend_sub + _W_BREAKOUT * breakout_sub
                      + _W_RSI * rsi_sub + _W_VOL * vol_sub)).round(1)

    liquid = adv_bn >= min_adv_bn
    is_opportunity = (liquid & above_sma & (rs20 > 0)
                      & (rsi.isna() | (rsi < _RSI_MAX))
                      & (not flagged) & (score >= OPPORTUNITY_SCORE))

    stop_pct = -_STOP_ATR_MULT * atr_pct
    return pd.DataFrame({
        "st_score": score, "rs20": rs20, "rs60": rs60,
        "st_above_sma": above_sma, "st_slope_up": slope_up,
        "from_hi20": from_hi20, "vol_confirm": vol_confirm,
        "rsi": rsi, "atr_pct": atr_pct, "adv_bn": adv_bn, "liquid": liquid,
        "stop_pct": stop_pct, "stop_price": close * (1 + stop_pct),
        "is_opportunity": is_opportunity.fillna(False).astype(bool),
    })


def compute(df: pd.DataFrame, index_close: pd.Series, min_adv_bn: float,
            flagged: bool = False) -> dict:
    """Today's tactical readout for one stock — the last row of score_history()
    plus the plain-language view the dashboard shows."""
    row = score_history(df, index_close, min_adv_bn, flagged).iloc[-1]

    def val(k):
        v = row[k]
        return None if (isinstance(v, float) and np.isnan(v)) else float(v)

    rsi = val("rsi")
    rs20 = row["rs20"]
    view, reason = _view(bool(row["is_opportunity"]), bool(row["st_above_sma"]),
                         rs20, row["from_hi20"], row["vol_confirm"],
                         row["stop_pct"], bool(row["liquid"]), rsi)
    return {
        "st_score": float(row["st_score"]),
        "rs20": val("rs20"), "rs60": val("rs60"),
        "st_above_sma": bool(row["st_above_sma"]),
        "st_slope_up": bool(row["st_slope_up"]),
        "from_hi20": val("from_hi20"), "vol_confirm": val("vol_confirm"),
        "stop_pct": val("stop_pct"), "stop_price": val("stop_price"),
        "is_opportunity": bool(row["is_opportunity"]),
        "st_view": view, "st_reason": reason,
    }


def _view(is_opp: bool, above_sma: bool, rs20: float, from_hi20: float,
          vol_confirm: float, stop_pct: float, liquid: bool,
          rsi: float | None) -> tuple[str, str]:
    """(short-term view label, one-sentence reason) — ID-first."""
    rs_txt = f"{rs20:+.1%} vs IHSG" if not np.isnan(rs20) else "data kurang"
    stop_txt = (f" Batas rugi usulan ~{stop_pct:+.1%} (1,5×ATR)." if not np.isnan(stop_pct) else "")
    if is_opp:
        hi_txt = ("menembus/di dekat puncak 20 hari"
                  if not np.isnan(from_hi20) and from_hi20 > -0.02 else "di atas rata-rata 20 hari")
        vol_txt = (" dengan volume menguat" if not np.isnan(vol_confirm) and vol_confirm > 1.05 else "")
        return ("peluang jangka pendek",
                f"Lebih kuat dari pasar ({rs_txt}), {hi_txt}{vol_txt} — setup jangka "
                f"pendek meski IHSG lemah.{stop_txt}")
    if not liquid:
        return ("likuiditas tipis",
                "Nilai transaksi harian di bawah ambang — sulit masuk/keluar dengan aman untuk trading jangka pendek.")
    if rsi is not None and not np.isnan(rsi) and rsi >= _RSI_MAX and above_sma:
        return ("terlalu jauh",
                f"Kuat ({rs_txt}) tetapi RSI {rsi:.0f} — sudah naik banyak; tunggu koreksi/konsolidasi sebelum masuk.")
    if above_sma and not np.isnan(rs20) and rs20 > 0:
        return ("pantau",
                f"Mulai memimpin pasar ({rs_txt}) dan di atas rata-rata 20 hari, tetapi setup belum matang — pantau.")
    return ("lemah jangka pendek",
            f"Tren jangka pendek belum mendukung ({rs_txt}); bukan kandidat trading saat ini.")


_ST_VIEW_META = {
    # view -> (english label, css class reused from the long-term views)
    "peluang jangka pendek": ("short-term opportunity", "buy"),
    "pantau": ("watch", "wait"),
    "terlalu jauh": ("over-extended", "wait"),
    "likuiditas tipis": ("thin liquidity", "muted"),
    "lemah jangka pendek": ("short-term weak", "weak"),
    "data kurang": ("insufficient data", "muted"),
}


def add_display_fields(r: dict) -> None:
    """Formatted strings for the tactical fields on a research card."""
    def pct(v, dec=1):
        return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * 100:+.{dec}f}%"
    r["rs20_str"] = pct(r["rs20"])
    r["rs60_str"] = pct(r["rs60"])
    r["from_hi20_str"] = pct(r["from_hi20"])
    r["stop_pct_str"] = pct(r["stop_pct"])
    r["stop_price_str"] = ("–" if r["stop_price"] is None
                           else f"{r['stop_price']:,.0f}".replace(",", "."))
    r["vol_confirm_str"] = ("–" if r["vol_confirm"] is None else f"{r['vol_confirm']:.2f}×")
    r["st_view_en"], r["st_view_css"] = _ST_VIEW_META[r["st_view"]]


def opportunities(cards: list[dict], n: int = 5) -> list[dict]:
    """Today's short-term opportunities: qualifying names, strongest first."""
    picks = [c for c in cards if c.get("is_opportunity")]
    picks.sort(key=lambda c: c["st_score"], reverse=True)
    return picks[:n]


def top_relative_strength_line(cards: list[dict], n: int = 3) -> str:
    """One Telegram line naming today's short-term opportunities.

    Deliberately reuses `opportunities()` so the message and the dashboard panel
    can never disagree — a name in the message is always findable on the site.
    Silent when nothing qualifies, matching the panel's empty state.
    """
    picks = opportunities(cards, n)
    if not picks:
        return ""
    parts = [f"{c['name']} {c['rs20'] * 100:+.0f}% vs IHSG" for c in picks]
    return "💪 Peluang jangka pendek: " + " · ".join(parts)
