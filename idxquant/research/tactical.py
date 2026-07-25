"""Short-term relative strength — DESCRIPTIVE ONLY. Tested, and it does not work.

This started as an opportunity screen: find names beating the index while the
market sags, on the theory that relative strength persists. It was backtested on
2010-2026 (21 names, 18 risk-off episodes) and the theory is wrong for this
universe. Annualised return during risk-off periods:

    tactical daily   -20.7% gross / -30.6% net
    JCI buy & hold    -6.4%
    BBCA buy & hold   +7.7%

It loses to holding the index in exactly the weak markets it was built for, and
it is negative GROSS of costs — so this is not friction a slower cadence fixes.
The cross-sectional rank IC of 20-day relative strength against forward 20-day
excess return is -0.047 in risk-off periods: these names mean-revert.
momentum.py already skips the most recent month to dodge short-term reversal;
this bought exactly that window.

So the numbers here are now presented as DESCRIPTION of what a stock has done,
never as a suggestion of what to do next. No score, no ranking, no entry, no
stop — those all imply a recommendation the evidence does not support. The
scoring machinery below is retained only so the finding stays reproducible via
idxquant/strategies/tactical_rs.py; it must not drive anything user-facing.

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
    return {
        "rs20": val("rs20"), "rs60": val("rs60"),
        "st_above_sma": bool(row["st_above_sma"]),
        "from_hi20": val("from_hi20"),
        "st_note": _note(rs20, row["from_hi20"], bool(row["st_above_sma"])),
    }


def _note(rs20: float, from_hi20: float, above_sma: bool) -> str:
    """One descriptive sentence — what the stock DID, never what to do.

    No entry, no stop, no "opportunity": the screen those implied was tested and
    underperformed buy-and-hold in weak markets (see the module docstring).
    """
    if np.isnan(rs20):
        return "Data belum cukup untuk mengukur pergerakan jangka pendek."
    vs = "lebih kuat dari" if rs20 > 0 else "lebih lemah dari"
    where = ("di atas" if above_sma else "di bawah")
    hi = ("" if np.isnan(from_hi20)
          else f" dan {abs(from_hi20) * 100:.0f}% dari puncak 20 harinya")
    return (f"Dalam 20 hari terakhir bergerak {vs} IHSG ({rs20:+.1%}), "
            f"{where} rata-rata 20 hari{hi}. Ini catatan pergerakan, bukan sinyal beli.")


def add_display_fields(r: dict) -> None:
    """Formatted strings for the descriptive short-term fields."""
    def pct(v, dec=1):
        return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * 100:+.{dec}f}%"
    r["rs20_str"] = pct(r["rs20"])
    r["rs60_str"] = pct(r["rs60"])
    r["from_hi20_str"] = pct(r["from_hi20"])
