"""The short-term block is DESCRIPTIVE. It must report movement accurately and
must never phrase it as advice.

This file used to assert the opposite — that a leader in a falling market was an
"opportunity". That screen was backtested and underperformed buy-and-hold in
exactly those conditions (see idxquant/research/tactical.py), so the guarantees
worth holding now are: the numbers are right, and the wording does not suggest
an entry.

Run:  .venv/bin/python tests/test_tactical.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from idxquant.research import tactical


def _ohlcv(close: pd.Series, vol: float = 1e7) -> pd.DataFrame:
    """Wrap a close series in a plausible OHLCV frame (tight bars, flat volume)."""
    return pd.DataFrame({
        "Open": close.shift(1).fillna(close.iloc[0]),
        "High": close * 1.005,
        "Low": close * 0.995,
        "Close": close,
        "Volume": vol,
    })


def _card(close: pd.Series, index_close: pd.Series, min_adv_bn: float,
          flagged: bool = False, vol: float = 1e7) -> dict:
    return tactical.compute(_ohlcv(close, vol), index_close, min_adv_bn, flagged)


# Words that would turn a description into a recommendation.
_ADVICE_WORDS = ["peluang", "beli", "masuk", "batas rugi", "target",
                 "kandidat", "rekomendasi", "sinyal beli"]


def main():
    n = 150
    dates = pd.bdate_range("2025-01-01", periods=n)
    index_close = pd.Series(np.linspace(1000, 800, n), index=dates)   # market -20%
    leader = pd.Series(np.linspace(1000, 1300, n), index=dates)       # rises anyway
    laggard = pd.Series(np.linspace(1000, 700, n), index=dates)       # falls harder

    min_adv_bn = 10.0
    lead = _card(leader, index_close, min_adv_bn, vol=1e7)
    lag = _card(laggard, index_close, min_adv_bn, vol=2e7)

    print(f"leader : rs20 {lead['rs20']:+.3f}  {lead['st_note']}")
    print(f"laggard: rs20 {lag['rs20']:+.3f}  {lag['st_note']}")

    # 1. The measured numbers are correct and correctly signed.
    assert lead["rs20"] > 0 > lag["rs20"], (lead["rs20"], lag["rs20"])
    assert lead["rs60"] > 0 > lag["rs60"], (lead["rs60"], lag["rs60"])
    assert lead["st_above_sma"] is True and lag["st_above_sma"] is False
    assert lead["from_hi20"] is not None and lead["from_hi20"] > -0.01, \
        "a steadily rising line should sit at its own 20-day high"

    # 2. The note describes direction honestly, in both directions.
    assert "lebih kuat dari IHSG" in lead["st_note"], lead["st_note"]
    assert "lebih lemah dari IHSG" in lag["st_note"], lag["st_note"]

    # 3. It must NOT read as advice. This is the point of the whole file.
    #    Scan the descriptive clause only — the trailing disclaimer is allowed to
    #    say "bukan sinyal beli", which is the opposite of a recommendation.
    for card, label in ((lead, "leader"), (lag, "laggard")):
        low = card["st_note"].lower()
        assert "bukan sinyal beli" in low, f"{label} note must disclaim: {card['st_note']}"
        described = low.split("ini catatan pergerakan")[0]
        for w in _ADVICE_WORDS:
            assert w not in described, \
                f"{label} note sounds like advice ({w!r}): {card['st_note']}"

    # 4. No entry/stop/score/ranking fields may reach the view layer — each one
    #    would reintroduce the recommendation the backtest rejected.
    for banned in ("stop_pct", "stop_price", "st_score", "is_opportunity",
                   "st_view", "st_reason"):
        assert banned not in lead, f"{banned} must not be exposed to the dashboard"

    # 5. The retired selectors must stay gone, not merely unused.
    for gone in ("opportunities", "top_relative_strength_line"):
        assert not hasattr(tactical, gone), f"tactical.{gone} should have been removed"

    # 6. Display strings are formatted for the template.
    tactical.add_display_fields(lead)
    assert lead["rs20_str"].endswith("%") and lead["rs20_str"].startswith("+"), lead["rs20_str"]
    assert "stop_pct_str" not in lead, "stop formatting must be gone too"

    # 7. Insufficient history degrades to a plain statement, not a crash.
    short = pd.Series(np.linspace(100, 110, 5), index=pd.bdate_range("2025-01-01", periods=5))
    tiny = _card(short, index_close.iloc[:5], min_adv_bn)
    assert tiny["st_note"], "must still say something with thin history"
    print(f"short history: {tiny['st_note']}")

    # 8. The scoring machinery survives for reproducing the backtest finding.
    hist = tactical.score_history(_ohlcv(leader), index_close, min_adv_bn)
    assert {"st_score", "is_opportunity", "stop_pct"} <= set(hist.columns), \
        "score_history must stay intact for idxquant/strategies/tactical_rs.py"

    print("PASS")


if __name__ == "__main__":
    main()
