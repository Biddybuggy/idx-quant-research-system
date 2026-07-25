"""Tactical layer sanity: a stock beating a falling market with a clean short-term
structure must be flagged an opportunity; a stock sinking with the market must not.

This is the "leaders in a weak market" claim the tactical layer exists to make —
so it is asserted on constructed data, independent of the DB.

Run:  .venv/bin/python tests/test_tactical.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from idxquant.features import indicators as ind
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
    df = _ohlcv(close, vol)
    c = float(close.iloc[-1])
    rsi = float(ind.rsi(close).iloc[-1])
    atr_pct = float(ind.atr(df).iloc[-1] / c)
    adv_bn = float(ind.avg_daily_value(df).iloc[-1]) / 1e9
    return tactical.compute(close, index_close, df, rsi=rsi, atr_pct=atr_pct,
                            adv_bn=adv_bn, min_adv_bn=min_adv_bn, flagged=flagged)


def main():
    n = 150
    dates = pd.bdate_range("2025-01-01", periods=n)

    # A weak overall market: the index grinds down ~20%.
    index_close = pd.Series(np.linspace(1000, 800, n), index=dates)

    # LEADER: rises steadily while the market falls -> strong relative strength,
    # above a rising 20d SMA, pressing new 20d highs. Should be an opportunity.
    leader = pd.Series(np.linspace(1000, 1300, n), index=dates)

    # LAGGARD: falls faster than the market -> negative relative strength. Not one.
    laggard = pd.Series(np.linspace(1000, 700, n), index=dates)

    min_adv_bn = 10.0  # matches the config liquidity floor (IDR 10bn)
    # Both liquid (higher volume on the lower-priced laggard keeps its ADV above
    # the floor) so the laggard is rejected for weakness, not for liquidity.
    lead = _card(leader, index_close, min_adv_bn, vol=1e7)
    lag = _card(laggard, index_close, min_adv_bn, vol=2e7)

    print(f"leader : score {lead['st_score']:5}  rs20 {lead['rs20']:+.3f}  "
          f"view {lead['st_view']}  opp={lead['is_opportunity']}")
    print(f"laggard: score {lag['st_score']:5}  rs20 {lag['rs20']:+.3f}  "
          f"view {lag['st_view']}  opp={lag['is_opportunity']}")

    # 1. The leader is an opportunity; the laggard is not.
    assert lead["is_opportunity"] is True, "rising leader in a falling market must qualify"
    assert lead["st_view"] == "peluang jangka pendek"
    assert lag["is_opportunity"] is False, "a name sinking with the market must not qualify"
    assert lag["st_view"] == "lemah jangka pendek"
    assert lead["st_score"] > lag["st_score"]

    # 2. Relative strength has the right sign; the leader presses its 20d high.
    assert lead["rs20"] > 0 > lag["rs20"]
    assert lead["from_hi20"] is not None and lead["from_hi20"] > -0.01

    # 3. A suggested stop exists and sits below entry.
    assert lead["stop_pct"] is not None and lead["stop_pct"] < 0
    assert lead["stop_price"] is not None and lead["stop_price"] < float(leader.iloc[-1])

    # 4. The liquidity gate bites: the same leader on thin volume is not an opportunity.
    thin = _card(leader, index_close, min_adv_bn, vol=1e5)
    assert thin["is_opportunity"] is False, "thin-liquidity names must be filtered out"
    assert thin["st_view"] == "likuiditas tipis"

    # 5. Selectors: opportunities() keeps only qualifying names; the RS line names the leader.
    cards = [dict(lead, name="LEAD"), dict(lag, name="LAG"), dict(thin, name="THIN")]
    opps = tactical.opportunities(cards)
    assert [o["name"] for o in opps] == ["LEAD"], f"unexpected opportunities: {opps}"
    line = tactical.top_relative_strength_line(cards)
    assert "LEAD" in line and "LAG" not in line and "THIN" not in line, line

    # 6. The Telegram line and the dashboard panel must never disagree — a name
    #    in the message has to be findable on the site.
    assert all(o["name"] in line for o in opps), (line, opps)
    assert tactical.top_relative_strength_line([dict(lag, name="LAG")]) == "", \
        "no qualifying names must produce no line, not a misleading one"

    print("PASS")


if __name__ == "__main__":
    main()
