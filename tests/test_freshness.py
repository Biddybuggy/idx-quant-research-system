"""Freshness banner: the page must say what it is EVERY day, and must not cry
stale over a healthy weekend.

The old rule counted calendar days (>5), so a Friday outage stayed silent until
the following Saturday. The rule now counts trading days, which is what makes a
tight threshold safe — Saturday is not "two days stale", it is the weekend.

Run:  .venv/bin/python tests/test_freshness.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from idxquant.api.app import STALE_AFTER_TRADING_DAYS, _freshness


def fresh(last: str, today: str) -> dict:
    return _freshness(pd.Timestamp(last), pd.Timestamp(today).date())


def main():
    # --- healthy: the weekend is not staleness -----------------------------
    # 2026-07-24 is a Friday; 25th Sat, 26th Sun, 27th Mon.
    for day, label in [("2026-07-24", "Fri"), ("2026-07-25", "Sat"),
                       ("2026-07-26", "Sun"), ("2026-07-27", "Mon")]:
        r = fresh("2026-07-24", day)
        assert not r["stale"], f"Friday's close must stay fresh on {label}: {r}"

    # The weekend states must actually SAY the market is closed, not imply a fault.
    for day in ("2026-07-25", "2026-07-26"):
        r = fresh("2026-07-24", day)
        assert "libur" not in r["fresh_id"], r      # holiday wording is for the stale case
        assert "tutup akhir pekan" in r["fresh_id"], r
        assert "weekend" in r["fresh_en"], r

    # Monday pre-close is one behind and must explain why, not alarm.
    mon = fresh("2026-07-24", "2026-07-27")
    assert mon["behind_days"] == 1 and not mon["stale"], mon
    assert "17.45" in mon["fresh_id"], mon

    # --- broken: caught within two trading days ----------------------------
    tue = fresh("2026-07-24", "2026-07-28")          # Friday close, still nothing by Tuesday
    assert tue["stale"] and tue["behind_days"] == 2, tue
    mon_gap = fresh("2026-07-23", "2026-07-27")      # Friday's close never arrived
    assert mon_gap["stale"], mon_gap
    week = fresh("2026-07-17", "2026-07-27")
    assert week["stale"] and week["behind_days"] == 6, week

    # The stale copy must not assert a cause it cannot know (holiday vs outage).
    assert "libur" in tue["fresh_id"] and "tersendat" in tue["fresh_id"], tue
    assert "holiday" in tue["fresh_en"] and "stuck" in tue["fresh_en"], tue

    # --- every day says something ------------------------------------------
    for offset in range(0, 15):
        day = (pd.Timestamp("2026-07-24") + pd.Timedelta(days=offset)).date()
        r = _freshness(pd.Timestamp("2026-07-24"), day)
        assert r["fresh_id"] and r["fresh_en"], f"no message for {day}: {r}"
        assert str(r["as_of"]) == "2026-07-24", r

    # The old calendar-day rule would have missed this; guard the intent.
    assert STALE_AFTER_TRADING_DAYS == 2
    assert not fresh("2026-07-24", "2026-07-26")["stale"], \
        "a calendar-day rule would flag Sunday — trading days must not"

    print(f"PASS (stale after {STALE_AFTER_TRADING_DAYS} trading days)")


if __name__ == "__main__":
    main()
