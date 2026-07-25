"""The action on a signal must describe what the portfolio will actually do.

Regression test. The action used to come from the strategy's own signal
transition (yesterday's 0/1 vs today's), which is only the same thing as the
portfolio's next move while one strategy has been running from the start. When
the configured strategy was switched to `regime_switch` on 2026-07-25, its
reversal leg had been signalling ASII and UNTR since March, so the signal file
said HOLD_LONG for names the paper portfolio had never bought — sitting directly
beside a panel that read "holding nothing".

Run:  .venv/bin/python tests/test_signal_actions.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from idxquant.config import load_config
from idxquant.signals.generate import generate_signal_file


class FakeStrategy:
    """A, B signalled today and yesterday. C signalled YESTERDAY ONLY (so the
    legacy transition path sees a 1->0 exit). D never signalled."""
    name = "fake"
    regime_sma = 200

    def __init__(self, index):
        self.index = index

    def signals(self, prices, index_close):
        s = pd.DataFrame(0, index=self.index, columns=["A.JK", "B.JK", "C.JK", "D.JK"])
        s.loc[:, ["A.JK", "B.JK"]] = 1
        s.iloc[:-1, s.columns.get_loc("C.JK")] = 1     # on until today
        return s

    def signal_context(self, ticker, prices, index_close):
        return {"confidence": "medium", "reasoning": "r", "invalidation": "i",
                "extra_risk": {}}


def _prices(index):
    out = {}
    for i, t in enumerate(["A.JK", "B.JK", "C.JK", "D.JK"]):
        close = pd.Series(np.linspace(1000, 1100 + i, len(index)), index=index)
        out[t] = pd.DataFrame({"Open": close, "High": close * 1.01,
                               "Low": close * 0.99, "Close": close, "Volume": 1e8})
    return out


def _actions(held):
    idx = pd.bdate_range("2024-01-01", periods=260)
    prices = _prices(idx)
    index_close = pd.Series(np.linspace(1000, 1200, len(idx)), index=idx)
    with tempfile.TemporaryDirectory() as d:
        p = generate_signal_file(prices, index_close, FakeStrategy(idx),
                                 load_config(), 30.0, Path(d) / "s.json",
                                 held_tickers=held)
    return {e["ticker"]: e["action"] for e in p["signals"]}


def main():
    # --- 1. nothing held: every signalled name is an ENTRY, never a HOLD -----
    a = _actions(held=set())
    print("1. held={}        ->", a)
    assert a["A.JK"] == "ENTER_LONG", a
    assert a["B.JK"] == "ENTER_LONG", a
    assert a["C.JK"] == "NO_POSITION", a
    assert a["D.JK"] == "NO_POSITION", a
    assert "HOLD_LONG" not in a.values(), \
        "said HOLD for a name the portfolio does not own — the exact 2026-07-25 bug"

    # --- 2. already holding a signalled name -> HOLD ------------------------
    a = _actions(held={"A.JK"})
    print("2. held={A}       ->", a)
    assert a["A.JK"] == "HOLD_LONG" and a["B.JK"] == "ENTER_LONG", a

    # --- 3. holding something no longer signalled -> EXIT -------------------
    a = _actions(held={"A.JK", "C.JK"})
    print("3. held={A,C}     ->", a)
    assert a["C.JK"] == "EXIT", a
    assert a["D.JK"] == "NO_POSITION", a

    # --- 4. omitting held_tickers keeps the old transition behaviour --------
    idx = pd.bdate_range("2024-01-01", periods=260)
    prices = _prices(idx)
    index_close = pd.Series(np.linspace(1000, 1200, len(idx)), index=idx)
    with tempfile.TemporaryDirectory() as d:
        p = generate_signal_file(prices, index_close, FakeStrategy(idx),
                                 load_config(), 30.0, Path(d) / "s.json")
    legacy = {e["ticker"]: e["action"] for e in p["signals"]}
    print("4. no held arg    ->", legacy)
    assert legacy["A.JK"] == "HOLD_LONG", "fallback path changed unexpectedly"
    assert legacy["C.JK"] == "EXIT", legacy

    print("PASS")


if __name__ == "__main__":
    main()
