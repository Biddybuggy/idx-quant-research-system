#!/usr/bin/env python3
"""Persist/restore the paper-portfolio state as small JSON files in state/.

GitHub Actions runners are stateless; prices are re-downloaded fresh every run,
so the ONLY state that must survive between runs is the paper portfolio:
positions, cash/meta, closed trades, the equity curve, and the latest signal
file. That's a few KB — committed back to the repo after each daily run.

  python scripts/state_sync.py export   # DB -> state/
  python scripts/state_sync.py import   # state/ -> DB (no-op if state/ absent)
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from idxquant.config import ROOT, load_config
from idxquant.data import db

STATE_DIR = ROOT / "state"
STATE_FILE = STATE_DIR / "paper_state.json"
SIGNALS_SRC = ROOT / "data" / "signals_latest.json"
SIGNALS_STATE = STATE_DIR / "signals_latest.json"


def export() -> None:
    con = db.connect(load_config().db_path)
    state = {
        "paper_meta": dict(con.execute("SELECT key, value FROM paper_meta")),
        "paper_positions": [
            dict(zip(("ticker", "shares", "entry_date", "entry_price", "entry_costs"), r))
            for r in con.execute(
                "SELECT ticker, shares, entry_date, entry_price, entry_costs FROM paper_positions")
        ],
        "trades": [
            dict(zip(("run_id", "ticker", "side", "entry_date", "entry_price", "exit_date",
                      "exit_price", "shares", "costs", "pnl", "return_pct",
                      "holding_days", "open_flag"), r))
            for r in con.execute(
                "SELECT run_id, ticker, side, entry_date, entry_price, exit_date, exit_price, "
                "shares, costs, pnl, return_pct, holding_days, open_flag "
                "FROM trades WHERE mode='paper' ORDER BY id")
        ],
        "equity_curve": [
            dict(zip(("run_id", "date", "equity", "cash", "exposure", "drawdown"), r))
            for r in con.execute(
                "SELECT run_id, date, equity, cash, exposure, drawdown "
                "FROM equity_curve WHERE mode='paper' ORDER BY date")
        ],
    }
    con.close()
    STATE_DIR.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=1))
    if SIGNALS_SRC.exists():
        shutil.copy(SIGNALS_SRC, SIGNALS_STATE)
    print(f"exported: {len(state['equity_curve'])} equity rows, "
          f"{len(state['trades'])} trades, {len(state['paper_positions'])} positions")


def import_() -> None:
    if not STATE_FILE.exists():
        print("no state/ to import — fresh start (first run initializes the portfolio)")
        return
    state = json.loads(STATE_FILE.read_text())
    con = db.connect(load_config().db_path)
    con.execute("DELETE FROM paper_meta")
    con.execute("DELETE FROM paper_positions")
    con.execute("DELETE FROM trades WHERE mode='paper'")
    con.execute("DELETE FROM equity_curve WHERE mode='paper'")
    con.executemany("INSERT INTO paper_meta (key, value) VALUES (?,?)",
                    list(state["paper_meta"].items()))
    con.executemany(
        "INSERT INTO paper_positions (ticker, shares, entry_date, entry_price, entry_costs) "
        "VALUES (:ticker,:shares,:entry_date,:entry_price,:entry_costs)",
        state["paper_positions"])
    con.executemany(
        "INSERT INTO trades (mode, run_id, ticker, side, entry_date, entry_price, exit_date, "
        "exit_price, shares, costs, pnl, return_pct, holding_days, open_flag) "
        "VALUES ('paper',:run_id,:ticker,:side,:entry_date,:entry_price,:exit_date,"
        ":exit_price,:shares,:costs,:pnl,:return_pct,:holding_days,:open_flag)",
        state["trades"])
    con.executemany(
        "INSERT OR REPLACE INTO equity_curve (mode, run_id, date, equity, cash, exposure, drawdown) "
        "VALUES ('paper',:run_id,:date,:equity,:cash,:exposure,:drawdown)",
        state["equity_curve"])
    con.commit()
    con.close()
    if SIGNALS_STATE.exists():
        SIGNALS_SRC.parent.mkdir(exist_ok=True)
        shutil.copy(SIGNALS_STATE, SIGNALS_SRC)
    print(f"imported: {len(state['equity_curve'])} equity rows, "
          f"{len(state['trades'])} trades, {len(state['paper_positions'])} positions")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("export", "import"):
        sys.exit(__doc__)
    export() if sys.argv[1] == "export" else import_()
