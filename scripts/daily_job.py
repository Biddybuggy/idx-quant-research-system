#!/usr/bin/env python3
"""The daily after-close job: ingest -> paper fills -> signal file -> notify.

Run manually:      python scripts/daily_job.py
In production:     scheduled at 17:45 WIB by the API process (see idxquant/api/app.py)
                   or by an external cron hitting this script.

Every stage is wrapped so a failure still produces a Telegram alert —
silence must mean 'broken', never 'nothing happened'.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from idxquant.config import ROOT, load_config
from idxquant.data import db
from idxquant.data.ingest import run_ingest
from idxquant.notify import telegram
from idxquant.paper.executor import process as paper_process
from idxquant.signals.generate import generate_signal_file
from idxquant.strategies.factory import make_strategy


def run() -> dict:
    cfg = load_config()
    stage = "ingest"
    try:
        run_ingest(cfg)

        stage = "paper"
        paper_summary = paper_process(cfg)

        stage = "signal"
        con = db.connect(cfg.db_path)
        prices = {t: db.load_prices(con, t) for t in cfg.tickers}
        index_close = db.load_prices(con, cfg.index_ticker)["Close"]
        strategy = make_strategy(cfg)
        payload = generate_signal_file(
            prices, index_close, strategy, cfg,
            expected_holding_days=30.0,
            out_path=ROOT / "data" / "signals_latest.json",
        )
        con.close()

        stage = "notify"
        telegram.send(telegram.compose_daily(payload, paper_summary))
        print("daily job OK:", paper_summary["processed_days"] or "no new days")
        return {"ok": True, "paper": paper_summary}
    except Exception as err:
        traceback.print_exc()
        telegram.send(telegram.compose_error(stage, err))
        return {"ok": False, "stage": stage, "error": str(err)}


if __name__ == "__main__":
    result = run()
    sys.exit(0 if result["ok"] else 1)
