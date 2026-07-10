#!/usr/bin/env python3
"""CLI for the IDX quant pipeline.

  python scripts/run_pipeline.py ingest        # download + validate + store prices
  python scripts/run_pipeline.py backtest      # run configured strategy, print metrics
  python scripts/run_pipeline.py robustness    # parameter grid + cost sensitivity
  python scripts/run_pipeline.py signal        # write data/signals_latest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from idxquant.backtest import metrics as m
from idxquant.backtest.engine import run_backtest
from idxquant.config import ROOT, load_config
from idxquant.data import db
from idxquant.data.ingest import run_ingest
from idxquant.paper.executor import process as paper_process
from idxquant.signals.generate import generate_signal_file
from idxquant.strategies.factory import make_strategy


def _load_all(cfg):
    con = db.connect(cfg.db_path)
    prices = {t: db.load_prices(con, t) for t in cfg.tickers}
    index_close = db.load_prices(con, cfg.index_ticker)["Close"]
    for t, df in prices.items():
        if df.empty:
            sys.exit(f"No stored data for {t} — run `ingest` first.")
    return con, prices, index_close


_make_strategy = make_strategy


def cmd_ingest(cfg):
    counts = run_ingest(cfg)
    for t, n in counts.items():
        print(f"  {t:<10} {n:>6} rows upserted")
    con = db.connect(cfg.db_path)
    issues = pd.read_sql(
        "SELECT ticker, issue, COUNT(*) n FROM data_quality GROUP BY ticker, issue", con)
    print("\nData-quality flags (review docs/ARCHITECTURE.md §7 before trusting signals):")
    print(issues.to_string(index=False) if len(issues) else "  none")


def cmd_backtest(cfg, quiet=False, cost_mult=1.0, **overrides):
    con, prices, index_close = _load_all(cfg)
    cfg.costs.buy_commission *= cost_mult
    cfg.costs.sell_commission *= cost_mult
    cfg.costs.half_spread *= cost_mult
    cfg.costs.slippage *= cost_mult
    strat = _make_strategy(cfg, **overrides)
    signals = strat.signals(prices, index_close)
    result = run_backtest(prices, signals, cfg)
    summary = m.summarize(result, {
        "BBCA": prices[cfg.benchmark_stock]["Close"],
        "JCI": index_close,
    })
    if not quiet:
        run_id = uuid.uuid4().hex[:12]
        cfg_hash = hashlib.sha256(json.dumps(cfg.raw, sort_keys=True).encode()).hexdigest()[:12]
        con.execute(
            "INSERT INTO backtest_runs (run_id, strategy, params_json, start, end, metrics_json, config_hash, code_version) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (run_id, strat.name, json.dumps(cfg.strategy), summary["start"],
             summary["end"], json.dumps(summary), cfg_hash, "mvp-0.1"))
        con.commit()
        print(f"=== {strat.name}  (run {run_id}) ===\n")
        _print_summary(summary)
        print("\nWorst trades:")
        wt = m.worst_trades(result.trades)
        print(wt.to_string(index=False) if len(wt) else "  none")
    return summary, result


def _print_summary(s):
    fmt_pct = lambda v: f"{v:+.1%}" if v == v else "n/a"
    print(f"  Period                {s['start']} → {s['end']}")
    print(f"  Final equity          IDR {s['final_equity_idr']:,.0f}")
    print(f"  CAGR                  {fmt_pct(s['cagr'])}   (B&H BBCA {fmt_pct(s.get('bh_BBCA_cagr', float('nan')))}, JCI {fmt_pct(s.get('bh_JCI_cagr', float('nan')))})")
    print(f"  Sharpe                {s['sharpe']:.2f}    (B&H BBCA {s.get('bh_BBCA_sharpe', float('nan')):.2f}, JCI {s.get('bh_JCI_sharpe', float('nan')):.2f})")
    print(f"  Sortino               {s['sortino']:.2f}")
    print(f"  Max drawdown          {fmt_pct(s['max_drawdown'])}  (B&H BBCA {fmt_pct(s.get('bh_BBCA_max_drawdown', float('nan')))}, JCI {fmt_pct(s.get('bh_JCI_max_drawdown', float('nan')))})")
    print(f"  Avg exposure          {s['avg_exposure']:.0%}")
    print(f"  Turnover/yr           {s['turnover_per_year']:.2f}x")
    print(f"  Trades (open)         {s['n_trades']} ({s.get('n_open', 0)})")
    print(f"  Win rate              {s.get('win_rate', float('nan')):.0%}")
    print(f"  Profit factor         {s.get('profit_factor', float('nan')):.2f}")
    print(f"  Avg / median holding  {s.get('avg_holding_days', 0):.0f} / {s.get('median_holding_days', 0):.0f} days")
    print(f"  Total costs           IDR {s.get('total_costs_idr', 0):,.0f}")


def cmd_robustness(cfg):
    print("Parameter grid (CAGR / Sharpe / MaxDD / trades) — look for a plateau, not a spike:\n")
    rows = []
    if cfg.strategy["name"] == "momentum":
        for lb in [126, 189, 252, 378]:
            for top_n in [1, 2, 3]:
                summary, _ = cmd_backtest(load_config(), quiet=True, lookback=lb, top_n=top_n)
                rows.append({"lookback": lb, "top_n": top_n, "cagr": summary["cagr"],
                             "sharpe": summary["sharpe"], "max_dd": summary["max_drawdown"],
                             "trades": summary["n_trades"]})
    else:
        for f in [10, 20, 30, 50]:
            for s in [60, 100, 150, 200]:
                if f >= s:
                    continue
                summary, _ = cmd_backtest(load_config(), quiet=True, fast=f, slow=s)
                rows.append({"fast": f, "slow": s, "cagr": summary["cagr"],
                             "sharpe": summary["sharpe"], "max_dd": summary["max_drawdown"],
                             "trades": summary["n_trades"]})
    grid = pd.DataFrame(rows)
    print(grid.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\nCost sensitivity (must survive 2x costs):\n")
    for mult in [0.5, 1.0, 2.0]:
        summary, _ = cmd_backtest(load_config(), quiet=True, cost_mult=mult)
        print(f"  costs x{mult:<4} CAGR {summary['cagr']:+.1%}  Sharpe {summary['sharpe']:.2f}  MaxDD {summary['max_drawdown']:+.1%}")


def cmd_signal(cfg):
    con, prices, index_close = _load_all(cfg)
    strat = _make_strategy(cfg)
    # expected holding period comes from the backtest's own trade history
    summary, _ = cmd_backtest(load_config(), quiet=True)
    holding = summary.get("median_holding_days", 30.0)
    out = ROOT / "data" / "signals_latest.json"
    payload = generate_signal_file(prices, index_close, strat, cfg, holding, out)
    for e in payload["signals"]:
        con.execute(
            "INSERT INTO signals (date, ticker, strategy, action, confidence, reasoning, "
            "expected_holding_days, invalidation, suggested_weight, risk_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (payload["as_of_close"], e["ticker"], payload["strategy"], e["action"],
             e["confidence"], e["reasoning"], e["expected_holding_days"],
             e["invalidation"], e["risk"]["suggested_weight"], json.dumps(e["risk"])))
    con.commit()
    print(f"Signal file written: {out}\n")
    print(json.dumps(payload, indent=2))


def cmd_paper(cfg, backfill_days=0):
    summary = paper_process(cfg, backfill_days=backfill_days)
    print(json.dumps(summary, indent=2, default=str))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["ingest", "backtest", "robustness", "signal", "paper"])
    p.add_argument("--backfill-days", type=int, default=0,
                   help="paper only, first run only: replay the last N trading days "
                        "as a simulated warm-up (labeled as such in paper_meta)")
    args = p.parse_args()
    cfg = load_config()
    if args.command == "paper":
        cmd_paper(cfg, backfill_days=args.backfill_days)
    else:
        {"ingest": cmd_ingest, "backtest": cmd_backtest,
         "robustness": cmd_robustness, "signal": cmd_signal}[args.command](cfg)


if __name__ == "__main__":
    main()
