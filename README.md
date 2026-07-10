# IDX Quant Research System

Research-grade quantitative research and **paper-trading** system for the
Indonesia Stock Exchange (IDX), with a mobile dashboard designed for iPhone.

**This is a research tool. It does not promise profits and is not investment
advice. The paper portfolio uses imaginary money.**

- Full design, roadmap, failure modes, promotion checklists: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Free deployment (GitHub Actions + Pages) + iPhone setup: [docs/DEPLOY_GITHUB.md](docs/DEPLOY_GITHUB.md)**
- Paid always-on server alternative (Fly.io, adds password protection): [docs/DEPLOY.md](docs/DEPLOY.md)
- All assumptions (universe, costs, lot size, risk limits, strategy params): [config/settings.yaml](config/settings.yaml)

## Quick start (research)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/run_pipeline.py ingest       # ~16y daily data, 21 liquid names + JCI
python scripts/run_pipeline.py backtest     # active strategy, full metrics vs benchmarks
python scripts/run_pipeline.py robustness   # parameter grid + cost sensitivity
python scripts/run_pipeline.py signal       # write data/signals_latest.json
python tests/test_engine_sanity.py          # engine reproduces buy-and-hold
```

## Quick start (product)

```bash
python scripts/run_pipeline.py paper --backfill-days 126   # init paper portfolio
python scripts/daily_job.py                                # ingest→paper→signal→notify
DISABLE_SCHEDULER=1 .venv/bin/python -m uvicorn idxquant.api.app:app --port 8642
# open http://localhost:8642  (set DASHBOARD_KEY env to enable auth)
```

In production the FastAPI process runs the daily job itself at 17:45 WIB
(APScheduler); `DISABLE_SCHEDULER=1` opts out for local dev.

## Architecture in one line

yfinance → SQLite → strategy (signals at close t) → engine `step()` (fills at
open t+1, IDX costs/lots/liquidity, drawdown halt) → shared by backtester and
paper executor → FastAPI read-only API + Bahasa-Indonesia-first PWA dashboard →
Telegram daily summary.

## Honesty guarantees built into the engine

- The signal→fill lag lives in exactly one place (`build_frames`), not in
  strategy code. Backtest and paper trading share the same `step()` — no drift.
- Buy/sell commissions (sell includes 0.1% IDX sales tax), half-spread,
  slippage, 100-share lots, ADV liquidity caps, long-only, no leverage.
- Portfolio drawdown halt: breach liquidates at next open + cooloff.
- Simulated warm-up history is labeled as simulation in the DB and on the dashboard.
- Every backtest run is stored with its config hash for reproducibility.
- The deployed API is read-only: there is no code path that can place real orders.

## Current research status (be honest with yourself)

No strategy tested so far beats buy-and-hold BBCA on risk-adjusted terms net of
realistic IDX costs. The paper portfolio runs the best current candidate
(cross-sectional 12-1 momentum, monthly, regime-filtered) as a *research
candidate*, not a validated edge. See the promotion checklists in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) §8–9 before ever risking money.
