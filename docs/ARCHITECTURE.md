# IDX Quant Research & Paper-Trading System — Architecture

> Research-grade system for the Indonesia Stock Exchange (IDX). No profit promises.
> The goal is statistically testable edges, honest backtests, and disciplined promotion
> from research → paper trading → (only if validated) live trading.

---

## 1. Architecture diagram

```
┌─────────────────────────────  iPhone (interface only)  ─────────────────────────────┐
│  PWA dashboard (Next.js/Streamlit, added Phase 4)      Telegram bot notifications   │
│  - portfolio & equity curve      - signal cards        - EOD signal alerts          │
│  - risk metrics                  - trade log           - risk-halt alerts           │
└───────────────▲──────────────────────────────────────────────────▲──────────────────┘
                │ HTTPS (read-only JSON)                            │ Telegram Bot API
┌───────────────┴───────────────────────────────────────────────────┴──────────────────┐
│                          API LAYER — FastAPI (Phase 4+)                              │
│  GET /portfolio  /signals/latest  /trades  /metrics  /equity  /health                │
│  Auth: single API token initially. Read-only. No order endpoints until Phase 6.      │
└───────────────▲───────────────────────────────────────────────────────────────────────┘
                │ reads
┌───────────────┴───────────────────────────────────────────────────────────────────────┐
│                     DATABASE — SQLite (→ PostgreSQL when needed)                      │
│  prices | features | signals | trades | equity_curve | backtest_runs | data_quality   │
└───────────────▲───────────────────────────────────────────────────────────────────────┘
                │ writes
┌───────────────┴───────────────────────────────────────────────────────────────────────┐
│                        QUANT ENGINE — Python (this repo)                              │
│                                                                                       │
│  data/ingest ──► validation ──► features/indicators ──► strategies ──► signals        │
│                                        │                                  │           │
│                                        ▼                                  ▼           │
│                              backtest engine (costs, lots,        paper-trading       │
│                              liquidity, no-lookahead)             executor (Phase 5)   │
│                                        │                                               │
│                                        ▼                                               │
│                              metrics + robustness reports                              │
└───────────────▲───────────────────────────────────────────────────────────────────────┘
                │ triggered by
┌───────────────┴───────────────────────────────────────────────────────────────────────┐
│  SCHEDULER — cron / APScheduler / GitHub Actions                                      │
│  ~17:30 WIB (after IDX close + data availability): ingest → validate → signals →      │
│  paper-trade fills → notify. Weekly: full re-backtest + drift report.                 │
└────────────────────────────────────────────────────────────────────────────────────────┘

External: Yahoo Finance (daily EOD, free, .JK tickers) → later: paid EOD vendor /
IDX-licensed data / broker API (Phase 6). Never rely on ToS-violating scraping.
```

**Key principle:** the iPhone never computes anything. It reads JSON and receives pushes.

---

## 2. Development roadmap

### Phase 1 — Data pipeline (week 1–2) ✅ in MVP
- Ingest daily OHLCV for watchlist + JCI (`^JKSE`) via yfinance into SQLite.
- Validation: missing dates vs index calendar, zero/negative prices, moves beyond
  auto-rejection bands (flag, don't silently fix), stale series detection.
- Store adjusted prices (total-return basis). Document the caveat: paper/live trading
  must use *raw* traded prices for order sizing; adjusted for research returns.
- Exit criteria: ≥10 years of clean data for the 5 core names + JCI; data-quality report is empty or explained.

### Phase 2 — Backtester (week 2–4) ✅ in MVP
- Custom daily-bar engine: signal at close *t* → execution at open *t+1*.
- IDX realism: 100-share lots, buy/sell commissions (sell includes 0.1% sales tax),
  half-spread + slippage, ADV participation cap, long-only, no leverage.
- Full metrics suite + trade log. Benchmarks: buy-and-hold BBCA and JCI.
- Exit criteria: a "buy day 1, hold forever" strategy through the engine reproduces
  buy-and-hold returns minus costs (engine sanity check).

### Phase 3 — Baseline strategies (week 4–8) ✅ one in MVP
- SMA crossover (+ JCI regime filter) — in the MVP.
- Then: 3/6/12-month momentum rank, 52-week breakout, RSI mean reversion,
  Bollinger z-score, bank-pair cointegration (BBCA/BBRI/BMRI) — long-only legs only
  unless shorting is confirmed available.
- Robustness harness: parameter grids, sub-period tests (incl. 2015, 2018, COVID-2020,
  2022), cost sensitivity (0.5×/1×/2×), per-ticker breakdown.
- Exit criteria: each surviving strategy has a one-page "strategy card" — hypothesis,
  parameter surface (must be a plateau, not a spike), sub-period table, cost sensitivity.

### Phase 4 — Dashboard + API (week 8–10)
- FastAPI read-only API over the SQLite DB.
- Streamlit first (fastest to iPhone-usable), Next.js PWA if you outgrow it.
- Telegram bot for EOD alerts (cheapest reliable push to iPhone; APNs later).
- Exit criteria: you can check portfolio, signals, and risk from Safari on the phone.

### Phase 5 — Paper trading (week 10 → +6 months minimum)
- Executor consumes the daily signal file, simulates fills at next open with the same
  cost model, writes to `trades`/`equity_curve` with `mode='paper'`.
- Nightly reconciliation: paper equity vs what the backtest engine would have said —
  divergence is a bug or a cost-model error; investigate every one.
- Exit criteria: see checklist §8/§9.

### Phase 6 — Optional broker integration (only after Phase 5 passes)
- Indonesian brokers with APIs are limited; realistic options: broker OpenAPI programs
  (e.g., some local brokers offer REST APIs to approved clients) or manual execution
  of system signals with strict logging. Start with the latter — it's honest and safe.
- Hard rules: order-size caps in code, kill switch, no market orders in thin books,
  human confirmation per order for the first months.

---

## 3. Project folder structure

```
idx_trading/
├── README.md
├── requirements.txt
├── config/
│   └── settings.yaml            # universe, costs, risk, strategy params — no code edits to retune
├── docs/
│   └── ARCHITECTURE.md          # this file
├── data/                        # SQLite DB + signal files (gitignored)
├── idxquant/
│   ├── __init__.py
│   ├── config.py                # typed config loader
│   ├── data/
│   │   ├── db.py                # schema + SQLite access
│   │   └── ingest.py            # yfinance download + validation
│   ├── features/
│   │   └── indicators.py        # SMA, RSI, Bollinger, ATR, momentum, drawdown, realized vol
│   ├── strategies/
│   │   ├── base.py              # Strategy interface: prices → signal frame
│   │   └── sma_crossover.py     # first baseline (+ regime filter)
│   ├── backtest/
│   │   ├── engine.py            # daily-bar simulator (costs, lots, liquidity, halt)
│   │   └── metrics.py           # CAGR, Sharpe, Sortino, DD, trade stats, benchmarks
│   ├── signals/
│   │   └── generate.py          # EOD signal file (JSON) with reasoning/invalidation/risk
│   ├── paper/                   # Phase 5: executor.py, reconcile.py
│   ├── api/                     # Phase 4: FastAPI app
│   └── notify/                  # Phase 4: telegram.py
├── scripts/
│   └── run_pipeline.py          # CLI: ingest | backtest | robustness | signal
└── tests/                       # start with engine sanity tests
```

---

## 4. Database schema (SQLite; PostgreSQL-compatible types)

```sql
prices(ticker TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL,
       volume REAL, source TEXT, ingested_at TEXT,
       PRIMARY KEY(ticker, date));

data_quality(ticker TEXT, date TEXT, issue TEXT, detail TEXT, created_at TEXT);

features(ticker TEXT, date TEXT, name TEXT, value REAL,
         PRIMARY KEY(ticker, date, name));          -- optional cache; recompute is cheap

signals(id INTEGER PK, date TEXT, ticker TEXT, strategy TEXT, action TEXT,
        confidence TEXT, reasoning TEXT, expected_holding_days REAL,
        invalidation TEXT, suggested_weight REAL, risk_json TEXT, created_at TEXT);

trades(id INTEGER PK, mode TEXT,                     -- 'backtest' | 'paper' | 'live'
       run_id TEXT, ticker TEXT, side TEXT,
       entry_date TEXT, entry_price REAL, exit_date TEXT, exit_price REAL,
       shares INTEGER, costs REAL, pnl REAL, return_pct REAL,
       holding_days INTEGER, open_flag INTEGER);

equity_curve(mode TEXT, run_id TEXT, date TEXT, equity REAL, cash REAL,
             exposure REAL, drawdown REAL, PRIMARY KEY(mode, run_id, date));

backtest_runs(run_id TEXT PK, strategy TEXT, params_json TEXT, start TEXT, "end" TEXT,
              metrics_json TEXT, config_hash TEXT, code_version TEXT, created_at TEXT);
```

`config_hash` + `code_version` on every run: you must always be able to reproduce a
backtest that made you excited.

---

## 5. IDX market realities baked into the design

| Reality | Handling |
|---|---|
| Round lots of 100 shares | All order sizes floor to lots |
| Sell-side sales tax 0.1% | Sell commission = broker fee + 0.1% |
| Broker commissions ~0.15%/0.25% | Configurable in `settings.yaml` |
| Auto-rejection bands (ARA/ARB) | Validation flags daily moves beyond bands; a gap-limit day means your fill may not exist — engine notes this as a limitation of daily data |
| No general retail short-selling | Engine is long-only; pairs strategies use long-leg-only or spread-timing variants |
| T+2 settlement | Cash from sells reusable immediately in-model; fine at daily frequency, revisit at Phase 6 |
| Liquidity concentration | ADV participation cap; liquidity filter (min 20-day average traded value) |
| Foreign-flow sensitivity | JCI regime filter now; foreign net-flow data as a Phase 3+ feature if a licensed source is found |
| Free float / governance | Watchlist restricted to liquid large caps; any watchlist addition needs a liquidity check |
| Tick-size tiers | Ignored at daily frequency (inside half-spread assumption); must be modeled if intraday ever added |

---

## 6. Mobile dashboard plan (iPhone)

**Phase 4a — Streamlit (days of work):** single-column layout, mobile-friendly by
default. Pages: *Today* (signal cards: action, confidence, reasoning, invalidation),
*Portfolio* (equity curve, exposure, drawdown, positions), *Risk* (rolling vol/Sharpe,
DD-halt state), *History* (trade log). Deploy on Fly.io/Railway behind basic auth.
Add to Home Screen from Safari → app-like icon.

**Phase 4b — Next.js PWA (if Streamlit feels clunky):** static frontend hitting the
FastAPI JSON endpoints; installable PWA with service worker; web-push works on iOS 16.4+
for installed PWAs. Only build this once the data is worth looking at.

**Notifications:** Telegram bot first — free, reliable, 30 lines of code, great on
iPhone. One EOD message: signals, portfolio delta, any data-quality or risk-halt alerts.
Silence is a failure mode: the bot should also send "pipeline ran, nothing to do."

---

## 7. Major failure modes and prevention

1. **Lookahead bias** — signal uses close *t*, fills at open *t+1*, enforced by a single
   `shift(1)` in the engine, not scattered through strategy code. Test: shuffle-forward test
   (feed the engine future-shifted signals; performance should jump — proving the guard matters).
2. **Overfitting / p-hacking** — parameter plateaus required; sub-period + per-ticker
   robustness mandatory; count every variant you tried (the real trial count includes
   discarded ones); hold out the most recent ~2 years untouched until a strategy is final.
3. **Survivorship bias** — current watchlist is survivors. Mitigate: strategies must also
   be tested on a broader historical liquid-IDX list including delisted/faded names when
   data allows; until then, treat absolute return estimates as optimistic.
4. **Bad data silently poisoning signals** — validation runs *every* ingest; anomalies go
   to `data_quality` and to Telegram; the pipeline refuses to emit signals from a series
   with an unexplained anomaly in the last 30 days.
5. **Cost-model optimism** — every strategy must survive 2× assumed costs. High-turnover
   strategies on IDX retail commissions usually die here; that's the test working.
6. **Adjusted vs raw price confusion** — research uses adjusted (total return); order
   sizing in paper/live must use raw prices. Kept in separate code paths, documented.
7. **Regime death** — a strategy validated in a bull decade fails in a bear market.
   Regime filter + max-DD halt are structural, not optional; sub-period tests must include 2015/2018/2020/2022.
8. **Silent scheduler failure** — heartbeat message even when nothing happens; missing
   heartbeat = investigate.
9. **Yahoo data quirks (.JK)** — occasional missing days, unadjusted corporate actions,
   revised bars. Re-download trailing 10 sessions on every ingest; diff against stored bars; alert on revisions.
10. **Risk creep** — position caps, no-leverage, and DD-halt live in config + engine, not
    in your discipline. Changing them requires a config edit with a git commit message.
11. **ARA/ARB gap risk** — daily bars hide limit-locked days; a "fill at open" on an
    ARA day may be fantasy. Validation flags such days; trades on flagged days get audited.

---

## 8. Checklist: backtest → paper trading

- [ ] ≥10 years of validated data; unexplained `data_quality` rows = 0.
- [ ] Engine sanity: buy-and-hold through the engine ≈ buy-and-hold arithmetic minus costs.
- [ ] Strategy beats both benchmarks on **risk-adjusted** terms (Sharpe/Sortino/MaxDD) net of costs — or has a clearly articulated reason to exist (e.g., lower DD with similar CAGR).
- [ ] Parameter plateau: ≥60% of a sensible parameter grid remains acceptable.
- [ ] Survives: 2× costs, each sub-period (incl. crash windows), leave-one-ticker-out.
- [ ] Trade count ≥ ~80–100 (else statistics are noise).
- [ ] Held-out recent period tested exactly once, after everything else was frozen.
- [ ] Strategy card written: hypothesis, why the edge should exist, when it should fail.
- [ ] Run is reproducible from `run_id` (config hash + code version).
- [ ] Signal file includes confidence, reasoning, holding period, invalidation, risk — every field populated, no placeholders.

## 9. Checklist: paper trading → live trading

- [ ] ≥6 months (prefer 12) of uninterrupted paper trading, ≥30 completed trades.
- [ ] Paper results within backtest expectation bands (rolling Sharpe/DD inside the
      distribution implied by bootstrapped backtest trades) — no "it's different live but fine."
- [ ] Zero unexplained reconciliation breaks between paper executor and engine.
- [ ] Slippage estimate validated: compare assumed fill vs actual open prices on signal days.
- [ ] Ops: scheduler uptime ≥99% over the paper period; alerting tested (kill the job, confirm you get paged).
- [ ] Risk framework rehearsed: DD-halt has actually triggered in paper or in a fire drill, and you followed it.
- [ ] Broker path tested with 1-lot orders; costs measured, not assumed.
- [ ] Written risk budget: max capital, max loss before full stop (an absolute IDR number), who decides to stop (you, in writing, in advance).
- [ ] You can afford to lose the entire allocated capital without life impact.
- [ ] Regulatory sanity: personal account, own capital, no advice to others, taxes understood.
```
