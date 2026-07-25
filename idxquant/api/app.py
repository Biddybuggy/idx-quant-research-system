"""Read-only API + mom-facing dashboard.

Security model: one shared secret (env DASHBOARD_KEY). The dashboard link
carries ?key=... once; a cookie keeps the session. Everything is read-only —
there is deliberately NO endpoint that places orders or changes state.
If DASHBOARD_KEY is unset (local dev), auth is disabled.

The daily job is scheduled in-process (17:45 WIB, after IDX close) so a single
deployed container is the whole backend. Set DISABLE_SCHEDULER=1 to opt out
(e.g., when an external cron runs scripts/daily_job.py instead).
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from ..config import ROOT, load_config
from ..data import db
from ..research.report import market_overview, stock_research
from ..research.riskoff import screen as riskoff_screen
from .chart import equity_chart_svg

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
STATIC = Path(__file__).parent / "static"
PUBLIC_PATHS = {"/api/health", "/manifest.webmanifest", "/icon-180.png", "/icon-512.png"}


def _key() -> str | None:
    return os.environ.get("DASHBOARD_KEY") or None


@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler = None
    if not os.environ.get("DISABLE_SCHEDULER"):
        from apscheduler.schedulers.background import BackgroundScheduler
        from scripts.daily_job import run as daily_run
        scheduler = BackgroundScheduler(timezone="Asia/Jakarta")
        scheduler.add_job(daily_run, "cron", hour=17, minute=45,
                          misfire_grace_time=3600)
        scheduler.start()
    yield
    if scheduler:
        scheduler.shutdown(wait=False)


app = FastAPI(title="IDX Paper Dashboard", lifespan=lifespan)


@app.middleware("http")
async def auth(request: Request, call_next):
    key = _key()
    if key and request.url.path not in PUBLIC_PATHS:
        supplied = request.query_params.get("key") or request.cookies.get("dk")
        if supplied != key:
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
        response = await call_next(request)
        if request.query_params.get("key") == key:
            response.set_cookie("dk", key, max_age=365 * 24 * 3600,
                                httponly=True, samesite="lax")
        return response
    return await call_next(request)


def _con() -> sqlite3.Connection:
    return db.connect(load_config().db_path)


def _meta(con) -> dict:
    return dict(con.execute("SELECT key, value FROM paper_meta").fetchall())


def _equity_df(con) -> pd.DataFrame:
    return pd.read_sql(
        "SELECT date, equity, cash, exposure, drawdown FROM equity_curve "
        "WHERE mode='paper' AND run_id='paper-live' ORDER BY date",
        con, parse_dates=["date"])


def _positions(con) -> list[dict]:
    rows = con.execute(
        "SELECT ticker, shares, entry_date, entry_price FROM paper_positions").fetchall()
    out = []
    for ticker, shares, entry_date, entry_price in rows:
        last = con.execute(
            "SELECT close FROM prices WHERE ticker=? ORDER BY date DESC LIMIT 1",
            (ticker,)).fetchone()
        last_close = last[0] if last else None
        out.append({
            "ticker": ticker, "shares": shares, "entry_date": entry_date,
            "entry_price": round(entry_price, 2), "last_close": last_close,
            "value_idr": round(shares * last_close, 2) if last_close else None,
            "pnl_pct": round(last_close / entry_price - 1, 4) if last_close else None,
        })
    return out


def _signal_payload() -> dict | None:
    p = ROOT / "data" / "signals_latest.json"
    return json.loads(p.read_text()) if p.exists() else None


# ---------- JSON API ----------

@app.get("/api/health")
def health():
    con = _con()
    meta = _meta(con)
    con.close()
    return {"status": "ok", "last_processed": meta.get("last_processed"),
            "strategy": meta.get("strategy")}


@app.get("/api/portfolio")
def portfolio():
    con = _con()
    eq = _equity_df(con)
    meta = _meta(con)
    positions = _positions(con)
    con.close()
    if eq.empty:
        return {"error": "paper portfolio not initialized"}
    return {
        "equity_idr": float(eq.equity.iloc[-1]),
        "cash_idr": float(eq.cash.iloc[-1]),
        "as_of": str(eq.date.iloc[-1].date()),
        "return_since_start": float(eq.equity.iloc[-1] / eq.equity.iloc[0] - 1),
        "drawdown": float(eq.drawdown.iloc[-1]),
        "halted": meta.get("halted") == "1",
        "started_at": meta.get("started_at"),
        "backfill_until": meta.get("backfill_until"),
        "positions": positions,
    }


@app.get("/api/equity")
def equity():
    con = _con()
    eq = _equity_df(con)
    con.close()
    return {"dates": [str(d.date()) for d in eq.date],
            "equity": [round(v, 2) for v in eq.equity]}


@app.get("/api/signals/latest")
def signals_latest():
    return _signal_payload() or {"error": "no signal file yet"}


@app.get("/api/trades")
def trades(limit: int = 20):
    con = _con()
    rows = pd.read_sql(
        "SELECT ticker, entry_date, exit_date, shares, return_pct, pnl "
        "FROM trades WHERE mode='paper' ORDER BY exit_date DESC LIMIT ?",
        con, params=(limit,))
    con.close()
    return rows.to_dict("records")


# ---------- PWA assets ----------

@app.get("/manifest.webmanifest")
def manifest(request: Request):
    start = "/"
    key = _key()
    # bake the key into start_url so the installed home-screen app authenticates
    # in its own storage container (iOS PWAs don't share Safari's cookies)
    if key and request.query_params.get("key") == key:
        start = f"/?key={key}"
    return JSONResponse({
        "name": "Dasbor Saham", "short_name": "Saham",
        "start_url": start, "display": "standalone",
        "background_color": "#f9f9f7", "theme_color": "#2a78d6",
        "icons": [{"src": "/icon-192.png", "sizes": "192x192", "type": "image/png"},
                  {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png"}],
    }, media_type="application/manifest+json")


@app.get("/icon-180.png")
def icon180():
    return FileResponse(STATIC / "icon-180.png")


@app.get("/icon-192.png")
def icon192():
    return FileResponse(STATIC / "icon-192.png")


@app.get("/icon-512.png")
def icon512():
    return FileResponse(STATIC / "icon-512.png")


# ---------- Dashboard ----------

ACTION_ID = {"ENTER_LONG": ("Beli (latihan)", "buy"),
             "HOLD_LONG": ("Tahan", "hold"),
             "EXIT": ("Jual (latihan)", "sell"),
             "NO_POSITION": ("Menunggu", "wait")}


_ID_MONTHS = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
              "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
_EN_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Trading days behind before the numbers are called stale. One day of slack is
# normal: today's close only lands after the daily job runs (~17:45 WIB), so a
# reader opening the page mid-morning is correctly one close behind.
STALE_AFTER_TRADING_DAYS = 2


def _freshness(last_date: pd.Timestamp, today=None) -> dict:
    """Always say what the reader is looking at — every day, not just bad ones.

    Staleness is counted in TRADING days, never calendar days: on a Saturday,
    Friday's close IS current, and a calendar-day counter would cry stale over
    a perfectly healthy weekend. That distinction is what lets the threshold be
    tight enough to catch a broken pipeline within a day of trading.
    """
    last = last_date.date()
    today = today or pd.Timestamp.now().date()   # injectable so the calendar cases are testable
    # Business days strictly after the last close, up to and including today.
    behind = len(pd.bdate_range(last + timedelta(days=1), today))
    d_id = f"{last.day} {_ID_MONTHS[last.month - 1]}"
    d_en = f"{last.day} {_EN_MONTHS[last.month - 1]}"

    if behind >= STALE_AFTER_TRADING_DAYS:
        # Deliberately names both causes. We do not track IDX holidays, so a
        # holiday and a broken pipeline look identical from here — claiming
        # "broken" would be a guess, and either way the numbers are old.
        return {"stale": True, "as_of": str(last), "behind_days": behind,
                "fresh_id": f"Belum ada data baru sejak penutupan {d_id} "
                            f"({behind} hari bursa). Mungkin bursa libur, atau "
                            f"pembaruan tersendat — angka di bawah mungkin lama.",
                "fresh_en": f"No new data since the {d_en} close ({behind} trading days). "
                            f"The market may be on holiday, or the update may be "
                            f"stuck — numbers below may be old."}
    if today.weekday() >= 5:
        msg_id = f"Bursa tutup akhir pekan — angka ini penutupan {d_id}."
        msg_en = f"Market closed for the weekend — these are the {d_en} close."
    elif behind == 0:
        msg_id = f"Terbaru — penutupan {d_id}."
        msg_en = f"Up to date — the {d_en} close."
    else:
        msg_id = (f"Penutupan terakhir {d_id}. Angka hari ini masuk "
                  f"setelah bursa tutup (±17.45 WIB).")
        msg_en = (f"Latest close {d_en}. Today's numbers arrive "
                  f"after the market closes (~17:45 WIB).")
    return {"stale": False, "as_of": str(last), "behind_days": behind,
            "fresh_id": msg_id, "fresh_en": msg_en}


def build_dashboard_context() -> dict | None:
    """All data the dashboard template needs. Shared by the live server and
    scripts/build_static.py (GitHub Pages) so both render the same page.
    Returns None if the paper portfolio isn't initialized."""
    cfg = load_config()
    con = _con()
    eq = _equity_df(con)
    meta = _meta(con)
    positions = _positions(con)
    recent = pd.read_sql(
        "SELECT ticker, entry_date, exit_date, return_pct FROM trades "
        "WHERE mode='paper' ORDER BY exit_date DESC LIMIT 5", con)
    prices = {t: db.load_prices(con, t) for t in cfg.tickers}
    index_close = db.load_prices(con, cfg.index_ticker)["Close"]
    flagged = {r[0] for r in con.execute(
        "SELECT DISTINCT ticker FROM data_quality "
        "WHERE issue IN ('ingest_failed','extreme_move') AND date >= date('now','-30 days')")}
    con.close()

    if eq.empty:
        return None

    held = {p["ticker"] for p in positions}
    research = stock_research(prices, index_close, cfg, held, flagged)
    market = market_overview(index_close, cfg)
    # None while the market is risk-on — the screen is only validated in risk-off,
    # so the panel is absent rather than empty for most of the year.
    riskoff = riskoff_screen(prices, index_close, cfg)

    payload = _signal_payload() or {"signals": [], "regime": "?", "as_of_close": "?"}
    active = [s for s in payload["signals"] if s["action"] != "NO_POSITION"]
    n_waiting = len(payload["signals"]) - len(active)
    for s in active:
        s["label_id"], s["css"] = ACTION_ID[s["action"]]

    equity_now = float(eq.equity.iloc[-1])
    ret_start = equity_now / float(eq.equity.iloc[0]) - 1
    ret_day = (equity_now / float(eq.equity.iloc[-2]) - 1) if len(eq) > 1 else 0.0
    risk_off = payload.get("regime", "").startswith("risk-off")
    halted = meta.get("halted") == "1"

    if halted:
        status_id = "Sistem berhenti sementara — pelindung risiko aktif."
        status_en = "System paused — risk protection triggered."
    elif positions:
        status_id = f"Sistem memegang {len(positions)} saham (latihan)."
        status_en = f"Holding {len(positions)} stocks (practice)."
    elif risk_off and riskoff:
        # Don't say "waiting in cash" full stop when there IS a screen below; the
        # reader would stop reading at the first line and never reach it.
        status_id = ("Pasar sedang lesu — portofolio latihan menunggu di posisi tunai, "
                     "dan layar pasar lesu di bawah menunjukkan yang sedang diamati.")
        status_en = ("Market regime is weak — the practice portfolio is waiting in cash, "
                     "and the weak-market screen below shows what is being watched.")
    elif risk_off:
        status_id = "Pasar sedang lesu — sistem menunggu dengan aman di posisi tunai."
        status_en = "Market regime is weak — safely waiting in cash."
    else:
        status_id = "Sistem siap — belum ada sinyal beli hari ini."
        status_en = "Ready — no buy signal today."

    return {
        **_freshness(eq.date.iloc[-1]),
        "status_id": status_id, "status_en": status_en,
        "equity_str": f"{equity_now:,.0f}".replace(",", "."),
        "ret_start": ret_start, "ret_day": ret_day,
        "chart_svg": equity_chart_svg(eq, meta.get("backfill_until")),
        "backfill_until": meta.get("backfill_until"),
        "positions": positions,
        "active_signals": active, "n_waiting": n_waiting,
        "recent_trades": recent.to_dict("records"),
        "halted": halted,
        "research": research, "market": market, "riskoff": riskoff,
    }


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    ctx = build_dashboard_context()
    if ctx is None:
        return HTMLResponse("<h1>Paper portfolio not initialized — run "
                            "<code>scripts/run_pipeline.py paper</code></h1>")
    return TEMPLATES.TemplateResponse(request, "dashboard.html", ctx)
