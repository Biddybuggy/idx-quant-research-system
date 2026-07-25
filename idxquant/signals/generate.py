"""End-of-day signal file: what would the strategy do at tomorrow's open?

Every signal carries confidence, reasoning, expected holding period,
invalidation condition, and risk fields — no bare BUY/SELL flags.
Actions: ENTER_LONG, HOLD_LONG, EXIT, NO_POSITION.

Strategy-specific text comes from strategy.signal_context(ticker, ...).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..config import Config
from ..features import indicators as ind


def generate_signal_file(
    prices: dict[str, pd.DataFrame],
    index_close: pd.Series,
    strategy,
    cfg: Config,
    expected_holding_days: float,
    out_path: Path,
    held_tickers: set[str] | None = None,
) -> dict:
    """`held_tickers` is what the paper portfolio ACTUALLY owns right now.

    When supplied, the action is what the portfolio will do at the next open,
    which is what a reader means by "what is it doing". Without it the action
    can only describe the strategy's own signal transition, and the two diverge
    the moment the configured strategy changes: on 2026-07-25 the strategy was
    switched to regime_switch, whose reversal leg had been signalling ASII and
    UNTR since March, so the file said HOLD_LONG for names the portfolio had
    never bought while the dashboard next to it said "holding nothing".
    """
    sig = strategy.signals(prices, index_close)
    today, yesterday = sig.index[-1], sig.index[-2]
    regime_ok = bool(ind.regime_filter(index_close, strategy.regime_sma).iloc[-1])
    n_active = int(sig.loc[today].sum())
    weight = min(cfg.max_weight, 1.0 / n_active) if n_active else 0.0

    entries = []
    for t in sig.columns:
        now = int(sig.loc[today, t])
        # Fall back to the signal's own transition only when holdings are unknown.
        prev = (int(t in held_tickers) if held_tickers is not None
                else int(sig.loc[yesterday, t]))
        action = {(1, 0): "ENTER_LONG", (1, 1): "HOLD_LONG",
                  (0, 1): "EXIT", (0, 0): "NO_POSITION"}[(now, prev)]
        ctx = strategy.signal_context(t, prices, index_close)
        close = prices[t]["Close"]
        atr_pct = float(ind.atr(prices[t]).iloc[-1] / close.iloc[-1])
        entries.append({
            "ticker": t,
            "action": action,
            "confidence": ctx["confidence"] if now else "n/a",
            "reasoning": ctx["reasoning"],
            "expected_holding_days": expected_holding_days,
            "invalidation": ctx["invalidation"],
            "risk": {
                "suggested_weight": round(weight if now else 0.0, 4),
                "atr_pct_of_price": round(atr_pct, 4),
                "last_close_idr": float(close.iloc[-1]),
                "lot_size": cfg.lot_size,
                **ctx.get("extra_risk", {}),
            },
        })

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "as_of_close": str(today.date()),
        "execute_at": "next market open",
        "strategy": strategy.name,
        "regime": "risk-on" if regime_ok else "risk-off (JCI below regime SMA)",
        "disclaimer": "Research output, not investment advice. Paper trading only.",
        "signals": entries,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    return payload
