"""Telegram notifications.

Setup (once):
  1. In Telegram, talk to @BotFather -> /newbot -> get the bot token.
  2. Send your new bot any message, then open
     https://api.telegram.org/bot<TOKEN>/getUpdates to read your chat id.
     (For mom: have her message the bot too, and add her chat id.)
  3. Set env vars TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_IDS (comma-separated).

If the env vars are missing, sending is skipped gracefully — the pipeline
never fails because notifications aren't configured.
"""
from __future__ import annotations

import os

import requests


def configured() -> bool:
    return bool(os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_IDS"))


def send(text: str) -> bool:
    """Send to all configured chat ids. Returns True if every send succeeded."""
    if not configured():
        print("[telegram] not configured, skipping:", text[:80].replace("\n", " "))
        return False
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    ok = True
    for chat_id in os.environ["TELEGRAM_CHAT_IDS"].split(","):
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id.strip(), "text": text, "parse_mode": "HTML"},
            timeout=15,
        )
        if not r.ok:
            print(f"[telegram] send failed for chat {chat_id}: {r.text[:200]}")
            ok = False
    return ok


def compose_daily(signal_payload: dict, paper_summary: dict,
                  research_line: str = "") -> str:
    """One friendly EOD message, Indonesian first. Sent even on quiet days —
    silence must mean 'broken', never 'nothing happened'."""
    equity = paper_summary["equity"]
    n_pos = len(paper_summary["positions"])
    actions = [s for s in signal_payload["signals"]
               if s["action"] in ("ENTER_LONG", "EXIT")]
    risk_off = signal_payload["regime"].startswith("risk-off")

    lines = [f"📊 <b>Laporan harian — tutup {signal_payload['as_of_close']}</b>"]
    if paper_summary.get("halted"):
        lines.append("⛔ Sistem sedang berhenti sementara (batas penurunan tercapai).")
    if actions:
        lines.append("🗓️ <b>Rencana portofolio latihan untuk pembukaan besok:</b>")
        for s in actions:
            verb = "BELI" if s["action"] == "ENTER_LONG" else "JUAL"
            lines.append(f"• {verb} {s['ticker'].replace('.JK', '')} (latihan) — "
                         f"keyakinan {s['confidence']}")
    elif risk_off:
        lines.append("🗓️ Rencana besok: tetap menunggu di posisi aman (tidak ada saham). "
                     "Pasar sedang lesu.")
    else:
        lines.append("🗓️ Rencana besok: tidak ada transaksi; posisi dipertahankan.")
    if research_line:
        lines.append(research_line)
    lines.append(f"💼 Portofolio latihan: Rp {equity:,.0f} ({n_pos} saham)")
    lines.append("<i>Ini latihan (paper trading), bukan uang sungguhan dan bukan saran investasi.</i>")
    return "\n".join(lines)


def compose_error(stage: str, err: Exception) -> str:
    return (f"⚠️ <b>Pipeline error</b> di tahap <b>{stage}</b>:\n"
            f"<code>{type(err).__name__}: {err}</code>\n"
            f"Dasbor mungkin menampilkan data kemarin sampai ini diperbaiki.")
