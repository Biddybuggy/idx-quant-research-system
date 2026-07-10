#!/usr/bin/env python3
"""One-time Telegram setup helper.

Run:  .venv/bin/python scripts/setup_telegram.py

It will:
  1. check your bot token works,
  2. find the chat IDs of everyone who has messaged the bot,
  3. send each of them a test message,
  4. print the exact settings to use locally and on Fly.io.
"""
from __future__ import annotations

import sys

import requests

API = "https://api.telegram.org/bot{token}/{method}"


def call(token: str, method: str, **params) -> dict:
    r = requests.post(API.format(token=token, method=method), json=params, timeout=15)
    data = r.json()
    if not data.get("ok"):
        sys.exit(f"Telegram API error on {method}: {data.get('description')}")
    return data["result"]


def find_chats(token: str) -> dict[int, str]:
    chats: dict[int, str] = {}
    for upd in call(token, "getUpdates"):
        msg = upd.get("message") or upd.get("edited_message")
        if msg and "chat" in msg:
            c = msg["chat"]
            name = " ".join(filter(None, [c.get("first_name"), c.get("last_name")])) \
                   or c.get("title") or str(c["id"])
            chats[c["id"]] = name
    return chats


def main():
    print("Paste your bot token from @BotFather (looks like 123456789:AAF...):")
    token = input("> ").strip()

    me = call(token, "getMe")
    bot_name = me["username"]
    print(f"\n✓ Token works. Your bot is @{bot_name}\n")

    chats = find_chats(token)
    while not chats:
        print(f"No one has messaged the bot yet. Open Telegram, search @{bot_name},")
        print("press START and send it any message (e.g. 'halo').")
        print("Do the same from your mom's phone if she's ready.")
        input("Then press Enter here to check again... ")
        chats = find_chats(token)

    print("Found these people:")
    for cid, name in chats.items():
        print(f"  {name}: chat id {cid}")

    for cid, name in chats.items():
        call(token, "sendMessage", chat_id=cid,
             text=f"✅ Halo {name}! Bot dasbor saham sudah tersambung. "
                  f"Laporan harian akan dikirim setiap hari sekitar 17:45 WIB.")
    print("\n✓ Test message sent — check Telegram!\n")

    ids = ",".join(str(c) for c in chats)
    print("=" * 60)
    print("Use these settings.\n")
    print("To test the full daily report right now:\n")
    print(f'  TELEGRAM_BOT_TOKEN="{token}" TELEGRAM_CHAT_IDS="{ids}" \\')
    print("    .venv/bin/python scripts/daily_job.py\n")
    print("When you deploy to Fly.io:\n")
    print(f'  fly secrets set TELEGRAM_BOT_TOKEN="{token}" TELEGRAM_CHAT_IDS="{ids}"')
    print("=" * 60)
    print("\nRun this script again any time (e.g. after your mom messages the bot)")
    print("to pick up new chat ids.")


if __name__ == "__main__":
    main()
