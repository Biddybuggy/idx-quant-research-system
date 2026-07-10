# Deploying the dashboard (so it works on Mom's iPhone)

The whole backend is one container: FastAPI serves the dashboard + JSON API and
runs the daily job in-process at **17:45 WIB** every day. You need a host that
keeps one small machine always on with a persistent disk.

## 0. Prerequisites (10 minutes, one time)

1. **Telegram bot** (optional but recommended):
   - In Telegram, message **@BotFather** → `/newbot` → pick a name → copy the token.
   - Send your new bot any message ("halo").
   - Open `https://api.telegram.org/bot<TOKEN>/getUpdates` in a browser and copy
     your `chat.id` from the JSON.
   - Have Mom message the bot too, refresh that URL, and copy her chat id as well.
2. **Pick a dashboard key**: any long random string, e.g. run
   `python3 -c "import secrets; print(secrets.token_urlsafe(24))"`.

## 1. Fly.io (recommended)

```bash
brew install flyctl && fly auth signup      # or fly auth login

cd idx_quant_research_system
# edit fly.toml: change app = "idx-dashboard" to something globally unique

fly launch --no-deploy --copy-config        # accept the existing fly.toml
fly volumes create idxdata --size 1 --region sin

fly secrets set \
  DASHBOARD_KEY="<your-random-key>" \
  TELEGRAM_BOT_TOKEN="<botfather-token>" \
  TELEGRAM_CHAT_IDS="<your-id>,<moms-id>"

fly deploy
```

First-time data setup (one time, on the machine):

```bash
fly ssh console
cd /app
python scripts/run_pipeline.py ingest
python scripts/run_pipeline.py paper --backfill-days 126   # labeled warm-up curve
python scripts/run_pipeline.py signal
exit
```

Your dashboard is now at `https://<app-name>.fly.dev/?key=<your-key>`.

Costs: a shared-cpu-1x with 512MB and a 1GB volume is a few USD/month.
`auto_stop_machines = false` is required — a stopped machine has no scheduler.

## 2. Railway (alternative)

- New project → Deploy from GitHub repo (Railway detects the Dockerfile).
- Add a **Volume** mounted at `/app/data`.
- Set the same three environment variables.
- Networking → Generate Domain. Run the same first-time setup via the service shell.

## 3. Put it on Mom's iPhone (2 minutes)

1. Send her the full link (`https://…/?key=…`) via WhatsApp/Telegram.
2. She opens it in **Safari** (must be Safari, not the in-app browser: if it opens
   inside WhatsApp, tap the share icon → "Open in Safari").
3. Safari → **Share button** → **Add to Home Screen** → "Saham" icon appears.
4. The key is baked into the installed app, so she never sees a login.
5. Install Telegram, have her message the bot once, and she gets the daily
   17:45 WIB summary automatically.

## 4. Operating notes

- **Health check**: `https://…/api/health` is public (no key) — point a free
  uptime monitor (e.g. UptimeRobot) at it and you'll know if the app is down.
- **Silence = broken**: the bot messages every day, including "nothing happened".
  If a day passes with no message, check `fly logs`.
- **Data revisions**: every daily run re-downloads full history and upserts, so
  Yahoo corrections self-heal. Ingest failures for individual tickers are logged
  to the `data_quality` table and the pipeline continues on stored data.
- **Rotating the key**: `fly secrets set DASHBOARD_KEY=...` then re-add to Home
  Screen with the new link (the old installed app stops working — that's the point).
- **What this deployment can never do**: place real orders. There is no broker
  code path in the API. Keep it that way; Phase 6 experiments belong in a
  separate deployment.
