# Free deployment: GitHub Actions + GitHub Pages

Runs the entire system for **$0/month**: a GitHub Actions workflow executes the
daily job at 17:45 WIB (data update → paper trades → signal file → Telegram),
commits the tiny paper-portfolio state back to the repo, and publishes the
dashboard as a static page on GitHub Pages.

## The trade-offs vs a paid server (accept these first)

- **The repo must be public** (free GitHub Pages requires it). Code, paper
  portfolio state, and the dashboard URL are visible to anyone who finds them.
  Nothing sensitive is in them — practice money, public market prices — and
  the Telegram token lives in GitHub Secrets, never in the repo.
- **No password on the dashboard.** The URL is obscure but public.
- **Schedule jitter**: GitHub cron can run minutes-to-an-hour late at busy
  times. For a daily EOD report this doesn't matter.
- **Scheduled workflows can be auto-disabled** after ~60 days without repo
  activity. The daily state commit normally keeps it active, but if GitHub
  ever emails you about it, click "re-enable" — and treat a silent Telegram
  day as a prompt to check.
- **Yahoo Finance sometimes rate-limits cloud IPs.** Ingest retries and
  tolerates partial failures; a fully failed run emails you (workflow failure)
  and Telegram gets an error alert. If it becomes chronic, we revisit.

## Setup (one time, ~15 minutes)

### 1. Push the project to a public GitHub repo

```bash
cd ~/Documents/programming_folder/idx_quant_research_system
git init -b main           # if not already done
git add -A && git commit -m "IDX quant paper-trading system"
# create a PUBLIC repo named e.g. idx-trading on github.com, then:
git remote add origin https://github.com/<YOUR-USERNAME>/idx-trading.git
git push -u origin main
```

(Or with the GitHub CLI: `gh repo create idx-trading --public --source=. --push`)

### 2. Add the Telegram secrets

Repo → **Settings → Secrets and variables → Actions → New repository secret**:

- `TELEGRAM_BOT_TOKEN` — from @BotFather
- `TELEGRAM_CHAT_IDS` — comma-separated chat ids (yours, later Mom's too)

### 3. Seed the paper state

If you already ran the paper portfolio locally, its state is in `state/`
(created by `python scripts/state_sync.py export`) and was pushed with the repo
— the cloud continues the same portfolio. Otherwise the first run initializes
a fresh one automatically.

### 4. Run it once by hand

Repo → **Actions** tab → **daily** workflow → **Run workflow**. Watch it go
green (~3–5 min). This first run also enables GitHub Pages automatically.

Your dashboard is now at:

```
https://<YOUR-USERNAME>.github.io/idx-trading/
```

### 5. Mom's iPhone

1. Send her the link; she opens it **in Safari**.
2. Share → **Add to Home Screen** → "Saham" icon appears.
3. She messages the Telegram bot once; re-run
   `.venv/bin/python scripts/setup_telegram.py` locally to get her chat id,
   and update the `TELEGRAM_CHAT_IDS` secret to `yourid,momid`.

## How it stays in sync

- The workflow commits `state/paper_state.json` daily. If you also run the
  paper portfolio locally, `git pull` first, or local and cloud state fork.
  Simplest rule once deployed: **the cloud owns the portfolio; local is for
  research** (backtests don't touch paper state).
- A failed run = GitHub failure email + Telegram error alert + dashboard keeps
  showing yesterday (with the ⚠️ stale banner after 5 days). Silence on
  Telegram = go look at the Actions tab.
