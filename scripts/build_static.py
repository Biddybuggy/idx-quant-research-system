#!/usr/bin/env python3
"""Render the dashboard to a static site/ directory for GitHub Pages.

Uses the same template and the same context builder as the live FastAPI
server, so the static page is pixel-identical to the served one. Runs after
the daily job in .github/workflows/daily.yml.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jinja2 import Environment, FileSystemLoader, select_autoescape

from idxquant.api.app import STATIC, build_dashboard_context
from idxquant.config import ROOT

SITE = ROOT / "site"


def main():
    ctx = build_dashboard_context()
    if ctx is None:
        sys.exit("paper portfolio not initialized — run scripts/run_pipeline.py paper first")

    env = Environment(
        loader=FileSystemLoader(Path(__file__).resolve().parent.parent
                                / "idxquant" / "api" / "templates"),
        autoescape=select_autoescape(["html"]),
    )
    html = env.get_template("dashboard.html").render(**ctx)

    SITE.mkdir(exist_ok=True)
    (SITE / "index.html").write_text(html)
    for icon in ("icon-180.png", "icon-192.png", "icon-512.png"):
        shutil.copy(STATIC / icon, SITE / icon)
    (SITE / "manifest.webmanifest").write_text(json.dumps({
        "name": "Dasbor Saham", "short_name": "Saham",
        "start_url": ".", "display": "standalone",
        "background_color": "#f9f9f7", "theme_color": "#2a78d6",
        "icons": [{"src": "icon-192.png", "sizes": "192x192", "type": "image/png"},
                  {"src": "icon-512.png", "sizes": "512x512", "type": "image/png"}],
    }))
    # .nojekyll: serve files as-is, skip GitHub's Jekyll build
    (SITE / ".nojekyll").write_text("")
    print(f"static site built in {SITE} (as of {ctx['as_of']})")


if __name__ == "__main__":
    main()
