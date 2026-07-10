"""Server-rendered SVG equity chart (mobile-first, theme-aware via CSS vars).

Spec: 2px round-join line in the series hue, area wash at 10% opacity,
hairline solid gridlines, clean y-ticks, no legend (single series — the card
title names it). A vertical hairline marks where the simulated warm-up ends
and live paper trading begins. Hover crosshair/tooltip is added client-side
from the data-pts JSON baked onto the <svg>.
"""
from __future__ import annotations

import json
import math

import pandas as pd

W, H = 360, 180
ML, MR, MT, MB = 46, 10, 10, 24  # margins: left for y labels, bottom for dates


def _nice_step(raw: float) -> float:
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def _fmt_juta(v: float) -> str:
    return f"{v / 1e6:,.0f} jt".replace(",", ".")


def equity_chart_svg(eq: pd.DataFrame, backfill_until: str | None) -> str:
    df = eq[["date", "equity"]].dropna().reset_index(drop=True)
    if len(df) > 250:  # downsample, always keeping the last point
        idx = list(range(0, len(df), math.ceil(len(df) / 250)))
        if idx[-1] != len(df) - 1:
            idx.append(len(df) - 1)
        df = df.iloc[idx].reset_index(drop=True)

    x0, x1 = df.date.iloc[0].value, df.date.iloc[-1].value
    lo, hi = df.equity.min(), df.equity.max()
    pad = max((hi - lo) * 0.1, hi * 0.005)
    lo, hi = lo - pad, hi + pad
    step = _nice_step((hi - lo) / 4)
    tick0 = math.ceil(lo / step) * step
    ticks = []
    t = tick0
    while t <= hi:
        ticks.append(t)
        t += step

    def X(ts) -> float:
        return ML + (ts.value - x0) / max(x1 - x0, 1) * (W - ML - MR)

    def Y(v) -> float:
        return MT + (hi - v) / (hi - lo) * (H - MT - MB)

    pts = [(X(d), Y(v)) for d, v in zip(df.date, df.equity)]
    line = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in pts)
    base_y = H - MB
    area = f"{line} L{pts[-1][0]:.1f} {base_y} L{pts[0][0]:.1f} {base_y} Z"

    grid = "".join(
        f'<line x1="{ML}" y1="{Y(v):.1f}" x2="{W - MR}" y2="{Y(v):.1f}" class="grid"/>'
        f'<text x="{ML - 6}" y="{Y(v) + 3:.1f}" class="tick" text-anchor="end">{_fmt_juta(v)}</text>'
        for v in ticks)

    # 3 evenly spaced date labels
    n = len(df) - 1
    xlab = "".join(
        f'<text x="{X(df.date.iloc[i]):.1f}" y="{H - 8}" class="tick" '
        f'text-anchor="{a}">{df.date.iloc[i].strftime("%b %y")}</text>'
        for i, a in ((0, "start"), (n // 2, "middle"), (n, "end")))

    marker = ""
    if backfill_until:
        bu = pd.Timestamp(backfill_until)
        if df.date.iloc[0] < bu < df.date.iloc[-1]:
            mx = X(bu)
            marker = (f'<line x1="{mx:.1f}" y1="{MT}" x2="{mx:.1f}" y2="{base_y}" '
                      f'class="marker"/>')

    data = [[round(x, 1), round(y, 1), d.strftime("%d %b %Y"), f"{v:,.0f}".replace(",", ".")]
            for (x, y), d, v in zip(pts, df.date, df.equity)]

    return f'''<svg viewBox="0 0 {W} {H}" class="eqchart" data-pts='{json.dumps(data)}'
     preserveAspectRatio="xMidYMid meet" role="img"
     aria-label="Grafik nilai portofolio latihan dari waktu ke waktu">
  {grid}{xlab}{marker}
  <path d="{area}" class="area"/>
  <path d="{line}" class="line"/>
  <line class="hov-x" x1="0" y1="{MT}" x2="0" y2="{base_y}" style="display:none"/>
  <circle class="hov-dot" r="4" style="display:none"/>
</svg>'''
