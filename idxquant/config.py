"""Typed config loader. Single source of truth: config/settings.yaml."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Costs:
    buy_commission: float
    sell_commission: float
    half_spread: float
    slippage: float

    @property
    def buy_friction(self) -> float:
        """Price uplift applied to buys (spread + slippage)."""
        return self.half_spread + self.slippage

    @property
    def sell_friction(self) -> float:
        return self.half_spread + self.slippage


@dataclass
class Config:
    tickers: list[str]
    index_ticker: str
    benchmark_stock: str
    start: str
    db_path: Path
    max_daily_move: float
    min_price: float
    costs: Costs
    lot_size: int
    min_adv_idr: float
    max_adv_participation: float
    initial_cash: float
    max_weight: float
    regime_sma: int
    max_drawdown_halt: float
    halt_cooloff_days: int
    strategy: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict, repr=False)


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else ROOT / "config" / "settings.yaml"
    raw = yaml.safe_load(path.read_text())
    db_path = ROOT / raw["data"]["db_path"]
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return Config(
        tickers=list(raw["universe"]["tickers"]),
        index_ticker=raw["universe"]["index_ticker"],
        benchmark_stock=raw["universe"]["benchmark_stock"],
        start=raw["data"]["start"],
        db_path=db_path,
        max_daily_move=float(raw["data"]["max_daily_move"]),
        min_price=float(raw["data"]["min_price"]),
        costs=Costs(**{k: float(v) for k, v in raw["costs"].items()}),
        lot_size=int(raw["market"]["lot_size"]),
        min_adv_idr=float(raw["market"]["min_adv_idr"]),
        max_adv_participation=float(raw["market"]["max_adv_participation"]),
        initial_cash=float(raw["portfolio"]["initial_cash"]),
        max_weight=float(raw["portfolio"]["max_weight"]),
        regime_sma=int(raw["risk"]["regime_sma"]),
        max_drawdown_halt=float(raw["risk"]["max_drawdown_halt"]),
        halt_cooloff_days=int(raw["risk"]["halt_cooloff_days"]),
        strategy=dict(raw["strategy"]),
        raw=raw,
    )
