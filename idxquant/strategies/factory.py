"""Strategy construction from config — shared by CLI, paper executor, and API."""
from __future__ import annotations

from ..config import Config
from .momentum import CrossSectionalMomentum
from .sma_crossover import SmaCrossover


def make_strategy(cfg: Config, **overrides):
    s = {**cfg.strategy, **{k: v for k, v in overrides.items() if v is not None}}
    name = s["name"]
    if name == "momentum":
        return CrossSectionalMomentum(int(s["lookback"]), int(s["skip"]),
                                      int(s["top_n"]), cfg.regime_sma)
    if name == "sma_crossover":
        return SmaCrossover(int(s["fast"]), int(s["slow"]), cfg.regime_sma)
    raise ValueError(f"Unknown strategy: {name}")
