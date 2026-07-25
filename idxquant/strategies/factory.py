"""Strategy construction from config — shared by CLI, paper executor, and API."""
from __future__ import annotations

from ..config import Config
from .momentum import CrossSectionalMomentum
from .sma_crossover import SmaCrossover
from .tactical_rs import TacticalRelativeStrength


def make_strategy(cfg: Config, **overrides):
    s = {**cfg.strategy, **{k: v for k, v in overrides.items() if v is not None}}
    name = s["name"]
    if name == "momentum":
        return CrossSectionalMomentum(int(s["lookback"]), int(s["skip"]),
                                      int(s["top_n"]), cfg.regime_sma)
    if name == "sma_crossover":
        return SmaCrossover(int(s["fast"]), int(s["slow"]), cfg.regime_sma)
    if name == "tactical_rs":
        return TacticalRelativeStrength(
            top_n=int(s.get("tact_top_n", 5)),
            min_adv_bn=cfg.min_adv_idr / 1e9,
            min_score=float(s.get("tact_min_score", 55.0)),
            rebalance_days=int(s.get("tact_rebalance_days", 1)),
            use_stop=bool(s.get("tact_use_stop", True)))
    raise ValueError(f"Unknown strategy: {name}")
