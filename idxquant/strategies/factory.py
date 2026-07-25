"""Strategy construction from config — shared by CLI, paper executor, and API."""
from __future__ import annotations

from ..config import Config
from .momentum import CrossSectionalMomentum
from .regime_switch import RegimeSwitch
from .reversal import RiskOffReversal
from .sma_crossover import SmaCrossover
from .tactical_rs import TacticalRelativeStrength


def _momentum(s, cfg, top_n: int | None = None) -> CrossSectionalMomentum:
    return CrossSectionalMomentum(int(s["lookback"]), int(s["skip"]),
                                  int(top_n if top_n is not None else s["top_n"]),
                                  cfg.regime_sma)


def _reversal(s, cfg) -> RiskOffReversal:
    return RiskOffReversal(
        lookback=int(s.get("rev_lookback", 60)),
        top_n=int(s.get("rev_top_n", 5)),
        rebalance_days=int(s.get("rev_rebalance_days", 21)),
        regime_sma=cfg.regime_sma,
        min_adv_idr=cfg.min_adv_idr)


def make_strategy(cfg: Config, **overrides):
    s = {**cfg.strategy, **{k: v for k, v in overrides.items() if v is not None}}
    name = s["name"]
    if name == "momentum":
        return _momentum(s, cfg)
    if name == "reversal":
        return _reversal(s, cfg)
    if name == "regime_switch":
        # The composite's risk-on leg holds MORE names than momentum-alone does.
        # Concentration is what made the standalone strategy fragile, and top_n
        # 8-13 is a measured plateau, so the two are configured separately
        # rather than sharing one number that suits neither.
        return RegimeSwitch(_momentum(s, cfg, s.get("switch_top_n", 8)),
                            _reversal(s, cfg), cfg.regime_sma)
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
