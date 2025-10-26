from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import json


@dataclass
class AprioriConfig:
    data: Optional[str] = None
    min_support: Optional[float] = None
    min_confidence: Optional[float] = None
    min_lift: Optional[float] = None
    all_combos_max_k: Optional[int] = None
    seed: Optional[int] = None
    test_size: Optional[float] = None
    progress: Optional[bool] = None
    no_eval: Optional[bool] = None


def load_config_json(path: Optional[str]) -> AprioriConfig:
    if not path:
        return AprioriConfig()
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return AprioriConfig(
        data=obj.get("data"),
        min_support=obj.get("min_support"),
        min_confidence=obj.get("min_confidence"),
        min_lift=obj.get("min_lift"),
        all_combos_max_k=obj.get("all_combos_max_k"),
        seed=obj.get("seed"),
        test_size=obj.get("test_size"),
        progress=obj.get("progress"),
        no_eval=obj.get("no_eval"),
    )
