from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class PipelineConfig:
    processed_root: str = "/home/aacastro/mchai/processed_companies"
    emotions_order: Tuple[str, ...] = ("happy", "neutral", "surprise", "disgust", "anger", "sadness", "fear")
    min_n: int = 10
    ci_level: float = 0.95
    alpha_sig: float = 0.10
    keybert_top_n: int = 5