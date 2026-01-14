import numpy as np
import pandas as pd
from scipy.stats import norm

from multimodal_fin.analysis_qa_effects.config import PipelineConfig

class PopulationEvidenceAnalyzer:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def compute_population_evidence(self, df_all: pd.DataFrame) -> pd.DataFrame:
        """
        Population-level evidence per (emotion, modality) from company-level hedges_g.
        Uses normal approximation CI for the mean.
        """
        alpha = 1 - self.config.ci_level
        z = norm.ppf(1 - alpha / 2)

        rows = []
        for (emo, mod), s in df_all.groupby(["emotion", "modality"])["hedges_g"]:
            g = s.dropna().to_numpy(dtype=float)
            n = len(g)

            if n < self.config.min_n:
                rows.append({
                    "emotion": emo, "modality": mod, "n_companies": n,
                    "mean_g": np.nan, "sd_g": np.nan, "se_mean": np.nan,
                    "ci_low_mean": np.nan, "ci_high_mean": np.nan,
                    "share_g_gt0": np.nan, "share_g_lt0": np.nan,
                    "evidence_mean_nonzero": False
                })
                continue

            mean_g = float(np.mean(g))
            sd_g = float(np.std(g, ddof=1))
            se = sd_g / np.sqrt(n)
            ci_low = mean_g - z * se
            ci_high = mean_g + z * se

            rows.append({
                "emotion": emo,
                "modality": mod,
                "n_companies": int(n),
                "mean_g": mean_g,
                "sd_g": sd_g,
                "se_mean": float(se),
                "ci_low_mean": float(ci_low),
                "ci_high_mean": float(ci_high),
                "share_g_gt0": float(np.mean(g > 0)),
                "share_g_lt0": float(np.mean(g < 0)),
                "evidence_mean_nonzero": bool((ci_low > 0) or (ci_high < 0))
            })

        out = pd.DataFrame(rows)

        # stable emotion ordering
        canonical = list(self.config.emotions_order)
        if set(canonical).issubset(set(out["emotion"].unique())):
            out["emotion"] = pd.Categorical(out["emotion"], categories=canonical, ordered=True)
            out = out.sort_values(["emotion", "modality"]).reset_index(drop=True)

        return out