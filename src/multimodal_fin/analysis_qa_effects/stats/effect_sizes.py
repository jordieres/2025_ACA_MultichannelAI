import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from scipy.stats import norm

from multimodal_fin.analysis_qa_effects.config import PipelineConfig

class StatsTester:
    def __init__(self, config: PipelineConfig):
        self.config = config

    @staticmethod
    def _cohens_d_welch(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        nx, ny = len(x), len(y)
        if nx < 2 or ny < 2:
            return None
        mx, my = x.mean(), y.mean()
        sx, sy = x.std(ddof=1), y.std(ddof=1)
        denom = np.sqrt((sx**2 + sy**2) / 2.0)
        if not np.isfinite(denom) or denom == 0:
            return None
        return (mx - my) / denom

    @staticmethod
    def _hedges_g_from_d(d, nx, ny):
        if d is None or not np.isfinite(d):
            return None
        df = nx + ny - 2
        if df <= 0:
            return None
        J = 1 - (3 / (4 * df - 1))
        return J * d

    def compute_tests(self, df_audio: pd.DataFrame, df_text: pd.DataFrame) -> pd.DataFrame:
        emotions = list(self.config.emotions_order)
        min_n = self.config.min_n

        results = []
        for emo in emotions:
            audio_col = f"audio_{emo}"
            text_col  = f"text_{emo}"

            # AUDIO
            t_audio = p_audio = d_audio = g_audio = None
            yes_audio = ev_audio = np.array([], dtype=float)

            if audio_col in df_audio.columns:
                yes_audio = df_audio.loc[df_audio["answered"] == "yes", audio_col].dropna().to_numpy(float)
                ev_audio  = df_audio.loc[df_audio["is_evasive"],        audio_col].dropna().to_numpy(float)
                if len(yes_audio) >= min_n and len(ev_audio) >= min_n:
                    t_audio, p_audio = ttest_ind(yes_audio, ev_audio, equal_var=False)
                    d_audio = self._cohens_d_welch(yes_audio, ev_audio)
                    g_audio = self._hedges_g_from_d(d_audio, len(yes_audio), len(ev_audio))

            # TEXT
            t_text = p_text = d_text = g_text = None
            yes_text = ev_text = np.array([], dtype=float)

            if text_col in df_text.columns:
                yes_text = df_text.loc[df_text["answered"] == "yes", text_col].dropna().to_numpy(float)
                ev_text  = df_text.loc[df_text["is_evasive"],        text_col].dropna().to_numpy(float)
                if len(yes_text) >= min_n and len(ev_text) >= min_n:
                    t_text, p_text = ttest_ind(yes_text, ev_text, equal_var=False)
                    d_text = self._cohens_d_welch(yes_text, ev_text)
                    g_text = self._hedges_g_from_d(d_text, len(yes_text), len(ev_text))

            def _safe_mean(a): return float(a.mean()) if len(a) else None
            def _safe_std(a):  return float(a.std(ddof=1)) if len(a) > 1 else None

            a_my, a_me = _safe_mean(yes_audio), _safe_mean(ev_audio)
            t_my, t_me = _safe_mean(yes_text),  _safe_mean(ev_text)

            results.append({
                "emotion": emo,
                "audio_t": t_audio,
                "audio_p_value": p_audio,
                "audio_n_yes": int(len(yes_audio)),
                "audio_n_evasive": int(len(ev_audio)),
                "audio_mean_yes": a_my,
                "audio_mean_evasive": a_me,
                "audio_std_yes": _safe_std(yes_audio),
                "audio_std_evasive": _safe_std(ev_audio),
                "audio_delta_mean": (a_my - a_me) if (a_my is not None and a_me is not None) else None,
                "audio_cohens_d": d_audio,
                "audio_hedges_g": g_audio,

                "text_t": t_text,
                "text_p_value": p_text,
                "text_n_yes": int(len(yes_text)),
                "text_n_evasive": int(len(ev_text)),
                "text_mean_yes": t_my,
                "text_mean_evasive": t_me,
                "text_std_yes": _safe_std(yes_text),
                "text_std_evasive": _safe_std(ev_text),
                "text_delta_mean": (t_my - t_me) if (t_my is not None and t_me is not None) else None,
                "text_cohens_d": d_text,
                "text_hedges_g": g_text,
            })

        return pd.DataFrame(results)

    def add_hedges_g_ci(self, df_tests: pd.DataFrame, modality: str) -> pd.DataFrame:
        ci_level = self.config.ci_level
        alpha = 1 - ci_level
        z = norm.ppf(1 - alpha/2)

        gcol = f"{modality}_hedges_g"
        dcol = f"{modality}_cohens_d"
        n1c  = f"{modality}_n_yes"
        n2c  = f"{modality}_n_evasive"

        lo_list, hi_list = [], []
        for _, r in df_tests.iterrows():
            g = r[gcol]
            d = r[dcol]
            n1, n2 = int(r[n1c]), int(r[n2c])

            if g is None or d is None or not np.isfinite(g) or not np.isfinite(d) or n1 < 2 or n2 < 2:
                lo_list.append(None); hi_list.append(None); continue

            df_ = n1 + n2 - 2
            if df_ <= 0:
                lo_list.append(None); hi_list.append(None); continue

            se_d = np.sqrt((n1 + n2) / (n1 * n2) + (d**2) / (2 * df_))
            lo_list.append(g - z * se_d)
            hi_list.append(g + z * se_d)

        df_tests = df_tests.copy()
        df_tests[f"{modality}_g_ci_low"] = lo_list
        df_tests[f"{modality}_g_ci_high"] = hi_list
        return df_tests