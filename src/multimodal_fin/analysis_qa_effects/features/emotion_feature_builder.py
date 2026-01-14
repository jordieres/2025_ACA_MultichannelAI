from typing import Tuple
import pandas as pd
import numpy as np

from multimodal_fin.analysis_qa_effects.config import PipelineConfig

class EmotionFeatureBuilder:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def build_audio_text_views(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        order = list(self.config.emotions_order)

        # AUDIO
        df_audio = df.copy()
        mask_audio = df_audio["audio_emo_mean"].notna()
        if mask_audio.any():
            audio_mat = np.vstack(df_audio.loc[mask_audio, "audio_emo_mean"].values)
            assert audio_mat.shape[1] == len(order), "Audio embedding dim != 7"
            audio_cols = [f"audio_{e}" for e in order]
            for c in audio_cols: df_audio[c] = np.nan
            df_audio.loc[mask_audio, audio_cols] = audio_mat

        # TEXT
        df_text = df.copy()
        mask_text = df_text["text_emo_mean"].notna()
        if mask_text.any():
            text_mat = np.vstack(df_text.loc[mask_text, "text_emo_mean"].values)
            assert text_mat.shape[1] == len(order), "Text embedding dim != 7"
            text_cols = [f"text_{e}" for e in order]
            for c in text_cols: df_text[c] = np.nan
            df_text.loc[mask_text, text_cols] = text_mat

        for d in (df_audio, df_text):
            d["answered"] = d["answered"].str.lower()
            d["is_evasive"] = d["answered"].isin(["no", "partially"])

        return df_audio, df_text