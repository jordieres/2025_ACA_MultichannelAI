import os
import re
import json
import tempfile
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd
from pydub import AudioSegment

from multimodal_fin.processing.multimodal.audio.audio_emotion_analyzer import AudioEmotionAnalyzer
from multimodal_fin.processing.multimodal.text.text_emotion_analyzer import TextEmotionAnalyzer
from multimodal_fin.processing.multimodal.video.video_emotion_analyzer import VideoEmotionAnalyzer

# Patch for NumPy NEP 50 warning (avoid runtime errors) https://stackoverflow.com/questions/77064579/module-numpy-has-no-attribute-no-nep50-warning
def dummy_npwarn_decorator_factory():
    def npwarn_decorator(x):
        return x
    return npwarn_decorator
np._no_nep50_warning = getattr(np, '_no_nep50_warning', dummy_npwarn_decorator_factory)


@dataclass
class MultimodalEmbeddings:
    """
    Generates multimodal emotion-based embeddings (audio, text, video) from transcript data.
    """

    path_csv: str
    """Path to the classified CSV containing interventions."""

    path_json: str
    """Path to the LEVEL_3 JSON with temporal information."""

    audio_file_path: str
    """Path to the full audio file."""

    audio_emotion_analyzer: Optional[AudioEmotionAnalyzer] = None
    """Audio emotion embedding model."""

    text_emotion_analyzer: Optional[TextEmotionAnalyzer] = None
    """Text emotion embedding model."""

    video_emmotion_analyzer: Optional[VideoEmotionAnalyzer] = None
    """Video emotion embedding model."""

    def __post_init__(self):
        self.sentences_df = self._extract_sentences_df()
        self.full_audio = AudioSegment.from_mp3(self.audio_file_path)

    def _extract_sentences_df(self) -> pd.DataFrame:
        """Combines sentence-level information from CSV and JSON files."""
        df_csv = pd.read_csv(self.path_csv)

        with open(self.path_json, 'r', encoding='utf-8') as f:
            data_json = json.load(f)

        # Extract sentences from interventions and align it with timestamps
        frases_json = []
        for speaker in data_json.get("speakers", []):
            words = speaker.get("words", [])
            times = speaker.get("start_times", [])
            speaker_name = ((speaker or {}).get("speaker_info") or {}).get("name", "")

            frase = ""
            tiempos = []
            for palabra, tiempo in zip(words, times):
                frase += palabra + " "
                tiempos.append(tiempo)
                if re.match(r".*[\\.!?]$", palabra):
                    frases_json.append({
                        "speaker": speaker_name,
                        "text": frase.strip(),
                        "start_time": tiempos[0],
                        "end_time": tiempos[-1]
                    })
                    frase = ""
                    tiempos = []

        df_json = pd.DataFrame(frases_json)

        # Split interventions from CSV into individual sentences
        frases_expandidas = []
        for _, row in df_csv.iterrows():
            frases = re.split(r'(?<=[\.!?])\s+', row["text"])
            for frase in frases:
                if frase.strip():
                    nueva_fila = row.to_dict()
                    nueva_fila["text"] = frase.strip()
                    nueva_fila["intervention_id"] = row["intervention_id"]
                    frases_expandidas.append(nueva_fila)

        df_expandido = pd.DataFrame(frases_expandidas)

        # Align with JSON using minimum length
        min_len = min(len(df_expandido), len(df_json))
        df_expandido = df_expandido.iloc[:min_len].copy()
        df_json = df_json.iloc[:min_len].copy()

        df_expandido["start_time"] = df_json["start_time"].values
        df_expandido["end_time"] = df_json["end_time"].values
        df_expandido["Pair"] = df_expandido["Pair"].fillna("Monologue")

        return df_expandido[df_expandido["classification"].isin(["Monologue", "Question", "Answer"])]\
                         .reset_index(drop=True)

    def cortar_audio_temporal(self, start_time: int, end_time: int) -> Optional[tempfile.NamedTemporaryFile]:
        """
        Cuts a segment of the full MP3 audio and returns it as a temporary WAV file.

        Args:
            start_time: Start time in seconds.
            end_time: End time in seconds.

        Returns:
            A temporary WAV file or None on error.
        """
        try:
            start_ms = int(start_time * 1000)
            end_ms = int((end_time + 0.25) * 1000)
            segmento = self.full_audio[start_ms:end_ms]

            temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=True)
            segmento.export(temp_wav.name, format="wav")
            return temp_wav

        except Exception as e:
            print(f"[ERROR] Failed to cut audio segment: {e}")
            return None

    def cortar_video_temporal(self, start_time: int, end_time: int):
        """Placeholder for future video cutting implementation."""
        return None

    def generar_embeddings(self) -> pd.DataFrame:
        """
        Generates audio, text, and video embeddings for each sentence in the transcript.

        Returns:
            A DataFrame with columns: audio_embedding, text_embedding, video_embedding.
        """
        audio_embs = []
        text_embs = []
        video_embs = []

        for _, row in self.sentences_df.iterrows():
            # --- Audio ---
            if self.audio_emotion_analyzer is not None:
                audio_temp = self.cortar_audio_temporal(row['start_time'], row['end_time'])
                if audio_temp is not None:
                    audio_emb = self.audio_emotion_analyzer.get_embeddings(audio_temp.name)
                    audio_embs.append(audio_emb.tolist() if hasattr(audio_emb, 'tolist') else audio_emb)
                else:
                    audio_embs.append(None)
            else:
                audio_embs.append(None)

            # --- Text ---
            if self.text_emotion_analyzer is not None:
                text_emb = self.text_emotion_analyzer.get_embeddings(row['text'])
                text_embs.append(text_emb.tolist() if hasattr(text_emb, 'tolist') else text_emb)
            else:
                text_embs.append(None)

            # --- Video ---
            if self.video_emmotion_analyzer is not None:
                video_temp = self.cortar_video_temporal(row['start_time'], row['end_time'])
                if video_temp is not None:
                    video_emb = self.video_emmotion_analyzer.get_embeddings(video_temp.name)
                    video_embs.append(video_emb.tolist() if hasattr(video_emb, 'tolist') else video_emb)
                else:
                    video_embs.append(None)
            else:
                video_embs.append(None)

        # Add embedding columns to DataFrame
        self.sentences_df["audio_embedding"] = audio_embs
        self.sentences_df["text_embedding"] = text_embs
        self.sentences_df["video_embedding"] = video_embs

        return self.sentences_df
