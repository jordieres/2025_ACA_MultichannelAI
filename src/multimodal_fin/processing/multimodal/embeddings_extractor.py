import os
import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from multimodal_fin.processing.multimodal.audio.audio_emotion_analyzer import AudioEmotionAnalyzer
from multimodal_fin.processing.multimodal.video.video_emotion_analyzer import VideoEmotionAnalyzer
from multimodal_fin.processing.multimodal.text.text_emotion_analyzer import TextEmotionAnalyzer

from multimodal_fin.processing.multimodal.multimodal_embeddings import MultimodalEmbeddings

from multimodal_fin.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EmbeddingsExtractor:
    """
    Extracts multimodal emotion embeddings from a CSV of conference interventions.

    The extractor supports:
      - Audio emotion embeddings via `AudioEmotionAnalyzer`.
      - Text emotion embeddings via `TextEmotionAnalyzer`.
      - Video emotion embeddings via `VideoEmotionAnalyzer`.
    """

    audio_model_name: Optional[str] = None
    """Name of the model used for audio emotion recognition."""

    text_model_name: Optional[str] = None
    """Name of the model used for text emotion recognition."""

    video_model_name: Optional[str] = None
    """Name of the model used for video emotion recognition."""

    device: str = "cpu"
    """Computation device (e.g., 'cpu', 'cuda')."""

    verbose: int = 1
    """Verbosity level for logging."""

    def __post_init__(self):
        """Initializes emotion analyzers based on selected modalities."""
        self.audio_emotion = (
            AudioEmotionAnalyzer(model_name=self.audio_model_name, device=self.device)
            if self.audio_model_name else None
        )
        self.text_emotion = (
            TextEmotionAnalyzer(model_name=self.text_model_name, device=self.device)
            if self.text_model_name else None
        )
        self.video_emotion = (
            VideoEmotionAnalyzer(mode=self.video_model_name, device=self.device)
            if self.video_model_name else None
        )

        if self.verbose >= 2:
            logger.debug(f"Initialized EmbeddingsExtractor with device='{self.device}'")

    def extract(self, csv_path: str, original_dir: str) -> pd.DataFrame:
        """
        Loads the classified interventions CSV and computes multimodal embeddings.

        Args:
            csv_path: Path to the CSV file containing interventions.
            original_dir: Directory with associated media files and metadata (LEVEL_3.json, audio.mp3).

        Returns:
            A pandas DataFrame with added columns for each modality's embeddings.
        """
        logger.info(f"Starting embedding extraction for: {csv_path}")
        logger.debug(f"Original directory: {original_dir}")

        # Construct required paths
        path_json = os.path.join(original_dir, "LEVEL_3.json")
        path_audio = os.path.join(original_dir, "audio.mp3")

        # Initialize the embedding module
        self.multimodal = MultimodalEmbeddings(
            path_csv=csv_path,
            path_json=path_json,
            audio_file_path=path_audio,
            audio_emotion_analyzer=self.audio_emotion,
            text_emotion_analyzer=self.text_emotion,
            video_emmotion_analyzer=self.video_emotion
        )

        if self.verbose:
            logger.info("Generating multimodal embeddings...")

        # Run the pipeline
        self.multimodal.generar_embeddings()

        logger.info("Embedding extraction complete.")
        return self.multimodal.sentences_df
