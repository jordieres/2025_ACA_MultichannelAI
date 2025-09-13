from dataclasses import dataclass
from typing import Dict
import logging
import pandas as pd
import torch

from multimodal_fin.processing.multimodal.audio.recognizers.base import AudioEmotionRecognizer
from multimodal_fin.processing.multimodal.audio.recognizers.emotion2vec import Emotion2VecRecognizer

logger = logging.getLogger(__name__)


@dataclass
class AudioEmotionAnalyzer:
    """
    Extracts emotion-based audio embeddings or emotion classifications using a specified recognizer.

    Attributes:
        mode (str): The name of the recognition model type. Currently, only 'emotion2vec' is supported.
        device (str): The computation device to use ('cuda' or 'cpu').
        model_name (str): Name or path of the model to be loaded.
    """
    mode: str = "emotion2vec"
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name: str = "iic/emotion2vec_plus_large"

    def __post_init__(self):
        match self.mode:
            case "emotion2vec":
                self.recognizer: AudioEmotionRecognizer = Emotion2VecRecognizer(
                    model_name=self.model_name,
                    device=self.device
                )
            case _:
                raise ValueError(f"Unsupported mode '{self.mode}'. Only 'emotion2vec' is supported.")

    def classify_audio(self, audio_path: str) -> str:
        """Returns the top predicted emotion for a given audio file."""
        emotion_dict = self.recognizer.predict_from_wav(audio_path)
        top_emotion = self.recognizer.get_top_emotion(emotion_dict)
        return self._swap_disgust_fear(top_emotion)

    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a 'classification' column to a DataFrame by predicting emotions from audio file paths.

        Args:
            df (pd.DataFrame): Must contain a 'Path' column with paths to audio files.

        Returns:
            pd.DataFrame: The same DataFrame with a new 'classification' column.
        """
        if "Path" not in df.columns:
            raise ValueError("DataFrame must contain a 'Path' column with audio file paths.")

        df["classification"] = df["Path"].apply(self.classify_audio)
        return df

    def _swap_disgust_fear(self, emotion: str) -> str:
        """
        Swaps 'disgusted' and 'fearful' if mode is 'emotion2vec',
        to address known model misclassifications.
        """
        if self.mode == "emotion2vec":
            if emotion == "disgusted":
                return "fearful"
            elif emotion == "fearful":
                return "disgusted"
        return emotion

    def get_embeddings(self, audio_path: str) -> torch.Tensor:
        """
        Returns a centered logits vector representing emotional content from the given audio file.

        The vector is ordered as:
            ['happy', 'neutral', 'surprise', 'disgust', 'anger', 'sadness', 'fear']

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            torch.Tensor: Centered logits vector of emotion scores.
        """
        emotion_dict = self.recognizer.predict_from_wav(audio_path)

        # Define label mapping and standard order
        standard_order = ['happy', 'neutral', 'surprise', 'disgust', 'anger', 'sadness', 'fear']
        label_map = {
            'happy': 'happy',
            'neutral': 'neutral',
            'surprised': 'surprise',
            'disgusted': 'disgust',
            'angry': 'anger',
            'sad': 'sadness',
            'fearful': 'fear',
            'other': None
        }
        inverse_map = {v: k for k, v in label_map.items() if v is not None}

        try:
            ordered_probs = [list(emotion_dict.values())[0][inverse_map[label]] for label in standard_order]
        except KeyError as e:
            raise ValueError(f"Missing expected emotion in predictions: {e}")

        probs_tensor = torch.tensor(ordered_probs)
        logits = torch.log(probs_tensor)
        return logits - logits.mean()
