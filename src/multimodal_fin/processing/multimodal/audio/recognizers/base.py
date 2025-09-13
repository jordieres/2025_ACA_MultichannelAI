from abc import ABC, abstractmethod
from typing import Dict


class AudioEmotionRecognizer(ABC):
    """
    Abstract base class for all audio emotion recognizers.

    All concrete implementations must provide methods to:
    - Predict emotion probabilities from a WAV file.
    - Return the top predicted emotion label.

    Returns:
        A dictionary with structure:
        {
            "<audio_key>": {
                "<emotion_label>": float,  # probability
                ...
            }
        }
    """

    @abstractmethod
    def predict_from_wav(self, wav_path: str) -> Dict[str, Dict[str, float]]:
        """
        Run emotion recognition on a WAV audio file.

        Args:
            wav_path: Path to the WAV file.

        Returns:
            Dictionary of emotion probabilities per audio key.
        """
        pass

    @abstractmethod
    def get_top_emotion(self, emotion_dict: Dict[str, Dict[str, float]]) -> str:
        """
        Extract the most probable emotion from the prediction dictionary.

        Args:
            emotion_dict: Output from `predict_from_wav`.

        Returns:
            The top predicted emotion label as a string.
        """
        pass
