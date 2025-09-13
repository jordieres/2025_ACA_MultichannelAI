from dataclasses import dataclass
from typing import Dict

from funasr import AutoModel

from multimodal_fin.processing.multimodal.audio.recognizers.base import AudioEmotionRecognizer


@dataclass
class Emotion2VecRecognizer(AudioEmotionRecognizer):
    """
    Implementation of the AudioEmotionRecognizer using the Emotion2Vec model.

    This class wraps the Emotion2Vec family of models available via FunASR.

    Attributes:
        model_name: Name of the Emotion2Vec model.
        device: Device to run the model on ("cuda" or "cpu").
    """
    model_name: str = "iic/emotion2vec_plus_large"
    device: str = "cuda"

    def __post_init__(self):
        self.model = AutoModel(model=self.model_name, device=self.device)

    def predict_from_wav(self, wav_path: str) -> Dict[str, Dict[str, float]]:
        """
        Perform inference on a WAV file and return emotion probabilities.

        Args:
            wav_path: Path to the WAV file.

        Returns:
            Dictionary of probabilities per emotion label.
        """
        result = self.model.generate(
            wav_path,
            extract_embedding=False,
            device=self.device
        )
        return {
            entry["key"]: {
                label.split("/")[-1]: score
                for label, score in zip(entry["labels"], entry["scores"])
                if label.split("/")[-1] != "<unk>"
            }
            for entry in result
        }

    def get_top_emotion(self, emotion_dict: Dict[str, Dict[str, float]]) -> str:
        """
        Identify the most probable emotion from the prediction dictionary.

        Args:
            emotion_dict: Output from `predict_from_wav`.

        Returns:
            The emotion label with the highest probability.
        """
        inner_dict = list(emotion_dict.values())[0]
        top_emotion = max(inner_dict, key=inner_dict.get)
        print("Predicted emotion:", top_emotion)
        return top_emotion
