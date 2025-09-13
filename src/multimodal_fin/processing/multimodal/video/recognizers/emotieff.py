from dataclasses import dataclass, field
from typing import Dict, List

import torch
from multimodal_fin.processing.multimodal.video.recognizers.base import EmotionRecognizer
from emotiefflib.facial_analysis import EmotiEffLibRecognizer


@dataclass
class EmotiEffRecognizer(EmotionRecognizer):
    """
    Emotion recognizer using the EmotiEff library.

    Attributes:
        model (str): Name of the pretrained EmotiEff model.
        emotion_mapping (Dict[str, str]): Maps EmotiEff emotion names to standard labels.
        emotion_labels (List[str]): List of standardized emotion labels in fixed order.
    """

    model: str = "enet_b0_8_best_afew"
    emotion_mapping: Dict[str, str] = field(default_factory=lambda: {
        "anger": "angry",
        "happiness": "happy",
        "sadness": "sad",
        "fear": "fear",
        "surprise": "surprise",
        "disgust": "disgust",
        "neutral": "neutral"
    })
    emotion_labels: List[str] = field(default_factory=lambda: [
        "angry", "happy", "sad", "fear", "surprise", "disgust", "neutral"
    ])

    def __post_init__(self):
        self.recognizer = EmotiEffLibRecognizer(
            engine="torch",
            model_name=self.model,
            device=self.device
        )

    def predict_emotion(self, facial_images: List) -> Dict[str, float]:
        """
        Predict emotion probabilities for a list of facial images.

        Args:
            facial_images (List): List of cropped face arrays (as numpy arrays).

        Returns:
            Dict[str, float]: Emotion probabilities mapped to standardized labels.
        """
        if not facial_images:
            return {emotion: 0.0 for emotion in self.emotion_labels}

        _, raw_scores = self.recognizer.predict_emotions(facial_images[0])
        softmax_scores = torch.nn.functional.softmax(torch.tensor(raw_scores), dim=1)

        return {
            self.emotion_labels[i]: float(softmax_scores[0, i])
            for i in range(len(self.emotion_labels))
        }
