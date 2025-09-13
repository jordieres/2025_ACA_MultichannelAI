from dataclasses import dataclass, field
from typing import Dict
import torch
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForImageClassification, AutoConfig

from multimodal_fin.processing.multimodal.video.recognizers.base import EmotionRecognizer


@dataclass
class VITRecognizer(EmotionRecognizer):
    """
    Emotion recognizer using a ViT-based face expression classification model.
    """

    model_name: str = "trpakov/vit-face-expression"
    extractor: AutoFeatureExtractor = field(init=False)
    model: AutoModelForImageClassification = field(init=False)
    id2label: Dict[int, str] = field(init=False)

    def __post_init__(self):
        """
        Loads the feature extractor, model, and label mapping.
        """
        super().__init__(device=self.device)

        self.extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
        self.model = AutoModelForImageClassification.from_pretrained(self.model_name).to(self.device)
        config = AutoConfig.from_pretrained(self.model_name)
        self.id2label = config.id2label

    def predict_emotion(self, face: Image.Image) -> Dict[str, float]:
        """
        Predict emotion scores for a given facial image using a ViT model.

        Args:
            face (Image.Image): Cropped face image.

        Returns:
            Dict[str, float]: Mapping of emotion label to probability.
        """
        inputs = self.extractor(images=face, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        return {self.id2label[i]: float(prob) for i, prob in enumerate(probabilities)}
