from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np
import torch
from PIL import Image
from fer import FER

from multimodal_fin.processing.multimodal.video.recognizers.base import EmotionRecognizer


@dataclass
class FERRecognizer(EmotionRecognizer):
    """
    Emotion recognizer using the FER library with MTCNN face detection.
    """

    fer_detector: FER = field(default_factory=lambda: FER(mtcnn=True))

    def predict_emotion(self, face: Image.Image) -> Optional[Dict[str, float]]:
        """
        Predict emotion scores for a given facial image using the FER library.

        Args:
            face (Image.Image): Cropped face image.

        Returns:
            Optional[Dict[str, float]]: Dictionary of emotion scores or None if detection failed.
        """
        # Convert PIL to OpenCV format (BGR)
        img_cv = np.array(face)[:, :, ::-1]
        detections = self.fer_detector.detect_emotions(img_cv)
        return detections[0]['emotions'] if detections else None
