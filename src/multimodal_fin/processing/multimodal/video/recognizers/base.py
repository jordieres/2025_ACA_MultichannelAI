from abc import ABC, abstractmethod
from typing import Dict
from PIL import Image
import torch


class EmotionRecognizer(ABC):
    """
    Abstract base class for facial emotion recognition models.
    All emotion recognizers must implement this interface.
    """

    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            device (str): Device to run inference on ('cuda' or 'cpu').
        """
        self.device = device

    @abstractmethod
    def predict_emotion(self, face: Image.Image) -> Dict[str, float]:
        """
        Predict emotion scores for a given facial image.

        Args:
            face (Image.Image): Cropped face image.

        Returns:
            Dict[str, float]: Dictionary mapping emotion labels to probabilities.
        """
        pass
