from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
import torch
import logging

logger = logging.getLogger(__name__)


@dataclass
class FaceDetector:
    """
    Detects and crops faces from input images or video frames using MTCNN.
    """
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    mtcnn: MTCNN = field(init=False)

    def __post_init__(self):
        self.mtcnn = MTCNN(keep_all=False, post_process=True, device=self.device)

    def detect_faces(self, image: Image.Image) -> Optional[Image.Image]:
        """
        Detects a single face in the given PIL image.

        Args:
            image (Image.Image): Input PIL image.

        Returns:
            Optional[Image.Image]: Cropped face image or None.
        """
        boxes, _ = self.mtcnn.detect(image)
        if boxes is not None:
            x1, y1, x2, y2 = map(int, boxes[0])
            logger.debug(f"Detected face at: {(x1, y1, x2, y2)}")
            return image.crop((x1, y1, x2, y2))
        return None

    def recognize_faces(self, frame: np.ndarray) -> List[np.ndarray]:
        """
        Detects multiple faces in a video frame.

        Args:
            frame (np.ndarray): Input frame (BGR or RGB format).

        Returns:
            List[np.ndarray]: List of cropped face arrays.
        """
        boxes, probs = self.mtcnn.detect(frame)
        if boxes is None or probs is None:
            return []

        selected = boxes[probs > 0.9]
        faces = []
        for box in selected:
            x1, y1, x2, y2 = map(int, box)
            face = frame[y1:y2, x1:x2]
            faces.append(face)

        logger.debug(f"{len(faces)} faces detected in frame.")
        return faces
