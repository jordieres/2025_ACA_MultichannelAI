from dataclasses import dataclass
import pandas as pd
import logging
from PIL import Image
from typing import List

from multimodal_fin.processing.multimodal.video.face_detector import FaceDetector
from multimodal_fin.processing.multimodal.video.recognizers.base import EmotionRecognizer
from multimodal_fin.processing.multimodal.video.recognizers.fer import FERRecognizer
from multimodal_fin.processing.multimodal.video.recognizers.vit import VITRecognizer
from multimodal_fin.processing.multimodal.video.recognizers.emotieff import EmotiEffRecognizer
from multimodal_fin.processing.multimodal.video.processor import VideoProcessor


from multimodal_fin.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EmotionVideoAnalyzer:
    """
    Coordinates the facial emotion analysis from video frames.
    """

    recognizer: EmotionRecognizer
    """Model-specific emotion recognizer."""

    face_detector: FaceDetector
    """Face detector to crop faces from frames."""

    processor: VideoProcessor
    """Frame processor for sampling frames from video."""

    def analyze_video_frames(self, frames: List) -> pd.DataFrame:
        """
        Analyzes each frame to detect faces and classify their emotions.

        Args:
            frames (List): A list of video frames (np.ndarray or PIL-compatible).

        Returns:
            pd.DataFrame: Frame-wise emotion probabilities.
        """
        predictions = []

        if isinstance(self.recognizer, EmotiEffRecognizer):
            logger.info("Using EmotiEff recognizer with batch-capable input.")
            for frame in frames:
                faces = self.face_detector.recognize_faces(frame)
                emotion = self.recognizer.predict_emotion(faces)
                if emotion:
                    predictions.append(emotion)
        elif isinstance(self.recognizer, (FERRecognizer, VITRecognizer)):
            logger.info(f"Using recognizer: {type(self.recognizer).__name__}")
            for frame in frames:
                pil_image = Image.fromarray(frame)
                face = self.face_detector.detect_faces(pil_image)
                if face:
                    emotion = self.recognizer.predict_emotion(face)
                    if emotion:
                        predictions.append(emotion)
        else:
            logger.error("Unsupported recognizer type.")
            raise TypeError("Recognizer must be an instance of FERRecognizer, VITRecognizer, or EmotiEffRecognizer.")

        if not predictions:
            logger.warning("No valid emotion predictions found.")
            return pd.DataFrame()

        logger.info(f"Collected {len(predictions)} emotion predictions.")
        return pd.DataFrame(predictions)

    def analyze_video(self, video_path: str) -> pd.DataFrame:
        """
        Extracts frames and runs analysis on a given video file.

        Args:
            video_path (str): Path to the video file.

        Returns:
            pd.DataFrame: DataFrame of emotion probabilities.
        """
        logger.info(f"Analyzing video: {video_path}")
        frames = self.processor.extract_frames(video_path)
        return self.analyze_video_frames(frames)
