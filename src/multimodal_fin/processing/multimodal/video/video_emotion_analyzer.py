from dataclasses import dataclass
import pandas as pd
import torch
import logging

from multimodal_fin.processing.multimodal.video.analyzer import EmotionVideoAnalyzer
from multimodal_fin.processing.multimodal.video.face_detector import FaceDetector
from multimodal_fin.processing.multimodal.video.processor import VideoProcessor
from multimodal_fin.processing.multimodal.video.recognizers.vit import VITRecognizer
from multimodal_fin.processing.multimodal.video.recognizers.fer import FERRecognizer
from multimodal_fin.processing.multimodal.video.recognizers.emotieff import EmotiEffRecognizer

logger = logging.getLogger(__name__)


@dataclass
class VideoEmotionAnalyzer:
    """
    High-level video emotion classification pipeline.

    This class orchestrates the process of detecting faces,
    recognizing emotions per frame, and aggregating predictions
    to produce a single dominant emotion for the full video.

    Attributes:
        mode (str): Recognition model type ('vit', 'fer', 'emotieff').
        skips (float): Fraction of frames to process.
        method (str): Aggregation strategy ('mode', 'mean', 'abs').
        device (str): Device to use ('cuda' or 'cpu').
        emotieff_model (str): Model name for EmotiEffRecognizer.
    """
    mode: str
    skips: float = 0.1
    method: str = "mode"
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    emotieff_model: str = "enet_b0_8_best_afew"

    def __post_init__(self):
        logger.info(f"Initializing video analysis with mode={self.mode}, device={self.device}")

        self.face_detector = FaceDetector(device=self.device)
        self.video_processor = VideoProcessor(skips=self.skips)

        match self.mode:
            case "vit":
                self.recognizer = VITRecognizer(device=self.device)
            case "fer":
                self.recognizer = FERRecognizer(device=self.device)
            case "emotieff":
                self.recognizer = EmotiEffRecognizer(device=self.device, model=self.emotieff_model)
            case _:
                raise ValueError("Unsupported mode. Choose from: 'vit', 'fer', 'emotieff'.")

        self.analyzer = EmotionVideoAnalyzer(
            recognizer=self.recognizer,
            face_detector=self.face_detector,
            processor=self.video_processor
        )

    def analyze_video(self, video_path: str) -> str:
        """
        Runs emotion recognition on a full video.

        Args:
            video_path (str): Path to the video file.

        Returns:
            str: Predicted dominant emotion.
        """
        df = self.analyzer.analyze_video(video_path)

        if df.empty:
            logger.warning("No predictions were made from the video.")
            return 'unknown'

        prediction = self.get_aggregated_prediction(df)
        if self.mode != "emotieff":
            prediction = self.swap_disgust_fear(prediction)

        logger.info(f"Predicted emotion: {prediction}")
        return prediction

    def get_aggregated_prediction(self, df: pd.DataFrame) -> str:
        """
        Aggregates frame-level predictions using the selected strategy.

        Args:
            df (pd.DataFrame): DataFrame of frame-wise emotion probabilities.

        Returns:
            str: Final predicted emotion.
        """
        match self.method:
            case "mode":
                return df.idxmax(axis=1).mode()[0]
            case "mean":
                return df.mean().idxmax()
            case "abs":
                return df.max().idxmax()
            case _:
                raise ValueError(f"Unsupported aggregation method: {self.method}")

    def swap_disgust_fear(self, emotion: str) -> str:
        """
        Optionally swaps 'disgust' and 'fear' to align with common misclassifications.

        Args:
            emotion (str): The predicted emotion.

        Returns:
            str: Possibly corrected emotion.
        """
        if emotion == "disgust":
            return "fear"
        elif emotion == "fear":
            return "disgust"
        return emotion

    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies video emotion classification to each path in a DataFrame.

        Args:
            df (pd.DataFrame): Must contain a 'Path' column with video paths.

        Returns:
            pd.DataFrame: Same DataFrame with an added 'classification' column.
        """
        if "Path" not in df.columns:
            raise ValueError("DataFrame must contain a 'Path' column.")

        logger.info(f"Classifying {len(df)} videos...")
        df["classification"] = df["Path"].apply(self.analyze_video)
        return df

    def get_embeddings(self, video_path: str):
        """
        Placeholder for extracting emotion embeddings from video.

        Args:
            video_path (str): Path to video file.

        Returns:
            torch.Tensor: Emotion embedding (future implementation).
        """
        raise NotImplementedError("Embedding extraction is not yet implemented.")
