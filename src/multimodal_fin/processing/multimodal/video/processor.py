from dataclasses import dataclass
import numpy as np

from multimodal_fin.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class VideoProcessor:
    """
    Handles video processing tasks like frame extraction and frame sampling.
    """

    skips: float = 0.1
    """Proportion of frames to retain from the video (0 < skips <= 1)."""

    def reduce_video_frames(self, video_data: list) -> list:
        """
        Reduces the number of frames in a video by uniform sampling.

        Args:
            video_data (list): List of all video frames as numpy arrays.

        Returns:
            list: Sampled list of frames.

        Raises:
            ValueError: If skips is not within the valid range or results in zero frames.
        """
        if not (0 < self.skips <= 1):
            raise ValueError("The 'skips' value must be a proportion between 0 and 1.")

        total_frames = len(video_data)
        num_selected = int(total_frames * self.skips)

        if num_selected < 1:
            raise ValueError("The selected proportion is too small, resulting in zero frames.")

        sampled_frames = [video_data[i] for i in np.linspace(0, total_frames - 1, num_selected, dtype=int)]
        logger.info(f"Reduced {total_frames} frames to {len(sampled_frames)} sampled frames.")
        return sampled_frames

    def extract_frames(self, video_path: str) -> list:
        """
        Extracts frames from a video file. Implementation is placeholder.

        Args:
            video_path (str): Path to the video file.

        Returns:
            list: List of video frames (to be implemented).
        """
        logger.warning("Frame extraction from video file is not yet implemented.")
        raise NotImplementedError("Use an external extractor (e.g., MoviePy) and pass frames manually.")
