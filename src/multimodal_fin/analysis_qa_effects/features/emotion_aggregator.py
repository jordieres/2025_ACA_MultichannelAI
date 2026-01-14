from typing import Optional, Tuple
import numpy as np

class EmotionAggregator:
    @staticmethod
    def mean_emotions(mm: dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        audio_raw = mm.get("audio", [])
        audio = np.array(audio_raw, dtype=np.float32) if audio_raw else None
        audio_mean = audio.mean(axis=0) if isinstance(audio, np.ndarray) and audio.size > 0 else None

        text_raw = mm.get("text", [])
        text = np.array(text_raw, dtype=np.float32) if text_raw else None
        text_mean = text.mean(axis=0) if isinstance(text, np.ndarray) and text.size > 0 else None

        video_raw = mm.get("video", None)
        if video_raw is None:
            video_mean = None
        else:
            video = np.array(video_raw, dtype=np.float32)
            video_mean = video.mean(axis=0) if video.size > 0 else None

        return audio_mean, text_mean, video_mean