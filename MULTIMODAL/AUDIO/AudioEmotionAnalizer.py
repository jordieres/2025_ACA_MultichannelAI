from dataclasses import dataclass, field
from typing import Dict
import pandas as pd
import torch

from abc import ABC, abstractmethod
from funasr import AutoModel


class AudioEmotionRecognizer(ABC):
    """
    Abstract base class for any emotion recognizer that operates on audio input.
    All audio-based emotion recognition models should inherit from this class.
    """

    @abstractmethod
    def predict_from_wav(self, wav_path: str) -> Dict[str, Dict[str, float]]:
        """
        Predicts emotion probabilities from a WAV audio file.

        Args:
            wav_path (str): Path to the WAV audio file.

        Returns:
            Dict[str, Dict[str, float]]: A dictionary mapping each audio segment key to
            another dictionary of emotion labels and their associated scores.
        """
        pass

    @abstractmethod
    def get_top_emotion(self, emotion_dict: Dict[str, Dict[str, float]]) -> str:
        """
        Retrieves the top predicted emotion label from a dictionary of emotion scores.

        Args:
            emotion_dict (Dict[str, Dict[str, float]]): Emotion predictions probabilities per segment.

        Returns:
            str: The label of the most probable emotion.
        """
        pass


@dataclass
class Emotion2VecRecognizer(AudioEmotionRecognizer):
    """
    Concrete implementation of AudioEmotionRecognizer using the Emotion2Vec model.

    Attributes:
        model_name (str): Name of the pretrained Emotion2Vec model.
        device (str): Device to run the model on ("cuda" or "cpu").
        embeddings_output_dir (str): Directory where embeddings will be saved.
    """
    model_name: str = "iic/emotion2vec_plus_large"
    device: str = "cuda"
    embeddings_output_dir: str = f'/home/aacastro/Alejandro/ACA_MultichanelAI_2025/src/AUDIO/sim_results/emo2vec/{model_name}/'

    def __post_init__(self):
        """
        Initializes the model after dataclass attributes are set.
        """
        self.model = AutoModel(model=self.model_name, device=self.device)

    def predict_from_wav(self, wav_path: str) -> Dict[str, Dict[str, float]]:
        """
        Predicts emotion scores from a WAV audio file using the Emotion2Vec model.

        Args:
            wav_path (str): Path to the WAV file.

        Returns:
            Dict[str, Dict[str, float]]: Mapping from utterance key to emotion scores.
        """
        result = self.model.generate(
            wav_path,
            output_dir=self.embeddings_output_dir,
            granularity="utterance",
            extract_embedding=True,
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

    def get_top_emotion(self, emotion_dict):
        """
        Extracts the most probable emotion label from the predicted scores.

        Args:
            emotion_dict (Dict[str, Dict[str, float]]): Emotion scores per utterance.

        Returns:
            str: The top predicted emotion label.
        """
        inner_dict = list(emotion_dict.values())[0]
        top_emotion = max(inner_dict, key=inner_dict.get)
        print("predicted: ", top_emotion)
        return top_emotion
    


@dataclass
class AudioEmotionAnalysis:
    """
    General-purpose class to manage emotion recognition pipelines from audio files.

    Attributes:
        mode (str): Recognition mode to use (currently only "emotion2vec" is supported).
        device (str): Device on which the model will run ("cuda" or "cpu").
        model_name (str): Model identifier string for loading.
        embeddings_output_dir (str): Directory to store intermediate embeddings (optional).
    """
    mode: str = "emotion2vec"
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name: str = "iic/emotion2vec_plus_large"
    embeddings_output_dir: str = None

    def __post_init__(self):
        """
        Initializes the appropriate recognizer based on the selected mode.
        """
        match self.mode:
            case "emotion2vec":
                self.recognizer = Emotion2VecRecognizer(model_name=self.model_name, device=self.device, embeddings_output_dir=self.embeddings_output_dir)
            case _:
                raise ValueError("Unsupported mode. Currently only 'emotion2vec' is supported.")

    def analyze_audio(self, audio_path: str) -> str:
        """
        Predicts the dominant emotion for a given audio file.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            str: The dominant emotion label, possibly adjusted post-prediction.
        """
        emotion_dict = self.recognizer.predict_from_wav(audio_path)
        return self.change_disgust_fear(self.recognizer.get_top_emotion(emotion_dict))

    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies audio emotion classification to a DataFrame containing audio paths.

        Args:
            df (pd.DataFrame): A DataFrame with a 'Path' column pointing to audio files.

        Returns:
            pd.DataFrame: The same DataFrame with a new 'classification' column.
        """
        if "Path" not in df.columns:
            raise ValueError("DataFrame must contain a 'Path' column with audio file paths.")

        df["classification"] = df["Path"].apply(self.analyze_audio)
        return df
    
    def change_disgust_fear(self, prediction: str) -> str:
        """
        Swaps 'disgusted' and 'fearful' labels to handle possible model label confusion.

        Args:
            prediction (str): The original predicted emotion label.

        Returns:
            str: Adjusted label if applicable; otherwise returns the original.
        """
        if self.mode in ["emotion2vec"]:
            if prediction == "disgusted":
                return "fearful"
            elif prediction == "fearful":
                return "disgusted"
        return prediction