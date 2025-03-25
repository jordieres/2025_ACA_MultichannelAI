from dataclasses import dataclass, field
from typing import Dict
import pandas as pd
import torch

from abc import ABC, abstractmethod
from funasr import AutoModel


# Clase base para cualquier reconocedor de emociones a partir de audio
class AudioEmotionRecognizer(ABC):
    
    @abstractmethod
    def predict_from_wav(self, wav_path: str) -> Dict[str, Dict[str, float]]:
        pass

    @abstractmethod
    def get_top_emotion(self, emotion_dict: Dict[str, Dict[str, float]]) -> str:
        pass


@dataclass
class Emotion2VecRecognizer(AudioEmotionRecognizer):
    model_name: str = "iic/emotion2vec_plus_large"
    device: str = "cuda"
    embeddings_output_dir: str = f'/home/aacastro/Alejandro/ACA_MultichanelAI_2025/src/AUDIO/sim_results/emo2vec/{model_name}/'

    def __post_init__(self):
        self.model = AutoModel(model=self.model_name, device=self.device)

    def predict_from_wav(self, wav_path: str) -> Dict[str, Dict[str, float]]:
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
        inner_dict = list(emotion_dict.values())[0]
        top_emotion = max(inner_dict, key=inner_dict.get)
        print("predicted: ", top_emotion)
        return top_emotion
    


@dataclass
class AudioEmotionAnalysis:
    mode: str = "emotion2vec"
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name: str = "iic/emotion2vec_plus_large"
    embeddings_output_dir: str = None

    def __post_init__(self):
        match self.mode:
            case "emotion2vec":
                self.recognizer = Emotion2VecRecognizer(model_name=self.model_name, device=self.device, embeddings_output_dir=self.embeddings_output_dir)
            case _:
                raise ValueError("Unsupported mode. Currently only 'emotion2vec' is supported.")

    def analyze_audio(self, audio_path: str) -> str:
        emotion_dict = self.recognizer.predict_from_wav(audio_path)
        return self.change_disgust_fear(self.recognizer.get_top_emotion(emotion_dict))

    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Path" not in df.columns:
            raise ValueError("DataFrame must contain a 'Path' column with audio file paths.")

        df["classification"] = df["Path"].apply(self.analyze_audio)
        return df
    
    def change_disgust_fear(self, prediction: str) -> str:
        if self.mode in ["emotion2vec"]:
            if prediction == "disgusted":
                return "fearful"
            elif prediction == "fearful":
                return "disgusted"
        return prediction