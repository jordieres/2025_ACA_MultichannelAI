from dataclasses import dataclass
from typing import Dict

from funasr import AutoModel

from .Basics import AudioEmotionRecognizer


@dataclass
class Emotion2VecRecognizer(AudioEmotionRecognizer):
    model_name: str = "iic/emotion2vec_plus_large"
    device: str = "cuda"
    # embeddings_output_dir: str = f'/home/aacastro/Alejandro/ACA_MultichanelAI_2025/src/AUDIO/sim_results/emo2vec/{model_name}/'

    def __post_init__(self):
        self.model = AutoModel(model=self.model_name, device=self.device)

    def predict_from_wav(self, wav_path: str) -> Dict[str, Dict[str, float]]:
        result = self.model.generate(
            wav_path,
            # output_dir=self.embeddings_output_dir,
            # granularity="utterance",
            extract_embedding=False,
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