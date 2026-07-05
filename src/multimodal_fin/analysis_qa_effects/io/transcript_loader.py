import os, json
import pandas as pd

from multimodal_fin.analysis_qa_effects.config import PipelineConfig
from multimodal_fin.analysis_qa_effects.features.emotion_aggregator import EmotionAggregator

class TranscriptQALoader:
    def __init__(self, config: PipelineConfig, emo_agg: EmotionAggregator):
        self.config = config
        self.emo_agg = emo_agg

    def load_company(self, company: str) -> pd.DataFrame:
        root_path = os.path.join(self.config.processed_root, company)
        records = []

        for root, _, files in os.walk(root_path):
            for file in files:
                if file != "transcript.json":
                    continue

                conf_path = os.path.join(root, file)
                conference_id = os.path.basename(os.path.dirname(conf_path))             # Q1, Q2, etc.
                year = os.path.basename(os.path.dirname(os.path.dirname(conf_path)))    # 2024, etc.

                with open(conf_path, "r") as f:
                    data = json.load(f)

                # tu lógica: saltas la primera key
                for pair in list(data.keys())[1:]:
                    q = data[pair]["question"]
                    a = data[pair]["answer"]
                    ans_cat = data[pair]["qa_response_classification"]["Predicted_category"]
                    conf = data[pair]["qa_response_classification"]["Confidence"]
                    q_topic_10k = data[pair]['question_topic_classification']["Predicted_category"]
                    a_topic_10k = data[pair]['answer_topic_classification']["Predicted_category"]

                    mm = data[pair].get("multimodal_embeddings", {}).get("answer", None)
                    if mm is not None:
                        audio_mean, text_mean, video_mean = self.emo_agg.mean_emotions(mm)
                        audio_mean = audio_mean.tolist() if audio_mean is not None else None
                        text_mean  = text_mean.tolist()  if text_mean  is not None else None
                        video_mean = video_mean.tolist() if video_mean is not None else None
                        audio_sentences = mm.get("audio", [])  # lista de listas (n_sentences, 7)
                    else:
                        audio_mean = text_mean = video_mean = None

                    coherence_analyses = data[pair].get("coherence_analyses", [])

                    records.append({
                        "company": company,
                        "year": year,
                        "conference": conference_id,
                        "pair_id": pair,
                        "question": q,
                        "answer": a,
                        "answered": str(ans_cat).lower(),
                        "q_topic_10k": q_topic_10k,
                        "a_topic_10k": a_topic_10k,
                        "confidence": conf,
                        "qa_text": f"{q} {a}",
                        "audio_emo_mean": audio_mean,
                        "text_emo_mean": text_mean,
                        "video_emo_mean": video_mean,
                        "audio_sentences": audio_sentences,
                        "coherence_analyses": coherence_analyses,
                    })

        df = pd.DataFrame(records)
        print(f"Loaded {len(df)} QA pairs from {df[['year','conference']].drop_duplicates().shape[0]} conferences for company {company}.")
        return df