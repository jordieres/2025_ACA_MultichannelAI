from typing import Tuple
from tqdm import tqdm
import pandas as pd

from multimodal_fin.analysis_qa_effects.preprocessing.text_preprocessor import TextPreprocessor

class TopicModeler:
    def __init__(self, preproc: TextPreprocessor, topic_model):
        self.preproc = preproc
        self.topic_model = topic_model  # BERTopic ya instanciado fuera

    def add_topics(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # tqdm.pandas(desc="Cleaning texts")
        df = df.copy()
        df["qa_text_clean"] = df["qa_text"].apply(self.preproc.clean_text_spacy)
        # df["qa_text_clean"] = df["qa_text"].progress_apply(self.preproc.clean_text_spacy)

        topics, probs = self.topic_model.fit_transform(df["qa_text_clean"])
        df["topic_id"] = topics
        df["topic_prob"] = probs

        topic_info = self.topic_model.get_topic_info()
        return df, topic_info
