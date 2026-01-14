from typing import Dict, List, Tuple
import ast
import pandas as pd

class TopicLabeler:
    def __init__(self, topic_kw_model):
        self.topic_kw_model = topic_kw_model  # KeyBERT()

    @staticmethod
    def _generate_label(words: List[str]) -> str:
        if not words:
            return "Undefined"
        label = " ".join(words[:3]).title().replace("_", " ").strip()
        return label

    def _generate_semantic_label(self, words: List[str]) -> str:
        text = " ".join(words)
        keywords = self.topic_kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            top_n=1
        )
        return keywords[0][0].title() if keywords else "General"

    def add_topic_labels(self, topic_info: pd.DataFrame, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, str]]:
        topics_df = topic_info.copy()

        if isinstance(topics_df["Representation"].iloc[0], str):
            topics_df["Representation"] = topics_df["Representation"].apply(ast.literal_eval)

        topics_df["semantic_label"] = topics_df["Representation"].apply(self._generate_semantic_label)

        topic_labels = dict(zip(topics_df["Topic"], topics_df["semantic_label"]))

        df = df.copy()
        df["topic_label"] = df["topic_id"].map(topic_labels)
        return df, topic_labels

