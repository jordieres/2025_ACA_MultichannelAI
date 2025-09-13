import logging
from dataclasses import dataclass
from typing import List, Dict

import pandas as pd
import torch
from transformers import pipeline

# Initialize logger
logger = logging.getLogger(__name__)


@dataclass
class TextEmotionAnalyzer:
    """
    Recognizes emotions in text using a Hugging Face transformer model.

    Supports:
    - Full probability distribution (for emotion embeddings)
    - Top emotion label (mapped to standard format)
    - Classification of DataFrames

    Attributes:
        model_name: HF model to use for classification.
        device: 'cuda' or 'cpu'.
    """
    model_name: str = "j-hartmann/emotion-english-distilroberta-base"
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __post_init__(self):
        logger.info(f"Loading text emotion model: {self.model_name} on {self.device}")
        self.classifier = pipeline(
            task="text-classification",
            model=self.model_name,
            device=0 if self.device == "cuda" else -1,
            top_k=None,
            framework="pt"
        )

        # Map model-specific labels to standard format used across modalities
        self.label_map = {
            "anger": "angry",
            "disgust": "disgust",
            "fear": "fear",
            "joy": "happy",
            "neutral": "neutral",
            "sadness": "sad",
            "surprise": "surprise"
        }

    def predict(self, text: str) -> List[Dict[str, float]]:
        """
        Returns full emotion probability distribution for a given text.

        Args:
            text: Input sentence or phrase.

        Returns:
            List of dicts with mapped label and probability score.
        """
        raw_preds = self.classifier([text])[0]
        return [
            {"label": self.label_map.get(pred["label"], pred["label"]), "score": pred["score"]}
            for pred in raw_preds
        ]

    def get_top_emotion(self, text: str) -> str:
        """
        Returns the top predicted emotion label for the given text.

        Args:
            text: Input sentence.

        Returns:
            The most likely emotion label (standard format).
        """
        predictions = self.classifier([text])[0]
        top_prediction = max(predictions, key=lambda x: x['score'])
        mapped_label = self.label_map.get(top_prediction['label'], top_prediction['label'])
        logger.debug(f"Top emotion for text: {mapped_label}")
        return mapped_label

    def get_embeddings(self, text: str) -> torch.Tensor:
        """
        Returns centered logits (interpreted as emotion embeddings) for the given text.

        The tensor is centered by subtracting the mean log-probability.

        Args:
            text: Input sentence.

        Returns:
            Tensor of centered logits (length = number of emotion labels).
        """
        output = self.classifier([text])[0]
        probs = torch.tensor([item['score'] for item in output])
        logits = torch.log(probs)
        centered_logits = logits - logits.mean()
        logger.debug(f"Centered logits: {centered_logits}")
        return centered_logits

    def classify_dataframe(self, df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        """
        Classifies a column of text data and adds a new column with top predicted emotions.

        Args:
            df: Input DataFrame with text data.
            text_column: Column containing text to analyze.

        Returns:
            DataFrame with an additional 'classification' column.
        """
        if text_column not in df.columns:
            raise ValueError(f"'{text_column}' column not found in DataFrame.")
        logger.info(f"Classifying {len(df)} text entries using top emotion...")
        df['classification'] = df[text_column].apply(self.get_top_emotion)
        return df
