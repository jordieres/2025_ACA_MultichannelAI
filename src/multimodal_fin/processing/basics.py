from dataclasses import dataclass
from typing import List, Optional, Callable, Tuple
from collections import Counter

import ollama

from multimodal_fin.utils.logging import get_logger
logger = get_logger(__name__)


@dataclass
class LLMClient:
    """Client wrapper for interacting with Ollama models via chat API.

    Automatically normalizes and downloads the model if it's not available locally.
    """
    model: str

    def __post_init__(self) -> None:
        """Post-initialization: normalize model name and ensure it's downloaded."""
        self.model = self._normalize_model_name(self.model)
        self._ensure_model()

    def _normalize_model_name(self, model_name: str) -> str:
        """Appends ':latest' to the model name if no tag is provided.

        Args:
            model_name: Raw model name provided by the user.

        Returns:
            Normalized model name with tag.
        """
        return model_name if ':' in model_name else f"{model_name}:latest"

    def _ensure_model(self) -> None:
        """Checks if the model is available locally; if not, downloads it."""
        available_models = [m.model for m in ollama.list().models]
        if self.model not in available_models:
            logger.info(f"Model '{self.model}' not found locally. Downloading...")
            ollama.pull(self.model)
            logger.info(f"Model downloaded: {self.model}")

    def chat(self, messages: List[dict], schema: Optional[str] = None) -> str:
        """Sends a list of messages to the model and retrieves the response.

        Args:
            messages: List of message dictionaries in Ollama format.
            schema: Optional schema to enforce structured responses.

        Returns:
            The content string of the model's response.
        """
        response = (
            ollama.chat(model=self.model, messages=messages, format=schema, options={"temperature": 0})
            if schema else
            ollama.chat(model=self.model, messages=messages, options={"temperature": 0})
        )
        return response.message.content


class UncertaintyMixin:
    """Provides uncertainty estimation via majority voting."""

    def get_result_and_uncertainty(
        self,
        predict_fn: Callable[[str], str],
        text: str,
        n: int = 5
    ) -> Tuple[str, float]:
        """Estimates category and confidence using majority voting.

        Args:
            predict_fn: Prediction function to apply repeatedly.
            text: The input text to classify.
            n: Number of evaluations to perform.

        Returns:
            A tuple with:
              - The most frequent predicted category.
              - Confidence score as percentage.
        """
        predictions = [predict_fn(text) for _ in range(n)]
        counter = Counter(predictions)
        top_cat, top_freq = counter.most_common(1)[0]

        confidence = round((top_freq / n) * 100, 2)
        logger.debug(f"Predictions: {predictions}")
        logger.debug(f"Top category: {top_cat} with confidence: {confidence}%")

        return top_cat, confidence