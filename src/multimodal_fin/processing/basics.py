from dataclasses import dataclass
from typing import List, Optional, Callable, Tuple
from collections import Counter

import ollama

from multimodal_fin.utils.logging import get_logger
logger = get_logger(__name__)

@dataclass
class LLMClient:
    """Client wrapper for interacting with Ollama models via the chat API.

    This class provides:
      - Automatic model name normalization.
      - Automatic model download if not available locally.
      - Configurable Ollama server host.
    """

    model: str
    host: Optional[str] = "http://127.0.0.1:11500"

    def __post_init__(self) -> None:
        """Initialize client, normalize model name and ensure the model is available."""
        logger.debug("Initializing LLMClient with model='%s', host='%s'", self.model, self.host)
        self.client = ollama.Client(host=self.host) if self.host else ollama
        self.model = self._normalize_model_name(self.model)
        logger.debug("Normalized model name: %s", self.model)
        self._ensure_model()

    def _normalize_model_name(self, model_name: str) -> str:
        """Append ':latest' if the model name does not include a tag.

        Args:
            model_name (str): Raw model name provided by the user.

        Returns:
            str: Normalized model name with tag.
        """
        if ':' in model_name:
            return model_name
        logger.debug("Model '%s' does not include a tag, appending ':latest'", model_name)
        return f"{model_name}:latest"

    def _ensure_model(self) -> None:
        """Check if the model is available locally; if not, download it."""
        try:
            available_models = [m.model for m in self.client.list().models]
            logger.debug("Available models: %s", available_models)
            if self.model not in available_models:
                logger.info("Model '%s' not found locally. Downloading...", self.model)
                self.client.pull(self.model)
                logger.info("Model successfully downloaded: %s", self.model)
            else:
                logger.debug("Model '%s' is already available locally.", self.model)
        except Exception as e:
            logger.error("Failed to check or download model '%s': %s", self.model, str(e))
            raise

    def chat(self, messages: List[dict], schema: Optional[str] = None) -> str:
        """Send a list of messages to the model and retrieve the response.

        Args:
            messages (List[dict]): List of message dictionaries in Ollama format.
            schema (Optional[str]): JSON schema to enforce structured responses.

        Returns:
            str: The content string of the model's response.
        """
        logger.debug("Sending chat request to model '%s' with schema=%s", self.model, schema)
        try:
            response = (
                self.client.chat(model=self.model, messages=messages, format=schema, options={"temperature": 0})
                if schema else
                self.client.chat(model=self.model, messages=messages, options={"temperature": 0})
            )
            logger.debug("Received response from model '%s'", self.model)
            return response.message.content
        except Exception as e:
            logger.error("Chat request to model '%s' failed: %s", self.model, str(e))
            raise



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