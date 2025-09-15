from dataclasses import dataclass
from typing import Literal, Tuple

import json
import pandas as pd
from pydantic import BaseModel

from multimodal_fin.processing.basics import LLMClient, UncertaintyMixin
from multimodal_fin.processing.metadata.prompt_builder import PromptBuilder

from multimodal_fin.utils.logging import get_logger

logger = get_logger(__name__)


class CategoryQA(BaseModel):
    """Pydantic schema for Q&A classification output."""
    category: Literal['Question', 'Answer', 'Procedure']


@dataclass
class QAClassifier(UncertaintyMixin):
    """
    Classifier for identifying Q&A intervention types using an LLM with uncertainty estimation.
    """

    model: str = "llama3"
    """LLM model name."""

    NUM_EVALUATIONS: int = 5
    """Number of times to sample the model for uncertainty estimation."""
    
    model: str = 'llama3'
    NUM_EVALUATIONS: int = 5

    def __post_init__(self) -> None:
        """Initializes the LLM client."""
        self.llm = LLMClient(self.model)
        logger.info(f"Initialized QAClassifier with model: {self.model}")

    def classify_text(self, text: str) -> str:
        """Classifies a single intervention as 'Question', 'Answer' or 'Procedure'.

        Args:
            text: The input intervention text.

        Returns:
            A string category predicted by the LLM.
        """
        logger.debug(f"Classifying QA text: {text[:80]}...")
        messages = PromptBuilder.prompt_qa(text)
        response = self.llm.chat(messages, schema=CategoryQA.model_json_schema())
        try:
            category = json.loads(response)['category']
            logger.debug(f"Predicted category: {category}")
            return category
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse response: {response}", exc_info=True)
            raise ValueError(f"Invalid response from LLM: {response}") from e

    def get_pred(self, text: str) -> Tuple[str, float]:
        """Performs multiple classifications and computes uncertainty score.

        Args:
            text: Input intervention text.

        Returns:
            A tuple of (predicted_category, confidence_score).
        """
        logger.info("Running ensemble prediction for QA text...")
        predicted_categories = [self.classify_text(text) for _ in range(self.NUM_EVALUATIONS)]
        return self.get_result_and_uncertainty(
            predict_fn=lambda _: predicted_categories.pop(0),
            text=text,
            n=self.NUM_EVALUATIONS
        )

    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classifies all interventions in a DataFrame.

        Args:
            df: DataFrame containing a 'text' column.

        Returns:
            DataFrame with an added 'classification' column.
        """
        logger.info("Classifying QA dataframe...")
        df['classification'] = df['text'].apply(lambda text: self.get_pred(text)[0])
        return df
