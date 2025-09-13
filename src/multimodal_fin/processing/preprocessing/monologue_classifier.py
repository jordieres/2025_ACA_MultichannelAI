from dataclasses import dataclass
from typing import Literal, Tuple

import json
import pandas as pd
from pydantic import BaseModel

from multimodal_fin.processing.basics import LLMClient, UncertaintyMixin
from multimodal_fin.processing.metadata.prompt_builder import PromptBuilder

import logging
logger = logging.getLogger(__name__)


class CategoryPresentation(BaseModel):
    """Pydantic schema for monologue classification output."""
    category: Literal['Monologue', 'Procedure']


@dataclass
class MonologueClassifier(UncertaintyMixin):
    """Classifier for identifying monologue categories using an LLM and uncertainty estimation."""
    
    model: str = 'llama3'
    NUM_EVALUATIONS: int = 5
    output_path: str = 'output'

    def __post_init__(self) -> None:
        """Initializes the LLM client with the specified model."""
        self.llm = LLMClient(self.model)
        logger.info(f"Initialized MonologueClassifier with model: {self.model}")

    def classify_text(self, text: str) -> str:
        """Classifies a single text as either 'Monologue' or 'Procedure'.

        Args:
            text: Input string representing a conference intervention.

        Returns:
            Category string: 'Monologue' or 'Procedure'.
        """
        logger.debug(f"Classifying text: {text[:80]}...")
        messages = PromptBuilder.prompt_monologue(text)
        response = self.llm.chat(messages, schema=CategoryPresentation.model_json_schema())
        try:
            category = json.loads(response)['category']
            logger.debug(f"Predicted category: {category}")
            return category
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse response: {response}", exc_info=True)
            raise ValueError(f"Invalid response from LLM: {response}") from e

    def get_pred(self, text: str) -> Tuple[str, float]:
        """Performs multiple evaluations and computes uncertainty score.

        Args:
            text: Input intervention text.

        Returns:
            A tuple of (predicted_category, confidence_score).
        """
        logger.info("Running ensemble prediction...")
        predicted_categories = [self.classify_text(text) for _ in range(self.NUM_EVALUATIONS)]
        return self.get_result_and_uncertainty(
            predict_fn=lambda _: predicted_categories.pop(0),
            text=text,
            n=self.NUM_EVALUATIONS
        )

    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classifies all interventions in a DataFrame.

        Args:
            df: DataFrame containing a 'text' column with interventions.

        Returns:
            DataFrame with an added 'classification' column.
        """
        logger.info("Classifying monologue dataframe...")
        df['classification'] = df['text'].apply(lambda text: self.get_pred(text)[0])
        return df
