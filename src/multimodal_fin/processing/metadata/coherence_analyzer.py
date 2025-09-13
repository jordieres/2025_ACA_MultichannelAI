from dataclasses import dataclass
from pydantic import BaseModel
from typing import List
import json
import ollama

from multimodal_fin.processing.metadata.prompt_builder import PromptBuilder


class ContradictionDetail(BaseModel):
    """Represents a contradiction found between a monologue and a response."""
    monologue_excerpt: str
    response_excerpt: str
    explanation: str


class CoherenceAnalysis(BaseModel):
    """Represents the logical coherence analysis between a monologue and a response."""
    topic_covered: bool
    consistent: bool
    summary: str
    contradictions: List[ContradictionDetail]


@dataclass
class CoherenceAnalyzer:
    """
    Analyzes the logical coherence between a monologue and a response using an LLM model.

    Attributes:
        model_name (str): The name of the model used for inference (default is "llama3").
    """
    model_name: str = "llama3"

    def __post_init__(self):
        """Initializes the prompt builder used to generate system/user messages."""
        self.prompt_builder = PromptBuilder()

    def analyze_coherence(self, monologue: str, response: str) -> dict:
        """
        Evaluates whether a given response is topically and logically coherent with a preceding monologue.

        The analysis determines:
          - If the response addresses a topic mentioned in the monologue.
          - If the response is logically consistent with the content of the monologue.
          - If contradictions exist, it identifies and explains them.

        Args:
            monologue (str): The original monologue text (e.g., CEO/CFO statements).
            response (str): The follow-up response text (e.g., Q&A answers).

        Returns:
            dict: A dictionary following the CoherenceAnalysis schema.
        """
        messages = self.prompt_builder.check_coherence(monologue, response)

        result = ollama.chat(
            model=self.model_name,
            messages=messages,
            format=CoherenceAnalysis.model_json_schema(),
            options={'temperature': 0}
        )

        return json.loads(result.message.content)
