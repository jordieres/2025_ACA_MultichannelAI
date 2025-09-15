from dataclasses import dataclass
from pydantic import BaseModel
from typing import List, Literal, Optional
import pandas as pd
import difflib
import json

from multimodal_fin.processing.basics import LLMClient, UncertaintyMixin
from multimodal_fin.processing.metadata.prompt_builder import PromptBuilder


class EvaluatedQA(BaseModel):
    """Represents a single question from an intervention and its evaluation."""
    question: str
    answered: Literal['yes', 'partially', 'no']
    answer_summary: Optional[str] = None
    answer_quote: Optional[str] = None


class InterventionAnalysis(BaseModel):
    """Represents the full QA analysis for an intervention."""
    intervention: str
    response: str
    evaluations: List[EvaluatedQA]


@dataclass
class QAAnalyzer(UncertaintyMixin):
    """Analyzes Q&A interactions by evaluating whether questions were answered in responses."""

    model_name: str = "llama3"
    """Name of the LLM to use."""

    NUM_EVALUATIONS: int = 5
    """Number of LLM passes to estimate uncertainty."""

    def __post_init__(self):
        """Initializes the LLM client and prompt builder."""
        self.prompt_builder = PromptBuilder()
        self.llm = LLMClient(self.model_name)

    def analize_qa(self, intervention: str, response: str) -> dict:
        """Runs the LLM to analyze whether each question in the intervention is answered in the response.

        Args:
            intervention (str): The text containing one or more questions.
            response (str): The response from the speaker.

        Returns:
            dict: The parsed LLM response following `InterventionAnalysis` schema.
        """
        messages = self.prompt_builder.analize_qa(intervention, response)
        response = self.llm.chat(messages, schema=InterventionAnalysis.model_json_schema())
        return json.loads(response)

    def get_pred(self, question: str, response: str) -> tuple:
        """Performs multiple evaluations to determine the QA label and uncertainty.

        Args:
            question (str): A single question to evaluate.
            response (str): The response to evaluate against.

        Returns:
            tuple: (predicted_label, confidence, extra_info_dict)
        """
        raw_outputs = []
        labels = []

        for _ in range(self.NUM_EVALUATIONS):
            result = self.analize_qa(question, response)
            label = self._extract_best_match_label(question, result)
            if label:
                labels.append(label)
                raw_outputs.append(result)

        if not labels:
            return None, 0.0, {}

        final_label, confidence = self.get_result_and_uncertainty(
            lambda _: labels.pop(0), question, len(raw_outputs)
        )

        return final_label, confidence, {
            "raw_outputs": raw_outputs
        }

    def _extract_best_match_label(self, question: str, result: dict) -> Optional[str]:
        """Finds the most relevant question in the LLM evaluation and returns its answer status.

        Args:
            question (str): The original question.
            result (dict): The LLM output in dictionary form.

        Returns:
            Optional[str]: One of ['yes', 'partially', 'no'] or None if nothing is matched.
        """
        evaluations = result.get('evaluations', [])
        if not evaluations:
            return None

        if len(evaluations) == 1:
            return evaluations[0]['answered']

        best_match = max(
            evaluations,
            key=lambda ev: difflib.SequenceMatcher(
                None, question.lower(), ev.get("question", "").lower()
            ).ratio()
        )
        return best_match.get("answered")

    def evaluate_qa_model(self, data: list) -> pd.DataFrame:
        """Evaluates a QA model on a dataset of interventions with ground truth.

        Args:
            data (list): A list of dicts, each containing a 'response' and list of 'label' dicts with true Q&A labels.

        Returns:
            pd.DataFrame: DataFrame with predicted and true labels per question.
        """
        results = []

        for example in data:
            response = example['response']
            for q in example['label']:
                question = q['question']
                true_label = q['answered']
                pred_label = self.get_pred_question(question, response)

                if pred_label is not None:
                    results.append({
                        "question": question,
                        "response": response,
                        "label": true_label,
                        "classification": pred_label
                    })

        return pd.DataFrame(results)

    def get_pred_question(self, question: str, response: str) -> Optional[str]:
        """Returns the answer status for a given question-response pair using a single pass.

        Args:
            question (str): The question to check.
            response (str): The response to evaluate.

        Returns:
            Optional[str]: One of ['yes', 'partially', 'no'] or None if failed.
        """
        try:
            result = self.analize_qa(question, response)
            evaluations = result.get('evaluations', [])

            if not evaluations:
                print("⚠️ No evaluations returned")
                return None

            if len(evaluations) == 1:
                return evaluations[0]['answered']

            best_match = max(
                evaluations,
                key=lambda ev: difflib.SequenceMatcher(
                    None, question.lower(), ev.get("question", "").lower()
                ).ratio()
            )

            similarity = difflib.SequenceMatcher(
                None, question.lower(), best_match.get("question", "").lower()
            ).ratio()
            print(f"🔍 Best match similarity: {similarity:.2f} -> '{best_match['question']}'")

            return best_match.get("answered")

        except Exception as e:
            print(f"❌ Error processing question: {question[:30]}... -> {e}")
            return None
