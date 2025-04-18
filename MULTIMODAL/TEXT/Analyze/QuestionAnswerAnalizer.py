from dataclasses import dataclass
from pydantic import BaseModel
from typing import List, Literal, Optional

import pandas as pd
import difflib
import json

from ..Basics import LLMClient, UncertaintyMixin
from ..Prompt_builder import PromptBuilder


class EvaluatedQA(BaseModel):
    question: str
    answered: Literal['yes', 'partially', 'no']
    answer_summary: str = None
    answer_quote: str = None

class InterventionAnalysis(BaseModel):
    intervention: str
    response: str
    evaluations: List[EvaluatedQA]


@dataclass
class QAAnalyzer(UncertaintyMixin):
    model_name: str = "llama3"
    NUM_EVALUATIONS=1

    def __post_init__(self):
        self.prompt_builder = PromptBuilder()
        self.llm = LLMClient(self.model_name)

    def analize_qa(self, intervention: str, response: str):
        messages = self.prompt_builder.analize_qa(intervention, response)
        response = self.llm.chat(messages, schema=InterventionAnalysis.model_json_schema())
        return json.loads(response)
    
    def get_pred(self, question: str, response: str):
        predicted_categories = [self.get_pred_question(question, response) for _ in range(self.NUM_EVALUATIONS)]
        return self.get_result_and_uncertainty(lambda _: predicted_categories.pop(0), text, self.NUM_EVALUATIONS)

    def get_pred_question(self, question: str, response: str) -> Optional[str]:
        try:
            result = self.analize_qa(question, response)
            evaluations = result.get('evaluations', [])

            if not evaluations:
                print("⚠️ No evaluations returned")
                return None

            # Si solo hay una, usarla directamente
            if len(evaluations) == 1:
                return evaluations[0]['answered']

            # Buscar la evaluación más parecida a la pregunta original
            best_match = max(
                evaluations,
                key=lambda ev: difflib.SequenceMatcher(None, question.lower(), ev.get("question", "").lower()).ratio()
            )

            similarity = difflib.SequenceMatcher(None, question.lower(), best_match.get("question", "").lower()).ratio()
            print(f"🔍 Best match similarity: {similarity:.2f} -> '{best_match['question']}'")

            return best_match.get("answered")

        except Exception as e:
            print(f"❌ Error processing question: {question[:30]}... -> {e}")
            return None
    
    def evaluate_qa_model(self, data: list):
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
    
