from dataclasses import dataclass
from pydantic import BaseModel
from typing import List, Literal, Optional

import pandas as pd
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


# @dataclass
# class QAAnalyzer:
#     model_name: str = "llama3"

#     def __post_init__(self):
#         self.prompt_builder = PromptBuilder()
#         self.llm = LLMClient(self.model_name)

#     def analize_qa(self, intervention: str, response: str):
#         messages = self.prompt_builder.analize_qa(intervention, response)
#         response = self.llm.chat(messages, schema=InterventionAnalysis.model_json_schema())

#         return json.loads(response)
    
#     def get_pred(self, intervention: str, response: str) -> Optional[str]:
#         try:
#             result = self.analize_qa(intervention, response)
#             evaluations = result.get('evaluations', [])
#             if len(evaluations) == 1:
#                 return evaluations[0]['answered']
#         except Exception as e:
#             print(f"Error processing intervention: {intervention[:30]}... -> {e}")
#         return None  # si no hay una única evaluación o hay error, se devuelve None

#     def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
#         def classify_row(row):
#             return self.get_pred(row['intervention'], row['response'])

#         df['classification'] = df.apply(classify_row, axis=1)
#         return df

import difflib
def find_best_eval(question, evaluations):
    best_match = None
    best_ratio = 0.0
    for eval in evaluations:
        ratio = difflib.SequenceMatcher(None, question.lower(), eval.get('question', '').lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = eval
    return best_match

@dataclass
class QAAnalyzer:
    model_name: str = "llama3"

    def __post_init__(self):
        self.prompt_builder = PromptBuilder()
        self.llm = LLMClient(self.model_name)

    def analize_qa(self, intervention: str, response: str):
        messages = self.prompt_builder.analize_qa(intervention, response)
        response = self.llm.chat(messages, schema=InterventionAnalysis.model_json_schema())
        return json.loads(response)

    def get_pred(self, intervention: str, response: str) -> Optional[str]:
        try:
            result = self.analize_qa(intervention, response)
            evaluations = result.get('evaluations', [])
            if len(evaluations) == 1:
                return evaluations[0]['answered']
        except Exception as e:
            print(f"Error processing intervention: {intervention[:30]}... -> {e}")
        return None

    def analize_qa_question(self, question: str, response: str):
        messages = self.prompt_builder.analize_qa(question, response)
        response = self.llm.chat(messages, schema=InterventionAnalysis.model_json_schema())
        return json.loads(response)

    def get_pred_question(self, question: str, response: str) -> Optional[str]:
        try:
            result = self.analize_qa_question(question, response)
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

    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        def classify_row(row):
            return self.get_pred_question(row['intervention'], row['response'])
        df['classification'] = df.apply(classify_row, axis=1)
        return df
    
