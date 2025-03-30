from dataclasses import dataclass, field
from typing import Literal, Optional, List

import pandas as pd 
import json 
import os


from ..Basics import LLMClient, BaseModel, UncertaintyMixin
from ..Prompt_builder import PromptBuilder

class Category10K(BaseModel):
    category: Literal[
        'Business', 'Risk Factors', 'Selected Financial Data', 'MD&A',
        'Financial Statements and Supplementary Data', 'Other'
    ]

@dataclass
class Form10KClassifier(UncertaintyMixin):
    model: str = 'llama3'
    NUM_EVALUATIONS: int = 10

    def __post_init__(self):
        self.llm = LLMClient(self.model)

    def classify_text(self, text: str) -> str:
        messages = PromptBuilder.prompt_10k(text)
        response = self.llm.chat(messages, schema=Category10K.model_json_schema())
        return json.loads((response))['category']

    def explain_other_category(self, text: str) -> str:
        messages = PromptBuilder.explain_why_other(text)
        return self.llm.chat(messages)

    def get_pred(self, text: str):
        predicted_categories = [self.classify_text(text) for _ in range(self.NUM_EVALUATIONS)]
        return self.get_result_and_uncertainty(lambda _: predicted_categories.pop(0), text)

    def get_pairs_from_csv(self, df: pd.DataFrame) -> dict:
        df = df[df['classification'].isin(['Question', 'Answer'])]
        result = {}
        for pair, group in df.groupby('Pair'):
            question = group.loc[group['classification'] == 'Question', 'text'].values
            answer = group.loc[group['classification'] == 'Answer', 'text'].values
            result[pair] = {
                'Question': question[0] if len(question) else None,
                'Answer': answer[0] if len(answer) else None
            }
        return dict(sorted(result.items(), key=lambda p: int(p[0].split('_')[1])))

    def process_text(self, text: str, pairs: dict, pair_key: str, q_a: str):
        result = self.get_pred(text)
        print(f'Predicted category for {q_a}: {result[0]} | Confidence: {result[1]}')
        key = f"{q_a.lower()}_pred"
        pairs[pair_key][key] = {'Predicted_category': result[0], 'Confidence': result[1]}

        if result[0] == 'Other':
            explanation_key = f"why_other_{q_a.lower()}"
            pairs[pair_key][explanation_key] = self.explain_other_category(text)

        return pairs

    def classify_pairs(self, pairs: dict) -> dict:
        for pair_key, pair_value in pairs.items():
            question = pair_value.get("Question", "No question available")
            answer = pair_value.get("Answer", "No answer available")
            pairs = self.process_text(question, pairs, pair_key, 'Question')
            pairs = self.process_text(answer, pairs, pair_key, 'Answer')
            print('-'*50)
        return pairs

    def save_json_from_csv(self, file_path: str, classified_pairs: dict):
        json_path = os.path.splitext(file_path)[0] + ".json"
        with open(json_path, 'w') as json_file:
            json.dump(classified_pairs, json_file, indent=4)

    def process_all_csvs(self, annotated_csv_path: str):
        for root, _, files in os.walk(annotated_csv_path):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    try:
                        pairs = self.get_pairs_from_csv(pd.read_csv(file_path))
                        classified_pairs = self.classify_pairs(pairs)
                        self.save_json_from_csv(file_path, classified_pairs)
                        print(f"[OK] File processed and saved {file_path}")
                    except Exception as e:
                        print(f"[ERROR]Failed to process {file_path}: {e}")