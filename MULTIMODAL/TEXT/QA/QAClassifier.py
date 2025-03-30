from dataclasses import dataclass, field
from typing import Literal, Optional, List

import pandas as pd
import json
import os

from ..Basics import LLMClient, BaseModel, UncertaintyMixin
from ..Prompt_builder import PromptBuilder


class CategoryQA(BaseModel):
    category: Literal['Question', 'Answer', 'Procedure']

@dataclass
class QAClassifier(UncertaintyMixin):
    model: str = 'llama3'
    NUM_EVALUATIONS: int = 5
    output_path: str = 'output'

    def __post_init__(self):
        self.llm = LLMClient(self.model)

    def classify_text(self, text: str) -> str:
        messages = PromptBuilder.prompt_qa(text)
        response = self.llm.chat(messages, schema=CategoryQA.model_json_schema())
        return json.loads(response)['category']

    def get_pred(self, text: str):
        predicted_categories = [self.classify_text(text) for _ in range(self.NUM_EVALUATIONS)]
        return self.get_result_and_uncertainty(lambda _: predicted_categories.pop(0), text, n=self.NUM_EVALUATIONS)

    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df['classification'] = df['text'].apply(lambda text: self.get_pred(text)[0])
        return df

    def annotate_question_answer_pairs(self, df: pd.DataFrame) -> pd.DataFrame:
        pair_id = 1
        current_question_row = None
        pairs = []

        for index, row in df.iterrows():
            if row['classification'] == "Question":
                current_question_row = index
                pairs.append(None)
            elif row['classification'] == "Answer" and current_question_row is not None:
                pairs[current_question_row] = f"pair_{pair_id}"
                pairs.append(f"pair_{pair_id}")
                pair_id += 1
                current_question_row = None
            else:
                pairs.append(None)

        df['Pair'] = pairs
        pair_counts = df['Pair'].value_counts(dropna=True)
        invalid_pairs = pair_counts[pair_counts != 2]

        if not invalid_pairs.empty:
            raise ValueError(f"Error: The following pairs do not have exactly 2 observations:\n{invalid_pairs.to_dict()}")

        return df

    def save_to_csv(self, df: pd.DataFrame, output_path: str) -> None:
        try:
            df.to_csv(output_path, index=False)
        except Exception as e:
            raise ValueError(f"Failed to save DataFrame to {output_path}: {e}")

    def _generate_output_path(self, file_path: str) -> str:
        parts = file_path.strip('/').split('/')
        company = parts[-4] if len(parts) >= 4 else 'unknown_company'
        year = parts[-3] if len(parts) >= 3 else 'unknown_year'
        filename = parts[-2] + '.csv' if len(parts) >= 2 else os.path.basename(file_path)

        output_dir = os.path.join(self.output_path, company, year)
        os.makedirs(output_dir, exist_ok=True)

        return os.path.join(output_dir, filename)

    def expand_df(self, file_path: str) -> pd.DataFrame:
        output_path = self._generate_output_path(file_path)
        df = self.classify_dataframe(pd.read_csv(file_path))
        df = self.annotate_question_answer_pairs(df)
        self.save_to_csv(df, output_path)
        print('[OK] File saved')
        return df

    def process_all_csvs(self, input_directory: str) -> None:
        for root, _, files in os.walk(input_directory):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    try:
                        self.expand_df(file_path)
                        print(f"[OK] File processed {file_path}")
                    except Exception as e:
                        print(f"[ERROR] Failed to process {file_path}: {e}")