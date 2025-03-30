from dataclasses import dataclass, field
from typing import List

import pandas as pd
import os

from .QAClassifier import QAClassifier
from .TranscriptPreprocessor import TranscriptPreprocessor

@dataclass
class EnsembleQAClassifier:
    model_names: List[str] = field(default_factory=lambda: ['llama3', 'mistral', 'mixtral'])
    NUM_EVALUATIONS: int = 5
    output_path: str = 'output'
    preprocessor = TranscriptPreprocessor()

    def __post_init__(self):
        self.classifiers = [
            QAClassifier(model=name, NUM_EVALUATIONS=self.NUM_EVALUATIONS)
            for name in self.model_names
        ]

    def ensemble_predict(self, text: str):
        results = [
                (cat, conf)
                for clf in self.classifiers
                for cat, conf in [clf.get_pred(text)]
                if not print(f"[{clf.model}] Predicted: {cat} | Confidence: {conf:.2f}%")
            ]

        # Sumamos confianza por categoría
        conf_sum = {}
        for cat, conf in results:
            conf_sum[cat] = conf_sum.get(cat, 0.0) + conf

        best_cat, total_conf = max(conf_sum.items(), key=lambda x: x[1])
        avg_conf = round(total_conf / len(self.classifiers), 2)

        print(f"\n========== Resultado combinado ==========")
        print(f"✅ Final prediction: {best_cat} | Combined confidence: {avg_conf:.2f}%")
        print('')
        print('='*100)
        print('')

        return best_cat, avg_conf

    # def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
    #     print('')
    #     preds = df['text'].apply(lambda text: self.ensemble_predict(text))
    #     df['classification'] = preds.apply(lambda x: x[0])
    #     df['global_confidence'] = preds.apply(lambda x: x[1])
    #     return df
    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        print('')

        # Inicializar columnas con valores por defecto
        df['classification'] = 'Presentation'
        df['global_confidence'] = 100.0

        # Filtrar solo las filas con sección 'q_a'
        qna_mask = df['Conf_Section'] == 'q_a'

        if qna_mask.any():
            preds = df.loc[qna_mask, 'text'].apply(lambda text: self.ensemble_predict(text))
            df.loc[qna_mask, 'classification'] = preds.apply(lambda x: x[0])
            df.loc[qna_mask, 'global_confidence'] = preds.apply(lambda x: x[1])

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

    def _generate_output_path(self, file_path: str) -> str:
        parts = file_path.strip('/').split('/')
        company = parts[-4] if len(parts) >= 4 else 'unknown_company'
        year = parts[-3] if len(parts) >= 3 else 'unknown_year'
        filename = parts[-2] + '.csv' if len(parts) >= 2 else os.path.basename(file_path)

        output_dir = os.path.join(self.output_path, company, year)
        os.makedirs(output_dir, exist_ok=True)

        return os.path.join(output_dir, filename)

    def save_to_csv(self, df: pd.DataFrame, output_path: str) -> None:
        try:
            df.to_csv(output_path, index=False)
        except Exception as e:
            raise ValueError(f"Failed to save DataFrame to {output_path}: {e}")

    def expand_df(self, file_path: str, preprocessed_df: pd.DataFrame) -> pd.DataFrame:
        output_path = self._generate_output_path(file_path)
        # df = pd.read_csv(file_path)
        df = self.classify_dataframe(preprocessed_df)
        df = self.annotate_question_answer_pairs(df)
        self.save_to_csv(df, output_path)
        print('[OK] File saved')
        return df

    def process_all_csvs(self, input_directory: str) -> None:
        for root, _, files in os.walk(input_directory):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    json_path = os.path.join(root, "LEVEL_4.json")

                    try:
                        df = self.preprocessor.preprocess(file_path, json_path)
                        self.expand_df(file_path, df)
                        print(f"[OK] File processed {file_path}")
                    except Exception as e:
                        print(f"[ERROR] Failed to process {file_path}: {e}")