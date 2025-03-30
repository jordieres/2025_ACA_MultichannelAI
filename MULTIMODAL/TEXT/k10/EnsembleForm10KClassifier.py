from dataclasses import dataclass, field
from typing import Literal, Optional, List

import pandas as pd
import json
import os

from .Form10KClassifier import Form10KClassifier


def print_header(title):
    print(f"\n{'='*10} {title} {'='*10}")

@dataclass
class Ensemble10KClassifier:
    model_names: List[str] = field(default_factory=lambda: ['llama3:8b', 'llama3.1:8b', 'phi4'])
    NUM_EVALUATIONS: int = 5

    def __post_init__(self):
        self.classifiers = [
            Form10KClassifier(model=name, NUM_EVALUATIONS=self.NUM_EVALUATIONS)
            for name in self.model_names
        ]

    def ensemble_predict(self, text: str):
        results = []
        print_header("Predicciones individuales")

        for clf in self.classifiers:
            cat, conf = clf.get_pred(text)
            print(f"[{clf.model}] Predicted: {cat} | Confidence: {conf:.1f}%")
            results.append((cat, conf))

        conf_sum = {}
        for cat, conf in results:
            conf_sum[cat] = conf_sum.get(cat, 0.0) + conf

        best_cat, total_conf = max(conf_sum.items(), key=lambda x: x[1])
        avg_conf = round(total_conf / len(self.classifiers), 2)

        print_header("Resultado combinado")
        print(f"✅ Final prediction: {best_cat} | Combined confidence: {avg_conf:.1f}%")

        return best_cat, avg_conf

    def process_text(self, text: str, pairs: dict, pair_key: str, q_a: str):
        print_header(f"Procesando {q_a} del par {pair_key}")
        category, confidence = self.ensemble_predict(text)

        key = f"{q_a.lower()}_pred"
        pairs[pair_key][key] = {'Predicted_category': category, 'Confidence': confidence}

        if category == 'Other':
            explanation_key = f"why_other_{q_a.lower()}"
            pairs[pair_key][explanation_key] = self.classifiers[0].explain_other_category(text)

        return pairs

    def classify_pairs(self, pairs: dict) -> dict:
        for pair_key, pair_value in pairs.items():
            question = pair_value.get("Question", "No question available")
            answer = pair_value.get("Answer", "No answer available")

            pairs = self.process_text(question, pairs, pair_key, 'Question')
            pairs = self.process_text(answer, pairs, pair_key, 'Answer')
            print('-'*50)
        return pairs

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
                        print_header(f"Procesando archivo: {file}")
                        df = pd.read_csv(file_path)
                        pairs = self.get_pairs_from_csv(df)
                        classified_pairs = self.classify_pairs(pairs)
                        self.save_json_from_csv(file_path, classified_pairs)
                        print(f"[OK] Archivo procesado y guardado: {file_path}")
                    except Exception as e:
                        print(f"[ERROR] Fallo al procesar {file_path}: {e}")