from dataclasses import dataclass, field
from typing import List

import pandas as pd
import json
import time
import os

from MULTIMODAL.TEXT.Analyze.EnsembleInterventionAnalyzer import EnsembleInterventionAnalyzer
from MULTIMODAL.TEXT.Classify.EnsembleInterventionClassifier import EnsembleInterventionClassifier


@dataclass
class ConferenceProcessor:
    input_csv_path: str = 'file_paths.csv'
    qa_models: List[str] = field(default_factory=lambda: ['llama3', 'llama3.1:8b', 'phi4'])
    monologue_models: List[str] = field(default_factory=lambda: ['llama3', 'llama3.1:8b', 'phi4'])
    sec10k_models: List[str] = field(default_factory=lambda: ['llama3', 'llama3.1:8b', 'phi4'])
    evals: int = 3
    verbose: int = 1

    def __post_init__(self):
        self.classifier = EnsembleInterventionClassifier(
            qa_model_names=self.qa_models,
            monologue_model_names=self.monologue_models,
            NUM_EVALUATIONS=self.evals,
            verbose=self.verbose
        )

        self.analyzer = EnsembleInterventionAnalyzer(
            sec10k_model_names=self.sec10k_models,
            NUM_EVALUATIONS=self.evals,
            verbose=self.verbose
        )

    def run(self):

        df_paths = pd.read_csv(self.input_csv_path)
        if 'path' not in df_paths.columns:
            raise ValueError("El CSV debe contener una columna 'path'")

        for original_path in df_paths['path']:
            file_path = os.path.join(original_path, 'transcript.csv')
            json_path = os.path.join(original_path, 'LEVEL_4.json')

            if not os.path.exists(file_path) or not os.path.exists(json_path):
                print(f"[WARN] Saltando: No se encontraron transcript.csv o LEVEL_4.json en {original_path}")
                continue

            # print("[INFO] Processing file with path: ", original_path)
            self._print(f"[INFO] Processing file with path: {original_path}")

            # === Output path ===
            processed_path = original_path.replace("companies", "processed_companies")
            os.makedirs(processed_path, exist_ok=True)

            output_csv_path = os.path.join(processed_path, "transcript.csv")
            output_json_path = os.path.join(processed_path, "transcript.json")

            try:

                start_time = time.time()

                # === Classification ===
                df = self.classifier.preprocessor.preprocess(file_path, json_path)
                df = self.classifier.classify_dataframe(df)
                df = self.classifier.annotate_question_answer_pairs(df)
                df.to_csv(output_csv_path, index=False)

                # === Analysis ===
                self.analyzer.initialize_multimodal_model(output_csv_path, original_path)
                output = self.analyzer.generate_structured_output()

                elapsed_time = time.time() - start_time
                output["processing_time_seconds"] = round(elapsed_time, 2)
                with open(output_json_path, 'w') as f:
                    json.dump(output, f, indent=4)

                self._print(f"[OK] Procesado y guardado en: {processed_path} (Tiempo: {round(elapsed_time, 2)}s)")

            except Exception as e:
                print(f"[ERROR] Fallo en {file_path}: {e}\n")

    
    def _print(self, *args, **kwargs):
        if self.verbose >= 1:
            print(*args, **kwargs)

    def _print_header(self, title: str):
        if self.verbose >= 1:
            print(f"\n{'='*10} {title} {'='*10}")