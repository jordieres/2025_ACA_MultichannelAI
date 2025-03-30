from dataclasses import dataclass, field
from typing import Literal, Optional, List

from TEXT.POO.scripts.Form10k.EnsembleForm10KClassifier import Ensemble10KClassifier
from TEXT.POO.scripts.QA.EnsembleQAClassifier import EnsembleQAClassifier
from TEXT.POO.scripts.QA.QAClassifier import QAClassifier

@dataclass
class ClassificationManager:
    mode: Literal['qa', '10k']
    input_path: str
    output_path: Optional[str] = None
    models: List[str] = field(default_factory=lambda: ['llama3', 'llama3.1:8b', 'phi4'])
    evals: Optional[int] = None  # Si se indica, se usa EnsembleQAClassifier

    def run(self):
        if self.mode == 'qa':
            print("[INFO] Starting QA classification...")

            if self.evals:  # Usar ensemble si se indica número de repeticiones
                print(f"[INFO] Using ensemble QA classifier with {self.evals} evaluations per model.")
                classifier = EnsembleQAClassifier(
                    model_names=self.models,
                    NUM_EVALUATIONS=self.evals,
                    output_path=self.output_path or 'output'
                )
            else:
                print(f"[INFO] Using single QA classifier: {self.models[0]}")
                print('')
                classifier = QAClassifier(
                    model=self.models[0],
                    NUM_EVALUATIONS=1,  # predicción directa sin repetición
                    output_path=self.output_path or 'output'
                )

            classifier.process_all_csvs(self.input_path)

        elif self.mode == '10k':
            print("[INFO] Starting 10-K classification with ensemble...")
            classifier = Ensemble10KClassifier(
                model_names=self.models,
                NUM_EVALUATIONS=self.evals or 5,
            )
            classifier.process_all_csvs(self.input_path)

        else:
            raise ValueError("Mode must be either 'qa' or '10k'")