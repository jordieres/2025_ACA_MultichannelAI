import os
import json
from dataclasses import dataclass
from typing import List

import pandas as pd

from multimodal_fin.processing.preprocessing.ensemble_classifier import EnsembleInterventionClassifier
from multimodal_fin.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Preprocessor:
    """
    Handles the full transcript preprocessing pipeline for a financial conference:

    Steps:
        1. Section segmentation between 'prepared_remarks' and 'q_a'.
        2. Classification using ensemble of Q&A and monologue classifiers.
        3. Annotation of question-answer pairs.
    """

    qa_model_names: List[str]
    monologue_model_names: List[str]
    num_evaluations: int = 5
    verbose: int = 1
    section_col: str = "Conf_Section"
    text_col: str = "text"
    qna_key: str = "questions_and_answers"

    def __post_init__(self):
        """Initializes the ensemble classifier used for intervention classification."""
        self.classifier = EnsembleInterventionClassifier(
            qa_model_names=self.qa_model_names,
            monologue_model_names=self.monologue_model_names,
            NUM_EVALUATIONS=self.num_evaluations,
            verbose=self.verbose
        )

    def extract_qna_intro(self, json_path: str) -> str | None:
        """
        Extracts the first sentence of the Q&A section from the provided JSON.

        Args:
            json_path: Path to the LEVEL_4.json file.

        Returns:
            First sentence of Q&A section or None if not found.
        """
        if not os.path.exists(json_path):
            return None
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.loads(f.read() or "{}")
            intro = data.get(self.qna_key)
            if isinstance(intro, str) and intro.strip():
                return intro.split(".")[0].strip()
        except Exception as e:
            logger.warning(f"Failed to read Q&A intro from {json_path}: {e}")
        return None

    def divide_conference(self, csv_path: str, json_path: str) -> pd.DataFrame:
        """
        Assigns sections ('prepared_remarks' or 'q_a') to each row based on intro location.

        Args:
            csv_path: Path to transcript CSV.
            json_path: Path to LEVEL_4.json with Q&A intro.

        Returns:
            DataFrame with new section column.
        """
        df = pd.read_csv(csv_path)
        intro = self.extract_qna_intro(json_path)

        if intro and self.text_col in df.columns:
            mask = df[self.text_col].str.contains(intro, case=False, na=False)
            if mask.any():
                start = mask.idxmax()
                df[self.section_col] = [
                    'prepared_remarks' if i < start else 'q_a' for i in df.index
                ]
                logger.info("Q&A section detected. Split applied.")
            else:
                df[self.section_col] = 'prepared_remarks'
                logger.info("Q&A intro not found. Defaulted to 'prepared_remarks'.")
        else:
            df[self.section_col] = 'prepared_remarks'
            logger.info("No intro extracted. Entire transcript set as 'prepared_remarks'.")
        return df

    def process(self, csv_path: str, json_path: str) -> pd.DataFrame:
        """
        Executes sectioning, classification, and annotation pipeline.

        Args:
            csv_path: Path to transcript CSV.
            json_path: Path to LEVEL_4.json

        Returns:
            Annotated and classified DataFrame.
        """
        df = self.divide_conference(csv_path, json_path)
        df = self.classifier.classify_dataframe(df)
        df = self.classifier.annotate_question_answer_pairs(df)
        return df

    def process_and_save(self, csv_path: str, json_path: str, output_csv_path: str) -> pd.DataFrame:
        """
        Runs the preprocessing pipeline and saves the final DataFrame to CSV.

        Args:
            csv_path: Input transcript CSV.
            json_path: LEVEL_4.json.
            output_csv_path: Path to save the processed CSV.

        Returns:
            Final processed DataFrame.
        """
        df = self.process(csv_path, json_path)
        df.to_csv(output_csv_path, index=False)
        logger.info(f"Processed transcript saved to {output_csv_path}")
        return df
