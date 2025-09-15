from dataclasses import dataclass
from typing import Optional
import pandas as pd
import json
import os

from multimodal_fin.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TranscriptPreprocessor:
    """
    Preprocesses a conference transcript by identifying the beginning of the Q&A section
    and labeling each row as either 'prepared_remarks' or 'q_a'.
    """

    section_col: str = "Conf_Section"
    """Name of the column to write the section labels."""

    text_col: str = "text"
    """Column containing the transcript text."""

    qna_key: str = "questions_and_answers"
    """Key used to extract the Q&A intro from the JSON metadata."""

    def extract_qna_intro(self, json_path: str) -> Optional[str]:
        """
        Extracts the first sentence of the Q&A section from the metadata JSON.

        Args:
            json_path: Path to the JSON metadata file.

        Returns:
            The first sentence of the Q&A intro, or None if not found or file is invalid.
        """
        if not os.path.exists(json_path):
            logger.warning(f"JSON path does not exist: {json_path}")
            return None

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    logger.warning(f"Empty JSON content at: {json_path}")
                    return None
                data = json.loads(content)
                qna_text = data.get(self.qna_key)
                if isinstance(qna_text, str):
                    intro = qna_text.split(".")[0].strip()
                    logger.debug(f"Extracted Q&A intro: {intro}")
                    return intro
        except Exception as e:
            logger.error(f"Error reading {json_path}: {e}", exc_info=True)

        return None

    def preprocess(self, csv_path: str, json_path: str) -> pd.DataFrame:
        """
        Labels each row in the transcript CSV as either 'prepared_remarks' or 'q_a'
        based on the location of the Q&A intro.

        Args:
            csv_path: Path to the transcript CSV file.
            json_path: Path to the metadata JSON file.

        Returns:
            A DataFrame with an added column (`section_col`) containing the section labels.
        """
        logger.info(f"Preprocessing transcript from {csv_path} with metadata {json_path}")
        df = pd.read_csv(csv_path)

        qna_intro = self.extract_qna_intro(json_path)

        if qna_intro and self.text_col in df.columns:
            match = df[df[self.text_col].str.contains(qna_intro, case=False, na=False)]
            if not match.empty:
                qna_start_index = match.index[0]
                logger.info(f"Q&A section starts at index: {qna_start_index}")
                df[self.section_col] = [
                    'prepared_remarks' if i < qna_start_index else 'q_a' for i in df.index
                ]
            else:
                logger.warning("Q&A intro not found in transcript. Defaulting all to 'prepared_remarks'.")
                df[self.section_col] = 'prepared_remarks'
        else:
            logger.warning("No valid Q&A intro found or missing text column. Defaulting all to 'prepared_remarks'.")
            df[self.section_col] = 'prepared_remarks'

        logger.info("Preprocessing completed.")
        return df
