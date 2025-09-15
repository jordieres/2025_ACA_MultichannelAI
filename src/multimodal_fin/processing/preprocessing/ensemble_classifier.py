from dataclasses import dataclass
from typing import List, Tuple, Optional

import pandas as pd

from multimodal_fin.processing.preprocessing.qa_classifier import QAClassifier
from multimodal_fin.processing.preprocessing.monologue_classifier import MonologueClassifier
from multimodal_fin.processing.preprocessing.transcript_preprocessor import TranscriptPreprocessor
from multimodal_fin.utils.logging import get_logger, log_ensemble_prediction

logger = get_logger(__name__)


@dataclass
class EnsembleInterventionClassifier:
    """
    Combines multiple Q&A and monologue classifiers to label interventions in a transcript.
    Handles classification and pairing of questions and answers.
    """

    qa_model_names: List[str]
    """List of Q&A classifier model names."""

    monologue_model_names: List[str]
    """List of monologue classifier model names."""

    NUM_EVALUATIONS: int = 5
    """Number of repeated evaluations per classifier for stability."""

    verbose: int = 1
    """Verbosity level for logging."""

    def __post_init__(self):
        self.qna_classifiers = [
            QAClassifier(model=name, NUM_EVALUATIONS=self.NUM_EVALUATIONS)
            for name in self.qa_model_names
        ]
        self.monologue_classifiers = [
            MonologueClassifier(model=name, NUM_EVALUATIONS=self.NUM_EVALUATIONS)
            for name in self.monologue_model_names
        ]
        self.preprocessor = TranscriptPreprocessor()

    def ensemble_predict(self, text: str, classifiers: List) -> Tuple[str, float, List[Tuple[str, str, float]]]:
        """
        Aggregates predictions from multiple classifiers for a given text.

        Args:
            text: Text to classify.
            classifiers: List of classifiers (Q&A or monologue).

        Returns:
            Tuple with predicted category, average confidence, and individual model predictions.
        """
        individual_preds = []

        for clf in classifiers:
            cat, conf = clf.get_pred(text)
            individual_preds.append((clf.model, cat, conf))

        # Aggregate confidence scores per category
        conf_sum = {}
        for _, cat, conf in individual_preds:
            conf_sum[cat] = conf_sum.get(cat, 0.0) + conf

        best_cat, total_conf = max(conf_sum.items(), key=lambda x: x[1])
        avg_conf = round(total_conf / len(classifiers), 2)

        if self.verbose >= 1:
            log_ensemble_prediction(individual_preds, best_cat, avg_conf, logger=logger)

        return best_cat, avg_conf, individual_preds

    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classifies each row in the transcript using an ensemble of classifiers.

        Args:
            df: DataFrame containing the transcript with a 'Conf_Section' column.

        Returns:
            DataFrame with added columns: 'classification', 'global_confidence', and 'model_predictions'.
        """
        df['classification'] = ' '
        df['global_confidence'] = 0.0
        df['model_predictions'] = None

        qna_mask = df['Conf_Section'] == 'q_a'
        if qna_mask.any():
            preds = df.loc[qna_mask, 'text'].apply(lambda text: self.ensemble_predict(text, self.qna_classifiers))
            df.loc[qna_mask, 'classification'] = preds.apply(lambda x: x[0])
            df.loc[qna_mask, 'global_confidence'] = preds.apply(lambda x: x[1])
            df.loc[qna_mask, 'model_predictions'] = preds.apply(lambda x: x[2])

        remarks_mask = df['Conf_Section'] == 'prepared_remarks'
        if remarks_mask.any():
            preds = df.loc[remarks_mask, 'text'].apply(lambda text: self.ensemble_predict(text, self.monologue_classifiers))
            df.loc[remarks_mask, 'classification'] = preds.apply(lambda x: x[0])
            df.loc[remarks_mask, 'global_confidence'] = preds.apply(lambda x: x[1])
            df.loc[remarks_mask, 'model_predictions'] = preds.apply(lambda x: x[2])

        return df

    def annotate_question_answer_pairs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assigns a unique ID to valid question-answer pairs in the transcript.

        Args:
            df: DataFrame already classified by `classify_dataframe`.

        Returns:
            DataFrame with an added 'Pair' column for Q&A associations and 'intervention_id'.
        
        Raises:
            ValueError: If any detected pair does not contain exactly two elements.
        """
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
            raise ValueError(
                f"Invalid Q&A pairs detected (must contain exactly 2 rows):\n{invalid_pairs.to_dict()}"
            )

        df["intervention_id"] = df.index
        return df
