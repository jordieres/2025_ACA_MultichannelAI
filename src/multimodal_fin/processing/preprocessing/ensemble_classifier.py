from dataclasses import dataclass
from typing import List, Tuple, Optional

import pandas as pd

from sentence_transformers import SentenceTransformer, util
import torch

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


    def annotate_question_answer_pairs(self, 
                                   df: pd.DataFrame, 
                                   window_size: int = 4, 
                                   similarity_threshold: float = 0.45) -> pd.DataFrame:
        """
        Annotates question-answer pairs based on semantic similarity within a contextual window,
        using the 'multi-qa-MiniLM-L6-cos-v1' model.

    #     El flujo ideal sería:
    #     Recorres el DataFrame ordenado cronológicamente.
    #     Cada vez que encuentras una respuesta, buscas hacia atrás las últimas N preguntas no emparejadas.
    #     Para cada una calculas una medida de similitud semántica (embedding). https://sbert.net/docs/sentence_transformer/pretrained_models.html#multi-qa-models
    #     Asignas la respuesta a la pregunta con mayor similitud (si pasa un umbral).
    #     Si la respuesta parece abarcar varias preguntas (similares entre sí), puedes emparejar más de una.

        Args:
            df (pd.DataFrame): Classified transcript with 'classification' and 'text' columns.
            window_size (int): How many previous questions to consider for each answer.
            similarity_threshold (float): Minimum cosine similarity to form a valid Q→A pair.

        Returns:
            pd.DataFrame: Annotated DataFrame with 'Pair' and 'intervention_id' columns.
        """
        df = df.reset_index(drop=True)
        pair_id = 1
        pairs = [None] * len(df)

        # Load the QA-optimized model
        model = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")

        # Precompute normalized embeddings for all texts
        texts = df["text"].fillna("").tolist()
        embeddings = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)

        # Maintain a buffer of open questions
        open_questions = []

        for idx, row in df.iterrows():
            role = str(row.get("classification", "")).lower()

            if role == "question":
                open_questions.append(idx)

            elif role == "answer" and open_questions:
                # Limit to last N open questions
                candidate_idxs = open_questions[-window_size:]
                question_embs = embeddings[candidate_idxs]
                answer_emb = embeddings[idx].unsqueeze(0)

                # Compute cosine similarities
                sim_scores = util.cos_sim(answer_emb, question_embs).cpu().numpy().flatten()

                # Pick the best matching question
                best_idx = int(torch.argmax(torch.tensor(sim_scores)))
                best_score = sim_scores[best_idx]
                chosen_q = candidate_idxs[best_idx]

                # Check similarity threshold
                if best_score >= similarity_threshold:
                    pair_label = f"pair_{pair_id}"
                    pairs[chosen_q] = pair_label
                    pairs[idx] = pair_label
                    pair_id += 1

                    # Remove matched question
                    open_questions = [q for q in open_questions if q != chosen_q]

                    logger.debug(
                        f"Matched Q{chosen_q} ↔ A{idx} with similarity={best_score:.3f} "
                        f"(Pair={pair_label})"
                    )
                else:
                    logger.debug(
                        f"No strong match for answer {idx}: best_score={best_score:.3f}"
                    )

        # Assign results
        df["Pair"] = pairs
        df["intervention_id"] = df.index

        # Logging unmatched elements
        unmatched_qs = [i for i, p in enumerate(pairs) if df.loc[i, "classification"] == "Question" and p is None]
        unmatched_as = [i for i, p in enumerate(pairs) if df.loc[i, "classification"] == "Answer" and p is None]

        logger.info(f"Annotated {pair_id - 1} valid Q&A pairs.")
        logger.info(f"Unmatched questions: {len(unmatched_qs)}, unmatched answers: {len(unmatched_as)}")

        df["similarity_score"] = [
            None if p is None else float(util.cos_sim(embeddings[idx], embeddings[df[df["Pair"] == p].index[0]]))
            for idx, p in enumerate(pairs)
        ]

        return df