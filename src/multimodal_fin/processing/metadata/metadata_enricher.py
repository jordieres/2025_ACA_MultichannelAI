from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd

from multimodal_fin.processing.metadata.sec10k_analyzer import SEC10KAnalyzer
from multimodal_fin.processing.metadata.qa_analyzer import QAAnalyzer
from multimodal_fin.processing.metadata.coherence_analyzer import CoherenceAnalyzer

from multimodal_fin.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MetadataEnricher:
    """Enriches a DataFrame with metadata about topic, QA, and coherence.

    Attributes:
        sec10k_model_names (List[str]): List of model names used for topic classification.
        qa_analyzer_models (List[str]): List of models used for QA answerability analysis.
        num_evaluations (int): Number of repetitions per model to compute uncertainty.
        device (str): Device for model inference (e.g., 'cpu' or 'cuda').
        verbose (int): Verbosity level (0: silent, 1: info).
    """

    sec10k_model_names: List[str]
    qa_analyzer_models: List[str]
    num_evaluations: int = 5
    device: str = 'cpu'
    verbose: int = 1

    def __post_init__(self):
        self.topic_classifiers = [
            SEC10KAnalyzer(model=name, NUM_EVALUATIONS=self.num_evaluations)
            for name in self.sec10k_model_names
        ]
        self.qa_analyzers = [
            QAAnalyzer(model_name=name, NUM_EVALUATIONS=self.num_evaluations)
            for name in self.qa_analyzer_models
        ]
        self.coherence_analyzer = None
        if self.qa_analyzers:
            first_model = self.qa_analyzers[0].model_name
            self.coherence_analyzer = CoherenceAnalyzer(model_name=first_model)

    def enrich(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Orchestrates enrichment of the dataframe into structured JSON.

        Args:
            df (pd.DataFrame): DataFrame containing sentences and their embeddings.

        Returns:
            Dict[str, Any]: Structured metadata dictionary with monologues and Q&A pairs.
        """
        result: Dict[str, Any] = {"monologue_interventions": {}}

        # --- Process monologues ---
        monologues = df[df['classification'] == 'Monologue']
        for idx, group in monologues.groupby('intervention_id'):
            text = " ".join(group['text'])
            embeddings = self._get_multimodal_dict(group)
            topic_cat, topic_conf, topic_models = self._classify_topics(text)
            logger.info(f"✅ Topic analysis completed for monologue {idx}: {topic_cat} ({topic_conf:.2f}%)")

            result['monologue_interventions'][str(idx)] = {
                'text': text,
                'multimodal_embeddings': embeddings,
                'topic_classification': {
                    'Predicted_category': topic_cat,
                    'Confidence': topic_conf,
                    'Model_confidences': topic_models
                }
            }

        # --- Process QA Pairs ---
        qa_df = df[df['classification'].isin(['Question', 'Answer'])]
        for pair_id, group in qa_df.groupby('Pair'):
            if not isinstance(pair_id, str) or not pair_id.startswith('pair_'):
                continue

            q_group = group[group['classification'] == 'Question']
            a_group = group[group['classification'] == 'Answer']
            question = " ".join(q_group['text'])
            answer = " ".join(a_group['text'])

            # Topic classification
            q_topic = self._classify_topics(question)
            a_topic = self._classify_topics(answer)
            logger.info(f"✅ Topic analysis for question in {pair_id}: {q_topic[0]} ({q_topic[1]:.2f}%)")
            logger.info(f"✅ Topic analysis for answer in {pair_id}: {a_topic[0]} ({a_topic[1]:.2f}%)")


            # QA classification
            qa_cat, qa_conf, qa_models, qa_details = self._analyze_qa_pair(question, answer)
            logger.info(f"✅ QA analysis completed for pair {pair_id}: {qa_cat} ({qa_conf:.2f}%)")
            answered = qa_details.get('answered') if isinstance(qa_details, dict) else None

            # Coherence analysis
            coherence = []
            if self.coherence_analyzer:
                for mono_id, mono in result['monologue_interventions'].items():
                    try:
                        coh = self.coherence_analyzer.analyze_coherence(mono['text'], answer)
                        coh['monologue_index'] = int(mono_id)
                        coherence.append(coh)
                    except Exception:
                        continue  # Skip coherence errors silently

            logger.info(f"✅ Coherence analysis completed for pair {pair_id} with {len(coherence)} monologue links")

            result[pair_id] = {
                'question': question,
                'answer': answer,
                'answered': answered,
                'question_topic_classification': {
                    'Predicted_category': q_topic[0],
                    'Confidence': q_topic[1],
                    'Model_confidences': q_topic[2]
                },
                'answer_topic_classification': {
                    'Predicted_category': a_topic[0],
                    'Confidence': a_topic[1],
                    'Model_confidences': a_topic[2]
                },
                'qa_response_classification': {
                    'Predicted_category': qa_cat,
                    'Confidence': qa_conf,
                    'Model_confidences': qa_models,
                    'details': qa_details
                },
                'coherence_analyses': coherence,
                'multimodal_embeddings': {
                    'question': self._get_multimodal_dict(q_group),
                    'answer': self._get_multimodal_dict(a_group)
                }
            }

        logger.info(f"✅ Metadata enrichment completed. Enriched {len(monologues)} monologues and {len(result) - 1} QA pairs.")

        return result

    def _classify_topics(self, text: str) -> Tuple[str, float, Dict[str, Dict[str, Any]]]:
        """Classifies the topic of a given text using all topic classifiers.

        Args:
            text (str): Text to classify.

        Returns:
            Tuple[str, float, Dict[str, Dict[str, Any]]]:
                - Predicted category
                - Average confidence
                - Per-model predicted category and confidence
        """
        predictions = [
            (clf.get_pred(text)[0], clf.get_pred(text)[1], clf.model)
            for clf in self.topic_classifiers
        ]
        conf_sum: Dict[str, float] = {}
        for cat, conf, _ in predictions:
            conf_sum[cat] = conf_sum.get(cat, 0.0) + conf
        best_cat, total_conf = max(conf_sum.items(), key=lambda x: x[1])
        avg_conf = round(total_conf / len(predictions), 2) if predictions else 0.0
        model_confidences = {
            model: {'Predicted_category': cat, 'Confidence': round(conf, 2)}
            for cat, conf, model in predictions
        }
        return best_cat, avg_conf, model_confidences

    def _analyze_qa_pair(
        self, question: str, answer: str
    ) -> Tuple[str, float, Dict[str, Dict[str, Any]], Dict[str, Any]]:
        """Analyzes whether a question is answered, using QA analyzers.

        Args:
            question (str): Question text.
            answer (str): Answer text.

        Returns:
            Tuple[str, float, model_confidences, details]:
                - Best predicted category
                - Average confidence
                - Model-level prediction info
                - Extra QA details (e.g., answer coverage)
        """
        results = []
        model_conf: Dict[str, Dict[str, float]] = {}

        for analyzer in self.qa_analyzers:
            cat, conf, details = analyzer.get_pred(question, answer)
            if not cat:
                continue
            results.append((cat, conf, analyzer.model_name, details))
            model_conf[analyzer.model_name] = {
                'Predicted_category': cat,
                'Confidence': round(conf, 2)
            }

        if not results:
            return None, 0.0, model_conf, {}

        conf_sum: Dict[str, float] = {}
        for cat, conf, *_ in results:
            conf_sum[cat] = conf_sum.get(cat, 0.0) + conf
        best_cat, total_conf = max(conf_sum.items(), key=lambda x: x[1])
        avg_conf = round(total_conf / len(results), 2)
        detail = next((d for c, _, _, d in results if c == best_cat and isinstance(d, dict)), {})

        return best_cat, avg_conf, model_conf, detail

    def _get_multimodal_dict(self, df_sub: pd.DataFrame) -> Dict[str, Any]:
        """Extracts embeddings from a subset of the DataFrame.

        Args:
            df_sub (pd.DataFrame): Subset of the main DataFrame.

        Returns:
            Dict[str, Any]: Embedding dictionary with audio, text, and video vectors.
        """
        return {
            'num_sentences': len(df_sub),
            'audio': df_sub.get('audio_embedding').tolist() if 'audio_embedding' in df_sub else None,
            'text': df_sub.get('text_embedding').tolist() if 'text_embedding' in df_sub else None,
            'video': df_sub.get('video_embedding').tolist() if 'video_embedding' in df_sub else None,
        }
