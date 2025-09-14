import json
import logging
from pathlib import Path
from typing import Optional

from multimodal_fin.processing.multimodal.embeddings_extractor import EmbeddingsExtractor
from multimodal_fin.processing.metadata.metadata_enricher import MetadataEnricher

from multimodal_fin.utils.logging import get_logger

logger = get_logger(__name__)


class Processor:
    """
    Orchestrates the multimodal analysis pipeline in two main steps:
      1. Embedding extraction for audio, text, and video.
      2. Metadata enrichment (QA analysis, coherence, topics).
      3. JSON serialization of enriched output.
    """

    def __init__(
        self,
        sec10k_model_names: list[str],
        qa_analyzer_models: list[str],
        audio_model_name: Optional[str] = None,
        text_model_name: Optional[str] = None,
        video_model_name: Optional[str] = None,
        num_evaluations: int = 5,
        device: str = "cpu",
        verbose: int = 1,
    ):
        """
        Initializes the multimodal processor with models and configurations.

        Args:
            sec10k_model_names: List of LLMs for 10-K topic classification.
            qa_analyzer_models: List of LLMs for QA classification.
            audio_model_name: Model name for audio embeddings (optional).
            text_model_name: Model name for text embeddings (optional).
            video_model_name: Model name for video embeddings (optional).
            num_evaluations: Number of repeated LLM evaluations for uncertainty.
            device: Device to use for inference (e.g., 'cpu' or 'cuda').
            verbose: Verbosity level for logging and debugging.
        """
        self.verbose = verbose

        self.extractor = EmbeddingsExtractor(
            audio_model_name=audio_model_name,
            text_model_name=text_model_name,
            video_model_name=video_model_name,
            device=device,
            verbose=verbose,
        )

        self.enricher = MetadataEnricher(
            sec10k_model_names=sec10k_model_names,
            qa_analyzer_models=qa_analyzer_models,
            num_evaluations=num_evaluations,
            device=device,
            verbose=verbose,
        )

    def process_and_save(
        self,
        input_csv_path: str,
        original_dir: Path,
        output_json_path: str
    ) -> dict:
        """
        Executes the full multimodal pipeline and writes enriched results to a JSON file.

        Args:
            input_csv_path: Path to classified interventions CSV.
            original_dir: Directory containing LEVEL_3.json and audio/video files.
            output_json_path: Destination path for saving the final JSON.

        Returns:
            A dictionary containing the enriched multimodal results.
        """
        logger.info("Starting multimodal processing pipeline.")
        logger.debug(f"Input CSV: {input_csv_path}")
        logger.debug(f"Original directory: {original_dir}")
        logger.debug(f"Output JSON: {output_json_path}")

        # Step 1: Extract multimodal embeddings
        logger.info("Step 1: Extracting multimodal embeddings...")
        df_with_embeddings = self.extractor.extract(
            csv_path=input_csv_path,
            original_dir=str(original_dir)
        )

        # Step 2: Enrich with metadata (topics, QA classification, coherence)
        logger.info("Step 2: Enriching with metadata...")
        enriched_result = self.enricher.enrich(df=df_with_embeddings)

        # Step 3: Serialize results to JSON
        logger.info("Step 3: Saving enriched results to JSON...")
        output_path = Path(output_json_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(enriched_result, f, ensure_ascii=False, indent=2)

        logger.info(f"Multimodal processing complete. Output saved to: {output_path}")
        return enriched_result
