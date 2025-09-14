# multimodal_fin/processing/runner.py

from pathlib import Path
import torch
import time

from multimodal_fin.config import Settings
from multimodal_fin.utils.logging import get_logger
from multimodal_fin.utils.files import read_paths_csv, make_processed_path

from multimodal_fin.processing.preprocessing.preprocessor import Preprocessor
from multimodal_fin.processing.processor import Processor

logger = get_logger(__name__)


class ConferencePipeline:
    """
    Orchestrates the full processing pipeline for a financial conference folder:
    
    Steps performed:
        1. Preprocessing of the transcript and section segmentation.
        2. Text classification and question-answer (Q&A) annotation.
        3. Multimodal embedding extraction (text, audio, video).
        4. Metadata enrichment using LLMs (topics, Q&A analysis, coherence).
        5. Result persistence in CSV and enriched JSON format.
    """

    def __init__(self, settings: Settings) -> None:
        """
        Initialize the processor with given settings.

        Args:
            settings (Settings): Configuration parameters for processing.
        """
        self.settings = settings
        self.device = settings.device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Step 1: Text pipeline (sectioning + classification)
        self.preprocessor = Preprocessor(
            qa_model_names=settings.qa_models,
            monologue_model_names=settings.monologue_models,
            num_evaluations=settings.evals,
            verbose=settings.verbose
        )

        # Step 2: Multimodal pipeline (embedding extraction + enrichment)
        self.multimodal_processor = Processor(
            sec10k_model_names=settings.sec10k_models,
            qa_analyzer_models=settings.qa_analyzer_models,
            audio_model_name=settings.audio_model,
            text_model_name=settings.text_model,
            video_model_name=settings.video_model,
            num_evaluations=settings.evals,
            device=self.device,
            verbose=settings.verbose
        )

    def run(self) -> None:
        """
        Run the processing pipeline on each conference folder path defined in the input CSV.
        """
        for original_path in read_paths_csv(self.settings.input_csv_path):
            try:
                self._process_conference(Path(original_path))
            except Exception as e:
                logger.error(f"Failed to process: {original_path}", exc_info=True)

    def _process_conference(self, original: Path) -> None:
        """
        Process a single conference folder end-to-end.

        Args:
            original (Path): Path to the original conference folder.

        Raises:
            FileNotFoundError: If transcript.csv or LEVEL_4.json is missing.
        """
        logger.info(f"🔄 Starting processing for conference: {original}")
        start_time = time.perf_counter()

        # Create output directory
        processed_dir = make_processed_path(original)
        processed_dir.mkdir(parents=True, exist_ok=True)

        # Validate required input files
        transcript_csv = original / "transcript.csv"
        level4_json = original / "LEVEL_4.json"
        if not transcript_csv.exists() or not level4_json.exists():
            raise FileNotFoundError(f"Required files not found in {original}")

        # Step 1: Preprocessing and CSV output
        output_csv = processed_dir / "transcript.csv"
        df = self.preprocessor.process_and_save(
            csv_path=str(transcript_csv),
            json_path=str(level4_json),
            output_csv_path=str(output_csv)
        )
        logger.info(f"✅ Transcript classified and saved at: {output_csv}")

        # Step 2: Multimodal processing and JSON output
        output_json = processed_dir / "transcript.json"
        self.multimodal_processor.process_and_save(
            input_csv_path=str(output_csv),
            original_dir=original,
            output_json_path=str(output_json)
        )
        logger.info(f"✅ Enriched JSON saved at: {output_json}")
        logger.info(f"⏱️ Finished processing conference: {original.name} in {(time.perf_counter() - start_time):.2f} seconds")
