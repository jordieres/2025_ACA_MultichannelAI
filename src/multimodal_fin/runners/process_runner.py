"""
Runner for executing the conference processing pipeline.

This runner handles classification, enrichment, and other processing steps
based on a provided configuration.
"""

from multimodal_fin.config import Settings
from multimodal_fin.processing.pipeline import ConferencePipeline
from multimodal_fin.runners.base import Runner


class ProcessRunner(Runner):
    """Runner responsible for executing the main conference processing pipeline."""

    def __init__(self, settings: Settings):
        """Initialize the processing runner.

        Args:
            settings (Settings): Configuration for the processing step.
        """
        self.processor = ConferencePipeline(settings)

    def run(self, **kwargs) -> None:
        """Run the full processing pipeline: classification and metadata enrichment."""
        self.processor.run()
