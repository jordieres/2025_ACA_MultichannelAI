"""
Factory method for obtaining CLI runners based on the selected mode.

This module exposes a single function `get_runner` that returns an instance of the
appropriate runner class based on the CLI command invoked.
"""

from multimodal_fin.config import FullConfig
from multimodal_fin.runners.base import Runner
from multimodal_fin.runners.process_runner import ProcessRunner
from multimodal_fin.runners.embeds_runner import EmbedRunner
from multimodal_fin.runners.downloads_runner import DataAdquisitionRunner


def get_runner(mode: str, config: FullConfig) -> Runner:
    """Return the appropriate runner instance based on the selected mode.

    Args:
        mode (str): One of 'process', 'embed', or 'download'.
        config (FullConfig): Aggregated pipeline configuration.

    Returns:
        Runner: Instantiated runner object.

    Raises:
        ValueError: If the configuration is missing or the mode is unknown.
    """
    if mode == "process":
        if not config.processing:
            raise ValueError("Missing 'processing' configuration in the config file.")
        return ProcessRunner(config.processing)

    elif mode == "embed":
        if not config.processing or not config.embeddings:
            raise ValueError("Missing 'processing' or 'embeddings' configuration in the config file.")
        return EmbedRunner(config.processing, config.embeddings)

    elif mode == "download":
        if not config.data_adquisition:
            raise ValueError("Missing 'data_adquisition' configuration in the config file.")
        return DataAdquisitionRunner(config.data_adquisition)

    raise ValueError(f"Unknown runner mode: '{mode}'")