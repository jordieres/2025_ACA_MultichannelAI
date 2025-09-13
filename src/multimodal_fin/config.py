"""
Configuration loader and schema definitions for the multimodal_fin package.

This module provides Pydantic-based configuration classes and utilities
to load structured settings from YAML files.
"""

import yaml
from pydantic import BaseModel, Field
from typing import List, Optional


def default_device() -> str:
    """Return the default device based on PyTorch availability.

    Returns:
        str: 'cuda' if a CUDA-enabled GPU is available, otherwise 'cpu'.
    """
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class DataAdquisitionSettings(BaseModel):
    """Settings for downloading earnings call data."""
    api_key: str
    base_path: str
    url: str


class NodeEncoderParams(BaseModel):
    """Model parameters for the node-level encoder."""
    hidden_dim: int
    meta_dim: int
    n_heads: int
    d_output: int
    weights_path: str


class ConferenceEncoderParams(BaseModel):
    """Model parameters for the conference-level encoder."""
    hidden_dim: int
    input_dim: int
    n_heads: int
    d_output: int
    weights_path: str


class EmbeddingsPipelineSettings(BaseModel):
    """Settings for the full embedding pipeline."""
    node_encoder: NodeEncoderParams
    conference_encoder: ConferenceEncoderParams
    device: Optional[str] = Field(default="cuda")


class Settings(BaseModel):
    """Settings for the full conference processing pipeline."""
    input_csv_path: str
    qa_models: List[str]
    monologue_models: List[str]
    sec10k_models: List[str]
    qa_analyzer_models: List[str]
    audio_model: Optional[str] = None
    text_model: Optional[str] = None
    video_model: Optional[str] = None
    evals: int = 3
    device: str = Field(default_factory=default_device)
    verbose: int = 1


class FullConfig(BaseModel):
    """Aggregated configuration object including all pipeline components."""
    processing: Optional[Settings] = None
    embeddings: Optional[EmbeddingsPipelineSettings] = None
    data_adquisition: Optional[DataAdquisitionSettings] = None


def _load_processing_settings(section: dict, config_name: str) -> Optional[Settings]:
    """Parse and load processing-related settings from config block.

    Args:
        section (dict): YAML section for 'conferences_processing'.
        config_name (str): Name of the specific configuration subsection.

    Returns:
        Optional[Settings]: Parsed processing settings or None if not found.
    """
    conf = section.get(config_name)
    if not conf:
        return None

    embeddings = conf.get("embeddings", {})
    audio_model = embeddings.get("audio", {}).get("model_name") if embeddings.get("audio", {}).get("enabled") else None
    text_model = embeddings.get("text", {}).get("model_name") if embeddings.get("text", {}).get("enabled") else None
    video_model = embeddings.get("video", {}).get("model_name") if embeddings.get("video", {}).get("enabled") else None

    return Settings(
        input_csv_path=conf['input_csv_path'],
        qa_models=conf['qa_models'],
        monologue_models=conf['monologue_models'],
        sec10k_models=conf['sec10k_models'],
        qa_analyzer_models=conf['qa_analyzer_models'],
        audio_model=audio_model,
        text_model=text_model,
        video_model=video_model,
        evals=conf.get('evals', 3),
        device=conf.get('device', default_device()),
        verbose=conf.get('verbose', 1),
    )


def _load_embeddings_settings(section: dict, config_name: str) -> Optional[EmbeddingsPipelineSettings]:
    """Parse and load embeddings pipeline settings.

    Args:
        section (dict): YAML section for 'embeddings_pipeline'.
        config_name (str): Name of the specific configuration subsection.

    Returns:
        Optional[EmbeddingsPipelineSettings]: Parsed settings or None if not found.
    """
    conf = section.get(config_name)
    if not conf:
        return None

    return EmbeddingsPipelineSettings(
        node_encoder=NodeEncoderParams(**conf["node_encoder"]),
        conference_encoder=ConferenceEncoderParams(**conf["conference_encoder"]),
        device=conf.get("device", default_device()),
    )


def _load_data_settings(section: dict, override_url: Optional[str] = None) -> Optional[DataAdquisitionSettings]:
    """Parse and load data acquisition settings.

    Args:
        section (dict): YAML section for 'conferences_data_adquisition'.
        override_url (Optional[str], optional): URL to override default. Defaults to None.

    Returns:
        Optional[DataAdquisitionSettings]: Parsed settings or None if section is empty.
    """
    if not section:
        return None

    return DataAdquisitionSettings(
        api_key=section['api_key'],
        base_path=section['base_path'],
        url=override_url or section['url'],
    )


def load_full_config(config_path: str, config_name: str = "default", override_url: Optional[str] = None) -> FullConfig:
    """Load all pipeline configuration components from YAML.

    Args:
        config_path (str): Path to the YAML configuration file.
        config_name (str, optional): Name of the config block for each section. Defaults to "default".
        override_url (Optional[str], optional): Optional override for the download URL.

    Returns:
        FullConfig: Aggregated configuration object.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)

    return FullConfig(
        processing=_load_processing_settings(raw.get("conferences_processing", {}), config_name),
        embeddings=_load_embeddings_settings(raw.get("embeddings_pipeline", {}), config_name),
        data_adquisition=_load_data_settings(raw.get("conferences_data_adquisition", {}), override_url),
    )
