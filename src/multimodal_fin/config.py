"""
Configuration loader and schemas for the multimodal_fin package.
This module uses Pydantic to define and validate pipeline settings loaded from YAML.
"""
import yaml
from pydantic import BaseModel, Field
from typing import List, Optional


def default_device() -> str:
    """
    Determine the default compute device based on PyTorch availability.

    Returns:
        str: 'cuda' if a CUDA-enabled GPU is available, otherwise 'cpu'.
    """
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

class DataAdquisitionSettings(BaseModel):
    api_key: str
    base_path: str
    url: str


class NodeEncoderParams(BaseModel):
    hidden_dim: int
    meta_dim: int
    n_heads: int
    d_output: int
    weights_path: str


class ConferenceEncoderParams(BaseModel):
    hidden_dim: int
    input_dim: int
    n_heads: int
    d_output: int
    weights_path: str


class EmbeddingsPipelineSettings(BaseModel):
    node_encoder: NodeEncoderParams
    conference_encoder: ConferenceEncoderParams
    device: Optional[str] = Field(default="cuda")


class Settings(BaseModel):
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



# Funciones de carga
def load_settings(config_path: str, config_name: str = "default") -> Settings:
    with open(config_path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)

    conf = raw['conferences_processing'][config_name]
    embeddings = conf.get('embeddings', {})
    audio_model = embeddings.get('audio', {}).get('model_name') if embeddings.get('audio', {}).get('enabled') else None
    text_model = embeddings.get('text', {}).get('model_name') if embeddings.get('text', {}).get('enabled') else None
    video_model = embeddings.get('video', {}).get('model_name') if embeddings.get('video', {}).get('enabled') else None

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
        verbose=conf.get('verbose', 1)
    )


def load_embed_settings(config_path: str, config_name: str = "default") -> EmbeddingsPipelineSettings:
    with open(config_path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)

    conf = raw['embeddings_pipeline'][config_name]

    return EmbeddingsPipelineSettings(
        node_encoder=NodeEncoderParams(**conf['node_encoder']),
        conference_encoder=ConferenceEncoderParams(**conf['conference_encoder'])
    )


def load_data_settings(config_path: str, config_name: str = "default", override_url: Optional[str] = None) -> DataAdquisitionSettings:
    with open(config_path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)

    conf = raw['conferences_data_adquisition']
    return DataAdquisitionSettings(
        api_key=conf['api_key'],
        base_path=conf['base_path'],
        url=override_url or conf['url']
    )