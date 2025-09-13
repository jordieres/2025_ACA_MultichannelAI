"""
Runner for generating multimodal hierarchical embeddings from enriched JSON files.

Embeddings are generated using both node-level and conference-level encoders.
"""

from typing import List

from multimodal_fin.config import Settings, EmbeddingsPipelineSettings
from multimodal_fin.embeddings.builder.pipeline import ConferenceEmbeddingPipeline
from multimodal_fin.runners.base import Runner
from multimodal_fin.utils.logging import get_logger


logger = get_logger(__name__)


class EmbedRunner(Runner):
    """Runner for generating embeddings from enriched transcript JSONs."""

    def __init__(self, settings: Settings, emb_cfg: EmbeddingsPipelineSettings):
        """Initialize the embedding pipeline runner.

        Args:
            settings (Settings): General pipeline settings.
            emb_cfg (EmbeddingsPipelineSettings): Encoder model configurations.
        """
        self.pipeline = ConferenceEmbeddingPipeline(
            node_encoder_params=emb_cfg.node_encoder.model_dump(),
            conference_encoder_params=emb_cfg.conference_encoder.model_dump(),
            device=emb_cfg.device or settings.device,
        )
        self.pipeline.node_encoder.eval()
        self.pipeline.conference_encoder.eval()

    def run(self, paths: List[str], **kwargs) -> None:
        """Generate and log embeddings for each input JSON file.


        Args:
        paths (List[str]): List of file paths to enriched transcript JSONs.
        """
        for p in paths:
            try:
                embedding = self.pipeline.generate_embedding(str(p), return_attn=True)
                array = embedding.detach().cpu().numpy().flatten()
                logger.info(f"📦 Embedded: {p} → shape={array.shape}")
                logger.info(f"Embedding vector (first 5 values): {array[:5]}")
            except Exception as e:
                logger.error(f"Failed to embed {p}: {e}", exc_info=True)
