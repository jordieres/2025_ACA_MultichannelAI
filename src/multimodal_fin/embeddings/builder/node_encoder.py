import logging
import torch
import torch.nn as nn
from multimodal_fin.embeddings.builder.sentence_attention_encoder import SentenceAttentionEncoder
from multimodal_fin.utils.logging import get_logger

logger = get_logger(__name__)

class NodeEncoder(nn.Module):
    """
    Encodes individual nodes in a conference tree using multimodal features and metadata.

    Attributes:
        frase_encoder (nn.Module): Encoder for sentence-level features using attention.
        meta_proj (nn.Linear): Projection layer for metadata features.
        output_proj (nn.Linear): Final projection layer to produce node embedding.
        categories_10k (List[str]): List of 10-K classification categories.
        qa_categories (List[str]): List of QA response categories.
        max_num_coherences (int): Maximum number of coherence entries per node.
    """

    def __init__(
        self,
        device: str = "cpu",
        input_dim: int = 21,  # 7 (text) + 7 (audio) + 7 (video)
        hidden_dim: int = 128,
        meta_dim: int = 32,
        d_output: int = 512,
        n_heads: int = 4,
        categories_10k: list = None,
        qa_categories: list = None,
        weights_path: str = "weights/node_encoder.pt"
    ):
        super().__init__()
        self.device = device
        self.d_output = d_output
        self.meta_dim = meta_dim
        self.categories_10k = categories_10k or ["MD&A", "Risk Factors", "Business", "Other"]
        self.qa_categories = qa_categories or ["yes", "no", "partially"]
        self.max_num_coherences = 5
        self.weights_path = weights_path

        self.frase_encoder = SentenceAttentionEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads
        )

        if weights_path:
            try:
                self.frase_encoder.load_state_dict(torch.load(weights_path, map_location=device))
                logger.info(f"✅ Weights loaded from {weights_path}")
            except Exception as e:
                logger.warning(f"❌ Failed to load weights from {weights_path}: {e}")
        else:
            logger.warning("⚠️ No pretrained weights path provided for NodeEncoder")

        self.frase_encoder.to(device)
        self.meta_proj = nn.Linear(self._get_meta_input_size(), meta_dim)
        self.output_proj = nn.Linear(hidden_dim + meta_dim, d_output)

    def _get_meta_input_size(self) -> int:
        """Returns the dimensionality of the metadata feature vector."""
        return (
            1 + len(self.categories_10k) +
            1 + len(self.qa_categories) +
            2 * self.max_num_coherences
        )
