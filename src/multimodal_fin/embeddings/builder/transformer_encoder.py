import torch.nn.functional as F
from torch import Tensor
from torch.nn import (
    MultiheadAttention,
    LayerNorm,
    Dropout,
    Linear,
    Module
)
from typing import Optional, Tuple

from multimodal_fin.utils.logging import get_logger

logger = get_logger(__name__)


class TransformerEncoderLayer(Module):
    """
    Custom Transformer encoder layer with self-attention, feedforward network,
    residual connections and layer normalization.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        """
        Initializes the TransformerEncoderLayer.

        Args:
            d_model: Input and output dimensionality of the model.
            nhead: Number of attention heads.
            dim_feedforward: Dimensionality of the inner feedforward layer.
            dropout: Dropout rate applied after attention and feedforward layers.
        """
        super().__init__()

        self.self_attn = MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = F.gelu

        # Will store attention weights from forward pass
        self.attn_weights: Optional[Tensor] = None

        logger.info("✅ TransformerEncoderLayer initialized")

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass of the transformer encoder layer.

        Args:
            src: Input tensor of shape [B, T, d_model].
            src_mask: Optional attention mask [T, T] or [B * num_heads, T, T].
            src_key_padding_mask: Optional mask [B, T] indicating padding positions.

        Returns:
            Output tensor of shape [B, T, d_model].
        """
        attn_output, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )

        self.attn_weights = attn_weights  # [B, n_heads, T, T]

        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)

        return src
