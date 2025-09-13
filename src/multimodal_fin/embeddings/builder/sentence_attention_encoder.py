import torch
import torch.nn as nn
from typing import Optional, Tuple

from multimodal_fin.utils.logging import get_logger

logger = get_logger(__name__)


class SentenceAttentionEncoder(nn.Module):
    """
    Encodes a sequence of token-level embeddings into a sentence-level embedding using self-attention.
    A learnable [CLS] token is prepended to attend over the sequence.
    """

    def __init__(
        self,
        input_dim: int = 21,
        hidden_dim: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Initializes the attention-based sentence encoder.

        Args:
            input_dim: Dimensionality of the input embeddings (per token).
            hidden_dim: Dimensionality of the internal hidden representation.
            n_heads: Number of attention heads in the multi-head attention layer.
            dropout: Dropout rate applied after the attention layer.
        """
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

        logger.info("✅ SentenceAttentionEncoder initialized.")

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for the sentence encoder.

        Args:
            x: Input tensor of shape [B, N, input_dim], where B is batch size, N is sequence length.
            mask: Optional mask of shape [B, N] indicating valid tokens (1) vs padding (0).
            return_weights: Whether to return attention weights from [CLS] token to input tokens.

        Returns:
            If return_weights is False:
                Tensor of shape [B, hidden_dim] representing sentence-level embeddings.
            If return_weights is True:
                Tuple of (embeddings, attention_weights), where:
                    - embeddings: [B, hidden_dim]
                    - attention_weights: [B, N] average attention from CLS to tokens
        """
        B, N, _ = x.shape
        x = self.input_proj(x)  # [B, N, hidden_dim]

        cls_token = self.cls_token.expand(B, 1, -1)  # [B, 1, hidden_dim]
        x = torch.cat([cls_token, x], dim=1)  # [B, N+1, hidden_dim]

        key_padding_mask = None
        if mask is not None:
            mask = torch.cat([torch.ones(B, 1, dtype=mask.dtype, device=mask.device), mask], dim=1)  # [B, N+1]
            key_padding_mask = ~mask.bool()  # [B, N+1]

        attn_output, attn_weights = self.attention(
            x, x, x,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False  # attn_weights: [B, n_heads, tgt_len, src_len]
        )

        x = self.norm(x + self.dropout(attn_output))  # [B, N+1, hidden_dim]
        cls_emb = x[:, 0, :]  # [B, hidden_dim]

        if return_weights:
            attn_to_tokens = attn_weights[:, :, 0, 1:]  # [B, n_heads, N]
            attn_mean = attn_to_tokens.mean(dim=1)  # [B, N]
            return cls_emb, attn_mean

        return cls_emb
