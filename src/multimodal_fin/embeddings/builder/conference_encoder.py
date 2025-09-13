import torch
import torch.nn as nn
from typing import Optional, Tuple

from multimodal_fin.embeddings.builder.transformer_encoder import TransformerEncoderLayer
from multimodal_fin.utils.logging import get_logger

logger = get_logger(__name__)


class ConferenceEncoder(nn.Module):
    """Encoder that aggregates node-level embeddings into a single conference-level embedding
    using a Transformer encoder with a [CLS] token and learned positional encodings."""

    def __init__(
        self,
        device: str = "cpu",
        input_dim: int = 512,
        hidden_dim: int = 256,
        n_heads: int = 4,
        d_output: int = 512,
        max_nodes: int = 1000,
        weights_path: Optional[str] = None,
    ):
        """
        Args:
            device: Device to run the model on.
            input_dim: Dimension of input node embeddings.
            hidden_dim: Hidden dimension of the Transformer.
            n_heads: Number of attention heads.
            d_output: Dimension of the output conference embedding.
            max_nodes: Max number of nodes to consider in a conference.
            weights_path: Optional path to a pretrained model checkpoint.
        """
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))
        self.pos_embedding = nn.Embedding(max_nodes + 1, input_dim)

        self.encoder_layer = TransformerEncoderLayer(
            d_model=input_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2
        )

        self.proj = nn.Linear(input_dim, d_output)

        if weights_path:
            try:
                state_dict = torch.load(weights_path, map_location=device)

                # Remove incompatible positional embeddings
                if 'pos_embedding.weight' in state_dict:
                    del state_dict['pos_embedding.weight']
                    # logger.warning("⚠️ Skipped 'pos_embedding.weight' due to size mismatch.")

                self.load_state_dict(state_dict, strict=False)
                logger.info(f"✅ Weights loaded from {weights_path}")
            except Exception as e:
                logger.error(f"❌ Failed to load weights from {weights_path}: {e}")
        else:
            logger.warning("⚠️ No pretrained weights path provided for ConferenceEncoder")

    def forward(self, node_embeddings: torch.Tensor, return_attn: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            node_embeddings: Tensor of shape [n_nodes, input_dim]
            return_attn: Whether to return attention weights from [CLS] token.

        Returns:
            Conference embedding of shape [1, d_output]
            Optionally, attention weights from [CLS] to all other nodes.
        """
        n_nodes = node_embeddings.size(0)

        # Insert [CLS] token
        cls = self.cls_token.expand(1, -1, -1)  # [1, 1, input_dim]
        input_seq = torch.cat([cls, node_embeddings.unsqueeze(0)], dim=1)  # [1, n+1, input_dim]

        # Positional encoding
        pos_ids = torch.arange(n_nodes + 1, device=input_seq.device).unsqueeze(0)  # [1, n+1]
        pos_emb = self.pos_embedding(pos_ids)
        input_seq = input_seq + pos_emb  # [1, n+1, input_dim]

        # Transformer
        out = self.encoder_layer(input_seq)  # [1, n+1, input_dim]

        if return_attn:
            attn_weights = self.encoder_layer.attn_weights  # [1, n_heads, T, T]
            attn_from_cls = attn_weights[0, 0, 0, 1:].detach().cpu().numpy()  # [n_nodes]
            return self.proj(out[:, 0, :]), attn_from_cls

        return self.proj(out[:, 0, :])
