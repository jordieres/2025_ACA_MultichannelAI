import torch
import torch.nn as nn

from .TransformerEncoder import TransformerEncoderLayer


class ConferenceEncoder(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, n_heads=4, d_output=512):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))

        # Usa nuestra nueva capa
        self.encoder_layer = TransformerEncoderLayer(
            d_model=input_dim, nhead=n_heads, dim_feedforward=hidden_dim*2
        )

        self.proj = nn.Linear(input_dim, d_output)

    def forward(self, node_embeddings: torch.Tensor, return_attn=False):
        cls = self.cls_token.expand(1, -1, -1)  # [1, 1, input_dim]
        input_seq = torch.cat([cls, node_embeddings.unsqueeze(0)], dim=1)  # [1, n+1, input_dim]

        out = self.encoder_layer(input_seq)  # [1, n+1, input_dim]

        if return_attn:
            # extrae atención de la primera cabeza del token CLS
            attn_weights = self.encoder_layer.attn_weights  # [1, n_heads, T, T]
            attn_from_cls = attn_weights[0, 0, 0, 1:].detach().cpu().numpy()  # [n_nodes]
            return self.proj(out[:, 0, :]), attn_from_cls  # [1, d_output], [n_nodes]

        return self.proj(out[:, 0, :]).squeeze(0)  # [1, d_output]