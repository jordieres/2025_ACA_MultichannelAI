import torch.nn as nn

from .SentenceAttentionEncoder import SentenceAttentionEncoder

class NodeEncoder(nn.Module):
    def __init__(self, 
                 input_dim=21,            # 7 (text) + 7 (audio) + 7 (video)
                 hidden_dim=128,          # Atención por frase
                 meta_dim=32,             # Proyección de metadatos
                 d_output=512,            # Embedding final
                 categories_10k=None,
                 qa_categories=None):
        super().__init__()

        self.d_output = d_output
        self.meta_dim = meta_dim
        self.categories_10k = categories_10k or ["MD&A", "Risk Factors", "Business", "Other"]
        self.qa_categories = qa_categories or ["yes", "no", "partially"]
        self.max_num_coherences = 5  # número máximo de entradas de coherencia

        # Encoder de frases con atención
        self.frase_encoder = SentenceAttentionEncoder(input_dim=input_dim, hidden_dim=hidden_dim)

        # Proyección de metadatos
        self.meta_proj = nn.Linear(self._get_meta_input_size(), meta_dim)

        # Proyección final
        self.output_proj = nn.Linear(hidden_dim + meta_dim, d_output)

    def _get_meta_input_size(self) -> int:
        return (
            1 + len(self.categories_10k) +    # clasificación 10K
            1 + len(self.qa_categories) +     # respuesta
            2 * self.max_num_coherences       # coherencias booleanas (1-hot de 2)
        )