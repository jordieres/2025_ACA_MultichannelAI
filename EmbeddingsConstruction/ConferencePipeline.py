import torch
from anytree import PreOrderIter

from .Embedding.FeatureExtractor import FeatureExtractor
from .Embedding.ConferenceEncoder import ConferenceEncoder
from .Embedding.NodeEncoder import NodeEncoder

from .Tree.ConferenceTreeBuilder import ConferenceTreeBuilder
from .Tree.ConferenceTreeVisualizer import ConferenceTreeVisualizer

class ConferenceEmbeddingPipeline:
    def __init__(self, node_encoder_params: dict, conference_encoder_params: dict):
        self.node_encoder = NodeEncoder(**node_encoder_params)
        self.conference_encoder = ConferenceEncoder(**conference_encoder_params)
        self.extractor = FeatureExtractor(
            categories_10k=self.node_encoder.categories_10k,
            qa_categories=self.node_encoder.qa_categories,
            max_num_coherences=self.node_encoder.max_num_coherences
        )

    def generate_embedding(self, json_path: str) -> torch.Tensor:

        builder = ConferenceTreeBuilder(json_path)
        self.root = builder.build_tree()

        node_embeddings = []
        for node in PreOrderIter(self.root):
            if node.is_leaf and node.node_type in {"monologue", "question", "answer"}:
                frases, mask, meta_vec = self.extractor.extract(node)
                frase_summary = self.node_encoder.frase_encoder(frases, mask)
                meta_summary = self.node_encoder.meta_proj(
                    torch.tensor(meta_vec, dtype=torch.float32).unsqueeze(0)
                )
                combined = torch.cat([frase_summary, meta_summary], dim=-1)
                node_embedding = self.node_encoder.output_proj(combined).squeeze(0)
                node_embeddings.append(node_embedding)

        if not node_embeddings:
            return torch.zeros(self.node_encoder.d_output)

        stacked = torch.stack(node_embeddings, dim=0)
        return self.conference_encoder(stacked)

    def visualize(self, structure=False, plot=False):
        visualizer = ConferenceTreeVisualizer(self.root)
        if structure:
            visualizer.show_text_tree()
        if plot:
            visualizer.show_networkx_tree()
        