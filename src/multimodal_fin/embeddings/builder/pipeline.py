import torch
import logging
from anytree import PreOrderIter

from multimodal_fin.embeddings.builder.feature_extractor import FeatureExtractor
from multimodal_fin.embeddings.builder.node_encoder import NodeEncoder
from multimodal_fin.embeddings.builder.conference_encoder import ConferenceEncoder

from multimodal_fin.embeddings.speech_tree.conference_tree_builder import ConferenceTreeBuilder

from multimodal_fin.embeddings.visualizer.conference_tree_visualizer import ConferenceTreeVisualizer
from multimodal_fin.embeddings.visualizer.tree_attention_visualizer import TreeAttentionVisualizer
from multimodal_fin.embeddings.visualizer.node_embeddings_visualizer import NodeEmbeddingVisualizer

logger = logging.getLogger(__name__)


class ConferenceEmbeddingPipeline:
    """Orchestrates the generation and visualization of conference-level embeddings."""

    def __init__(
        self,
        node_encoder_params: dict,
        conference_encoder_params: dict,
        device: str = "cpu"
    ):
        """Initializes the pipeline with encoders and feature extractor.

        Args:
            node_encoder_params (dict): Parameters for the NodeEncoder.
            conference_encoder_params (dict): Parameters for the ConferenceEncoder.
            device (str): Torch device to use ("cpu" or "cuda").
        """
        self.device = torch.device(device)
        self.node_encoder = NodeEncoder(self.device, **node_encoder_params).to(self.device)
        self.conference_encoder = ConferenceEncoder(self.device, **conference_encoder_params).to(self.device)
        self.extractor = FeatureExtractor(
            categories_10k=self.node_encoder.categories_10k,
            qa_categories=self.node_encoder.qa_categories,
            max_num_coherences=self.node_encoder.max_num_coherences
        )

        logger.info("ConferenceEmbeddingPipeline initialized.")

    def generate_embedding(self, json_path: str, return_attn: bool = False) -> torch.Tensor:
        """Generates the embedding for a given conference JSON.

        Args:
            json_path (str): Path to the JSON file describing the conference.
            return_attn (bool): Whether to return attention weights.

        Returns:
            torch.Tensor: Embedding vector for the full conference.
        """
        builder = ConferenceTreeBuilder(json_path)
        self.root = builder.build_tree()

        self._node_embeddings = []
        self._node_names = []
        self._node_types = []
        self._categories_10k = []

        for node in PreOrderIter(self.root):
            if node.is_leaf and node.node_type in {"monologue", "question", "answer"}:
                frases, mask, meta_vec = self.extractor.extract(node)
                frase_summary = self.node_encoder.frase_encoder(frases.to(self.device), mask.to(self.device))
                meta_tensor = torch.tensor(meta_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
                meta_summary = self.node_encoder.meta_proj(meta_tensor)
                combined = torch.cat([frase_summary, meta_summary], dim=-1)
                node_embedding = self.node_encoder.output_proj(combined).squeeze(0)

                self._node_embeddings.append(node_embedding)
                self._node_names.append(node.name)
                self._node_types.append(node.node_type)

                predicted_cat = node.metadata.get("classification", {}).get("Predicted_category", "None")
                self._categories_10k.append(predicted_cat if node.node_type != "monologue" else "None")

        if not self._node_embeddings:
            logger.warning("No valid leaf nodes found for embedding computation.")
            return torch.zeros(self.node_encoder.d_output)

        stacked = torch.stack(self._node_embeddings, dim=0)
        if return_attn:
            conference_embedding, attn_weights = self.conference_encoder(stacked, return_attn=True)
            self._attn_weights = attn_weights
            return conference_embedding

        return self.conference_encoder(stacked)

    def visualize(self, plots: dict = None):
        """Visualizes the results of the embedding process depending on selected plots.

        Args:
            plots (dict): Flags for which plots to generate.
        """
        plots = plots or {}
        visualizer = ConferenceTreeVisualizer(self.root)

        if plots.get("tree_structure"):
            visualizer.show_text_tree()

        if plots.get("plot"):
            visualizer.show_networkx_tree()

        if any(plots.get(k, False) for k in ("silhouette", "umap")):
            embedding_visualizer = NodeEmbeddingVisualizer(
                embeddings=self._node_embeddings,
                node_names=self._node_names,
                node_types=self._node_types,
                categories_10k=self._categories_10k
            )

            if plots.get("silhouette"):
                embedding_visualizer.show_metrics()

            if plots.get("umap"):
                embedding_visualizer.show_umap()

        if plots.get("attention_tree") and hasattr(self, "_attn_weights"):
            attention_viz = TreeAttentionVisualizer(
                root=self.root,
                node_names=self._node_names,
                attn_weights=self._attn_weights
            )
            attention_viz.show()
