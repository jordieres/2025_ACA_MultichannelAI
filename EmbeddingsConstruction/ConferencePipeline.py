import torch 
from anytree import PreOrderIter
import pandas as pd
import numpy as np
from umap.umap_ import UMAP
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score

from .Embedding.FeatureExtractor import FeatureExtractor
from .Embedding.ConferenceEncoder import ConferenceEncoder
from .Embedding.NodeEncoder import NodeEncoder
from .Embedding.EmbeddingsVisualizer import NodeEmbeddingVisualizer

from .Tree.ConferenceTreeBuilder import ConferenceTreeBuilder
from .Tree.ConferenceTreeVisualizer import ConferenceTreeVisualizer

def plot_tree_attention(root, node_names, attn_weights, label_angle=30):
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.colors import Normalize
    from matplotlib import cm
    from anytree import PreOrderIter

    def hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
        pos = {}
        def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
            if pos is None: pos = {}
            children = list(G.successors(root))
            if not children:
                pos[root] = (xcenter, vert_loc)
            else:
                dx = width / len(children)
                nextx = xcenter - width / 2 - dx / 2
                for child in children:
                    nextx += dx
                    pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                         vert_loc=vert_loc - vert_gap, xcenter=nextx, pos=pos, parent=root)
                pos[root] = (xcenter, vert_loc)
            return pos
        return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

    G = nx.DiGraph()
    node_colors = {}
    node_sizes = {}

    # Normalizar pesos
    norm = Normalize(vmin=0, vmax=max(attn_weights))
    cmap = cm.viridis

    attn_dict = dict(zip(node_names, attn_weights))

    for node in PreOrderIter(root):
        G.add_node(node.name)
        if node.parent:
            G.add_edge(node.parent.name, node.name)

        # Si es hoja y tiene atención, asignar color y tamaño
        if node.is_leaf and node.name in attn_dict:
            attn = attn_dict[node.name]
            node_colors[node.name] = cmap(norm(attn))
            node_sizes[node.name] = 1500 + 3000 * attn
        else:
            node_colors[node.name] = "#DDDDDD"  # gris para nodos internos
            node_sizes[node.name] = 800

    pos = hierarchy_pos(G, root.name)

    # Crear figura y eje
    fig, ax = plt.subplots(figsize=(20, 8))

    # Dibujar nodos y bordes
    nx.draw(
        G, pos, ax=ax,
        node_color=[node_colors[n] for n in G.nodes()],
        node_size=[node_sizes[n] for n in G.nodes()],
        edge_color='gray',
        with_labels=False
    )

    # Etiquetas
    for node_name, (x, y) in pos.items():
        ax.text(x, y, node_name,
                rotation=label_angle,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=9, fontweight='bold')

    # Añadir colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label("Atención asignada (solo nodos hoja)", fontsize=10)

    ax.set_title("🌳 Árbol de la Conferencia con Atención por Nodo", fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

class ConferenceEmbeddingPipeline:
    def __init__(self, node_encoder_params: dict, conference_encoder_params: dict, device: str = "cpu"):
        self.device = torch.device(device)
        self.node_encoder = NodeEncoder(self.device, **node_encoder_params).to(self.device)
        self.conference_encoder = ConferenceEncoder(self.device, **conference_encoder_params).to(self.device)
        self.extractor = FeatureExtractor(
            categories_10k=self.node_encoder.categories_10k,
            qa_categories=self.node_encoder.qa_categories,
            max_num_coherences=self.node_encoder.max_num_coherences
        )
        
    # def generate_embedding(self, json_path: str) -> torch.Tensor:
    #     builder = ConferenceTreeBuilder(json_path)
    #     self.root = builder.build_tree()

    #     self._node_embeddings = []
    #     self._node_names = []
    #     self._node_types = []
    #     self._categories_10k = []

    #     for node in PreOrderIter(self.root):
    #         if node.is_leaf and node.node_type in {"monologue", "question", "answer"}:
    #             frases, mask, meta_vec = self.extractor.extract(node)
    #             frase_summary = self.node_encoder.frase_encoder(frases.to(self.device), mask.to(self.device))
    #             meta_summary = self.node_encoder.meta_proj(
    #                 torch.tensor(meta_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
    #             )
    #             combined = torch.cat([frase_summary, meta_summary], dim=-1)
    #             node_embedding = self.node_encoder.output_proj(combined).squeeze(0)
    #             self._node_embeddings.append(node_embedding)
    #             self._node_names.append(node.name)
    #             self._node_types.append(node.node_type)

    #             if node.node_type == "monologue":
    #                 self._categories_10k.append("None")
    #             else:
    #                 cls = node.metadata.get("classification", {})
    #                 self._categories_10k.append(cls.get("Predicted_category", "None"))

    #     if not self._node_embeddings:
    #         return torch.zeros(self.node_encoder.d_output)

    #     stacked = torch.stack(self._node_embeddings, dim=0)
    #     return self.conference_encoder(stacked)
    def generate_embedding(self, json_path: str, return_attn: bool = False):
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
                meta_summary = self.node_encoder.meta_proj(
                    torch.tensor(meta_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
                )
                combined = torch.cat([frase_summary, meta_summary], dim=-1)
                node_embedding = self.node_encoder.output_proj(combined).squeeze(0)
                self._node_embeddings.append(node_embedding)
                self._node_names.append(node.name)
                self._node_types.append(node.node_type)

                if node.node_type == "monologue":
                    self._categories_10k.append("None")
                else:
                    cls = node.metadata.get("classification", {})
                    self._categories_10k.append(cls.get("Predicted_category", "None"))

        if not self._node_embeddings:
            return torch.zeros(self.node_encoder.d_output)

        stacked = torch.stack(self._node_embeddings, dim=0)
        if return_attn:
            conference_embedding, attn_weights = self.conference_encoder(stacked, return_attn=True)
            self._attn_weights = attn_weights
            return conference_embedding
        return self.conference_encoder(stacked)

    def visualize(self, plots: dict = None):
        plots = plots or {}
        visualizer = ConferenceTreeVisualizer(self.root)

        if plots.get("tree_structure", False):
            visualizer.show_text_tree()

        if plots.get("plot", False):
            visualizer.show_networkx_tree()

        if any(plots.get(k, False) for k in ("silhouette", "umap")):
            embedding_visualizer = NodeEmbeddingVisualizer(
                embeddings=self._node_embeddings,
                node_names=self._node_names,
                node_types=self._node_types,
                categories_10k=self._categories_10k
            )

            if plots.get("silhouette", False):
                embedding_visualizer.show_metrics()

            if plots.get("umap", False):
                embedding_visualizer.show_umap()

        if plots.get("attention_tree", False) and hasattr(self, "_attn_weights"):
            plot_tree_attention(
                root=self.root,
                node_names=self._node_names,
                attn_weights=self._attn_weights
            )




