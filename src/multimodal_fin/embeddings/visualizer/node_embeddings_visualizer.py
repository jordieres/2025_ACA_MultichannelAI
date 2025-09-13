import torch
import numpy as np
import pandas as pd
import logging
from sklearn.metrics import silhouette_score, davies_bouldin_score
from umap.umap_ import UMAP
from plotly.subplots import make_subplots
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

class NodeEmbeddingVisualizer:
    """
    Visualizes and evaluates node-level embeddings using clustering metrics and UMAP projection.

    Attributes:
        embeddings (List[torch.Tensor]): List of node embeddings.
        node_names (List[str]): Node identifiers or labels.
        node_types (List[str]): Node types (e.g., "question", "answer", "monologue").
        categories_10k (List[str]): SEC 10-K categories associated with each node.
    """

    def __init__(self, embeddings, node_names, node_types, categories_10k):
        """
        Initializes the visualizer with embeddings and associated metadata.

        Args:
            embeddings (List[torch.Tensor]): Embeddings per node.
            node_names (List[str]): Node identifiers.
            node_types (List[str]): Node type labels.
            categories_10k (List[str]): 10-K category labels.
        """
        self.embeddings = embeddings
        self.node_names = node_names
        self.node_types = node_types
        self.categories_10k = categories_10k

    def show_metrics(self):
        """
        Computes and logs clustering evaluation metrics (Silhouette and Davies-Bouldin)
        for both node type and 10-K category labels.
        """
        X = torch.stack(self.embeddings).detach().cpu().numpy()
        labels_type = np.array(self.node_types)
        labels_10k = np.array(self.categories_10k)

        silhouette_type = silhouette_score(X, labels_type, metric="cosine")
        silhouette_10k = silhouette_score(X, labels_10k, metric="cosine")
        dbi_type = davies_bouldin_score(X, labels_type)
        dbi_10k = davies_bouldin_score(X, labels_10k)

        logger.info(f"🔹 Davies-Bouldin (by node type): {dbi_type:.4f}")
        logger.info(f"🔹 Davies-Bouldin (by 10-K category): {dbi_10k:.4f}")
        logger.info(f"🔍 Silhouette score (by node type): {silhouette_type:.4f}")
        logger.info(f"🔍 Silhouette score (by 10-K category): {silhouette_10k:.4f}")

    def show_umap(self, n_neighbors=10, min_dist=0.1):
        """
        Displays a UMAP projection of the node embeddings colored by node type and 10-K category.

        Args:
            n_neighbors (int): Number of neighbors for UMAP.
            min_dist (float): Minimum distance between points in low-dimensional space.
        """
        X = torch.stack(self.embeddings).detach().cpu().numpy()
        umap_proj = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric='cosine', random_state=42)
        embeddings_2d = umap_proj.fit_transform(X)

        df = pd.DataFrame({
            "x": embeddings_2d[:, 0],
            "y": embeddings_2d[:, 1],
            "Name": self.node_names,
            "Node Type": self.node_types,
            "10-K Category": self.categories_10k
        })

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("UMAP by Node Type", "UMAP by 10-K Category"),
            horizontal_spacing=0.1
        )

        # Plot by Node Type
        for node_type in df["Node Type"].unique():
            subdf = df[df["Node Type"] == node_type]
            fig.add_trace(
                go.Scatter(
                    x=subdf["x"], y=subdf["y"],
                    mode='markers',
                    marker=dict(size=12),
                    name=node_type,
                    legendgroup="type",
                    hovertext=subdf["Name"],
                    hovertemplate="Name: %{text}<br>Type: " + node_type,
                    text=subdf["Name"]
                ),
                row=1, col=1
            )

        # Plot by 10-K Category
        for cat in sorted(df["10-K Category"].unique()):
            subdf = df[df["10-K Category"] == cat]
            color = 'lightgray' if cat == "None" else None
            fig.add_trace(
                go.Scatter(
                    x=subdf["x"], y=subdf["y"],
                    mode='markers',
                    marker=dict(size=12, color=color),
                    name=cat,
                    legendgroup="10k",
                    hovertext=subdf["Name"],
                    hovertemplate="Name: %{text}<br>10-K Category: " + cat,
                    text=subdf["Name"]
                ),
                row=1, col=2
            )

        fig.update_layout(
            title_text="UMAP Projection of Node Embeddings",
            height=600,
            width=1200,
            showlegend=True,
            legend=dict(font=dict(size=14)),
            font=dict(size=14),
            margin=dict(t=60)
        )

        fig.update_annotations(font=dict(size=20))
        fig.update_xaxes(title_text="UMAP Dimension 1", title_font=dict(size=18), row=1, col=1)
        fig.update_xaxes(title_text="UMAP Dimension 1", title_font=dict(size=18), row=1, col=2)
        fig.update_yaxes(title_text="UMAP Dimension 2", title_font=dict(size=18), row=1, col=1)
        fig.update_yaxes(title_text="UMAP Dimension 2", title_font=dict(size=18), row=1, col=2)

        logger.info("📊 Showing UMAP projection of node embeddings...")
        fig.show()
