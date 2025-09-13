import random
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Union

from anytree import PreOrderIter
from multimodal_fin.speech_tree.ConferenceTreeBuilder import ConferenceTreeBuilder
from multimodal_fin.speech_tree.ConferenceNode import ConferenceNode


class NodeContrastiveDataset(Dataset):
    """
    Dataset for contrastive training of node-level embeddings using two augmented views
    of the same node (monologue, question or answer).

    It loads nodes from a list of JSON conference files and generates fixed-shape
    multimodal input tensors from their embeddings.

    Attributes:
        nodes (List[ConferenceNode]): Flattened list of leaf nodes from all trees.
    """

    def __init__(self, json_paths: List[str]):
        """
        Args:
            json_paths (List[str]): List of paths to JSON files with conference data.
        """
        self.nodes: List[ConferenceNode] = []
        for path in json_paths:
            try:
                builder = ConferenceTreeBuilder(path)
                root_node = builder.build_tree()

                # Extract only leaf nodes of interest
                leaf_nodes = [
                    n for n in PreOrderIter(root_node)
                    if n.node_type in {"monologue", "question", "answer"}
                ]
                self.nodes.extend(leaf_nodes)

            except Exception as e:
                print(f"❌ Error processing {path}: {e}")

    def __len__(self) -> int:
        return len(self.nodes)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns two views of the same node for contrastive training.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Two tensors of shape [6, 21]
        """
        node = self.nodes[idx]
        view1 = self._augment(node)
        view2 = self._augment(node)
        return view1, view2

    def _augment(self, node: ConferenceNode) -> torch.Tensor:
        """
        Augments a node by sampling or padding its multimodal embeddings
        to a fixed shape.

        Args:
            node (ConferenceNode): The node to augment.

        Returns:
            torch.Tensor: A tensor of shape [6, 21] representing 6 token embeddings
                          with 21 features (7 text + 7 audio + 7 video).
        """

        def sample_modality(modality_data: Union[List, dict]) -> np.ndarray:
            """
            Samples or pads embeddings of a modality to shape [6, 7].

            Args:
                modality_data (Union[List, dict]): Raw embeddings.

            Returns:
                np.ndarray: Array of shape [6, 7]
            """
            if not modality_data or len(modality_data) == 0:
                return np.zeros((6, 7), dtype=np.float32)

            try:
                mat = np.array(modality_data)
                if mat.ndim != 2 or mat.shape[1] != 7:
                    return np.zeros((6, 7), dtype=np.float32)

                if mat.shape[0] >= 6:
                    idx = sorted(random.sample(range(mat.shape[0]), 6))
                    mat = mat[idx]
                else:
                    pad_len = 6 - mat.shape[0]
                    pad = np.zeros((pad_len, 7), dtype=np.float32)
                    mat = np.vstack([mat, pad])

                return mat
            except Exception:
                return np.zeros((6, 7), dtype=np.float32)

        text = sample_modality(node.text_embeddings)
        audio = sample_modality(node.audio_embeddings)
        video = sample_modality(node.video_embeddings)

        frases = np.concatenate([text, audio, video], axis=1)  # [6, 21]
        return torch.tensor(frases, dtype=torch.float32)
