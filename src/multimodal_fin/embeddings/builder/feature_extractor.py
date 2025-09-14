import numpy as np
import torch
from typing import List, Optional, Tuple, Union

from multimodal_fin.utils.logging import get_logger

logger = get_logger(__name__)


class FeatureExtractor:
    """
    Extracts multimodal (text, audio, video) and metadata features from a conference tree node.
    Converts data into tensors suitable for model input.
    """

    def __init__(
        self,
        categories_10k: Optional[List[str]] = None,
        qa_categories: Optional[List[str]] = None,
        max_num_coherences: int = 5
    ):
        """
        Initializes the feature extractor with configuration.

        Args:
            categories_10k: List of 10-K section labels used for one-hot encoding.
            qa_categories: List of response types for QA classification.
            max_num_coherences: Maximum number of coherence entries to encode.
        """
        self.categories_10k = categories_10k or ["MD&A", "Risk Factors", "Business", "Other"]
        self.qa_categories = qa_categories or ["yes", "no", "partially"]
        self.max_num_coherences = max_num_coherences

        logger.info("✅ FeatureExtractor initialized")

    def to_onehot(self, value: str, options: List[str]) -> np.ndarray:
        """Converts a categorical value to a one-hot encoded vector."""
        vec = np.zeros(len(options), dtype=np.float32)
        if value in options:
            vec[options.index(value)] = 1.0
        return vec

    def to_onehot_bool(self, value: bool) -> np.ndarray:
        """Encodes a boolean value as a 1-hot vector [1, 0] or [0, 1]."""
        return np.array([0.0, 1.0], dtype=np.float32) if value else np.array([1.0, 0.0], dtype=np.float32)

    def safe_len(self, emb: Union[List, dict]) -> int:
        """Safely computes the length of embeddings regardless of structure."""
        if isinstance(emb, list):
            return len(emb)
        if isinstance(emb, dict):
            return max((len(v) for v in emb.values()), default=0)
        return 0

    def get_array_from_embedding(self, emb_data: Union[List, dict], n_target: int) -> np.ndarray:
        """
        Converts raw embeddings into a padded NumPy array of shape [n_target, 7].

        Args:
            emb_data: List or dict of raw embeddings.
            n_target: Desired number of time steps (padding/truncating applied).

        Returns:
            A NumPy array of shape [n_target, 7].
        """
        if isinstance(emb_data, list):
            arr = np.array(emb_data)
        elif isinstance(emb_data, dict):
            if not emb_data:
                return np.zeros((n_target, 7), dtype=np.float32)
            for v in emb_data.values():
                arr = np.array(v)
                if arr.ndim == 2 and arr.shape[1] == 7:
                    break
            else:
                return np.zeros((n_target, 7), dtype=np.float32)
        else:
            return np.zeros((n_target, 7), dtype=np.float32)

        if arr.ndim == 1:
            arr = arr.reshape(0, 7)

        if arr.shape[0] < n_target:
            pad = np.zeros((n_target - arr.shape[0], 7), dtype=np.float32)
            arr = np.vstack([arr, pad])

        return arr[:n_target]

    def extract(self, node) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """
        Extracts a multimodal tensor and metadata vector from a tree node.

        Args:
            node: A `ConferenceNode` object containing multimodal data and metadata.

        Returns:
            frases: Tensor of shape [1, n, 21] with concatenated features.
            mask: Boolean tensor of shape [1, n] indicating valid time steps.
            meta_vec: Array of metadata of shape [expected_size].
        """
        n_text = self.safe_len(node.text_embeddings)
        n_audio = self.safe_len(node.audio_embeddings)
        n_video = self.safe_len(node.video_embeddings)
        n = max(n_text, n_audio, n_video, 1)

        text = self.get_array_from_embedding(node.text_embeddings, n)
        audio = self.get_array_from_embedding(node.audio_embeddings, n)
        video = self.get_array_from_embedding(node.video_embeddings, n)

        frases = np.concatenate([text, audio, video], axis=1)  # [n, 21]
        frases_tensor = torch.tensor(frases, dtype=torch.float32).unsqueeze(0)  # [1, n, 21]
        mask_tensor = torch.ones((1, n), dtype=torch.bool)  # [1, n]

        meta = []

        # Classification metadata (10-K section)
        cls = node.metadata.get("classification", {})
        meta.append(float(cls.get("Confidence", 0.0)))
        meta.extend(self.to_onehot(cls.get("Predicted_category", "Other"), self.categories_10k))

        # QA response metadata
        qa = node.metadata.get("qa_response", {})
        meta.append(float(qa.get("Confidence", 0.0)))
        pred_cat = str(qa.get("Predicted_category", "")).lower()
        meta.extend(self.to_onehot(pred_cat, self.qa_categories))

        # Coherence metadata
        for coh in node.metadata.get("coherence", [])[:self.max_num_coherences]:
            meta.extend(self.to_onehot_bool(coh.get("consistent", False)))

        expected_size = 1 + len(self.categories_10k) + 1 + len(self.qa_categories) + 2 * self.max_num_coherences
        meta_vec = np.array(meta, dtype=np.float32)

        if len(meta_vec) < expected_size:
            meta_vec = np.pad(meta_vec, (0, expected_size - len(meta_vec)))
        elif len(meta_vec) > expected_size:
            meta_vec = meta_vec[:expected_size]

        logger.debug(f"Extracted features from node: {node.name}")
        return frases_tensor, mask_tensor, meta_vec
