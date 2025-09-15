from typing import Dict, List, Optional
from dataclasses import dataclass, field
from anytree import NodeMixin


@dataclass
class ConferenceNode(NodeMixin):
    """
    Represents a node in the hierarchical structure of a financial conference (monologue, question, or answer).

    Each node may contain multimodal embeddings (text, audio, video), metadata, and a reference
    to its parent node (used by `anytree` for tree traversal).
    """

    name: str
    """Unique identifier for the node."""

    node_type: str
    """Type of node: 'monologue', 'question', or 'answer'."""

    text_embeddings: Dict[str, List[List[float]]] = field(default_factory=dict)
    """Textual embeddings per sentence."""

    audio_embeddings: Dict[str, List[List[float]]] = field(default_factory=dict)
    """Audio embeddings per sentence."""

    video_embeddings: Dict[str, List[List[float]]] = field(default_factory=dict)
    """Video embeddings per sentence."""

    num_sentences: Optional[int] = None
    """Total number of sentences (optional)."""

    metadata: Dict = field(default_factory=dict)
    """Metadata dictionary for classification, QA response, coherence, etc."""

    def __post_init__(self):
        """
        Hook called automatically after initialization to set the parent to None.
        The parent will be assigned later when the tree is built.
        """
        self.parent = None
