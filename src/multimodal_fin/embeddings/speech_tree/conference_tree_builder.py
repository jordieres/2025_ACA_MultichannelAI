import json
import re
import logging
from typing import Dict

from multimodal_fin.embeddings.speech_tree.conference_node import ConferenceNode

logger = logging.getLogger(__name__)


class ConferenceTreeBuilder:
    """
    Builds a hierarchical tree of a financial conference using the `ConferenceNode` class.

    This includes:
    - A root node for the whole conference.
    - Monologue nodes as direct children.
    - QA pair nodes, each with a question and an answer node.

    Attributes:
        json_path (str): Path to the JSON file containing the conference data.
    """

    def __init__(self, json_path: str):
        """
        Args:
            json_path (str): Path to the conference JSON file.
        """
        self.json_path = json_path

    def build_tree(self) -> ConferenceNode:
        """
        Builds and returns the root node of the conference tree.

        Returns:
            ConferenceNode: Root node with full tree structure as children.
        """
        logger.info(f"Building conference tree from: {self.json_path}")

        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        root = ConferenceNode(name="Conference", node_type="root")

        # Add monologue nodes
        for key, monologue in data.get("monologue_interventions", {}).items():
            node = ConferenceNode(
                name=f"Monologue_{key}",
                node_type="monologue",
                text_embeddings=monologue['multimodal_embeddings']['text'],
                audio_embeddings=monologue['multimodal_embeddings']['audio'],
                video_embeddings=monologue['multimodal_embeddings']['video'],
                num_sentences=monologue['multimodal_embeddings']['num_sentences'],
                metadata={"text": monologue.get("text", "")}
            )
            node.parent = root
            logger.debug(f"Added monologue node: {node.name}")

        # Sort QA pairs
        pair_keys = sorted(
            [k for k in data.keys() if re.match(r"pair_\d+", k)],
            key=lambda x: int(x.split("_")[1])
        )

        for pair_key in pair_keys:
            pair = data[pair_key]

            # Create QA pair parent node
            pair_node = ConferenceNode(name=pair_key, node_type="qa_pair")
            pair_node.parent = root
            logger.debug(f"Added QA pair node: {pair_node.name}")

            # Question node
            q_node = ConferenceNode(
                name=f"{pair_key}_Question",
                node_type="question",
                text_embeddings=pair['multimodal_embeddings']['question']['text'],
                audio_embeddings=pair['multimodal_embeddings']['question']['audio'],
                video_embeddings=pair['multimodal_embeddings']['question']['video'],
                num_sentences=pair['multimodal_embeddings']['question']['num_sentences'],
                metadata={
                    "text": pair.get("Question", ""),
                    "classification": pair.get("question_classification", {})
                }
            )
            q_node.parent = pair_node
            logger.debug(f"Added question node: {q_node.name}")

            # Answer node
            a_node = ConferenceNode(
                name=f"{pair_key}_Answer",
                node_type="answer",
                text_embeddings=pair['multimodal_embeddings']['answer']['text'],
                audio_embeddings=pair['multimodal_embeddings']['answer']['audio'],
                video_embeddings=pair['multimodal_embeddings']['answer']['video'],
                num_sentences=pair['multimodal_embeddings']['answer']['num_sentences'],
                metadata={
                    "text": pair.get("Answer", ""),
                    "classification": pair.get("answer_classification", {}),
                    "qa_response": pair.get("qa_response_classification", {}),
                    "coherence": pair.get("coherence_analyses", [])
                }
            )
            a_node.parent = pair_node
            logger.debug(f"Added answer node: {a_node.name}")

        logger.info("Conference tree construction complete.")
        return root
