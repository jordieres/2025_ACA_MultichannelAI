import logging
import matplotlib.pyplot as plt
import networkx as nx
from anytree import RenderTree, PreOrderIter

from multimodal_fin.embeddings.speech_tree.conference_node import ConferenceNode

logger = logging.getLogger(__name__)

class ConferenceTreeVisualizer:
    """
    Visualizes the hierarchical structure of a conference tree built with ConferenceNode objects.

    Supports both textual and graphical representations using `anytree` and `networkx`.
    """

    def __init__(self, root: ConferenceNode):
        """
        Initializes the visualizer with the root node of the conference tree.

        Args:
            root (ConferenceNode): Root of the conference tree.
        """
        self.root = root

    def show_text_tree(self):
        """
        Prints the structure of the conference tree as a plain-text hierarchy.
        """
        logger.info("\n📂 Conference Tree Structure:\n")
        for pre, _, node in RenderTree(self.root):
            print(f"{pre}{node.name} ({node.node_type})")

    def show_networkx_tree(self, label_angle: int = 30):
        """
        Displays the conference tree using NetworkX and Matplotlib.

        Args:
            label_angle (int, optional): Angle (in degrees) to rotate node labels. Default is 30.
        """
        def hierarchy_pos(G, root, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None):
            if pos is None:
                pos = {root: (xcenter, vert_loc)}
            else:
                pos[root] = (xcenter, vert_loc)

            children = list(G.successors(root))
            if children:
                dx = width / len(children)
                nextx = xcenter - width / 2 - dx / 2
                for child in children:
                    nextx += dx
                    pos = hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                        vert_loc=vert_loc - vert_gap, xcenter=nextx, pos=pos)
            return pos

        # Build the graph
        G = nx.DiGraph()
        for node in PreOrderIter(self.root):
            G.add_node(node.name, type=node.node_type)
            if node.parent:
                G.add_edge(node.parent.name, node.name)

        pos = hierarchy_pos(G, self.root.name)

        # Draw the graph
        plt.figure(figsize=(20, 8))
        nx.draw(
            G, pos,
            with_labels=False,
            arrows=True,
            node_size=2000,
            node_color='lightblue',
            edge_color='gray'
        )

        # Add labels manually for rotation
        for node_name, (x, y) in pos.items():
            plt.text(
                x, y, node_name,
                rotation=label_angle,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=9,
                fontweight='bold'
            )

        plt.axis('off')
        plt.tight_layout()
        logger.info("🧭 Rendering conference tree as NetworkX graph.")
        plt.show()
