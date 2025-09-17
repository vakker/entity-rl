import random
import time
from typing import Optional, Tuple

import torch
from torch_geometric.data import Data

from .graph_utils import create_edges, create_empty_graph, create_node_features
from .mot_base import MOTBaseDataset
from .mot_data import check_rectangle_overlap, scale_bboxes


class MOTGraphDataset(MOTBaseDataset):
    """
    Dataset that creates graph representations from MOT ground truth data for GNN training.

    Each sample consists of:
    - A graph with nodes representing detected entities (bounding boxes)
    - Node features encoding position, size, and visual information
    - Edge connectivity based on spatial relationships
    - A reward signal based on synthetic agent placement
    """

    def __init__(
        self,
        mot_data_dirs,
        agent_radius: int = 15,
        num_samples_per_epoch: int = 1000,
        image_size: Tuple[int, int] = (100, 100),
        max_entities: int = 20,
        connect_threshold: float = 50.0,
        use_gt: bool = True,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize the MOT GNN dataset.

        Args:
            mot_data_dirs: List of MOT data directories
            agent_radius: Radius of the synthetic agent for collision detection
            num_samples_per_epoch: Number of samples per epoch
            image_size: Target image size (width, height)
            max_entities: Maximum number of entities to include per sample
            connect_threshold: Distance threshold for connecting entities in the graph
            use_gt: Whether to use ground truth (gt.txt) or detections (det.txt)
        """
        self.max_entities = max_entities
        self.connect_threshold = connect_threshold

        super().__init__(
            mot_data_dirs,
            agent_radius,
            num_samples_per_epoch,
            image_size,
            use_gt,
            max_samples,
        )

    def _generate_sample(self) -> Tuple[Data, int]:
        """
        Generate a single graph sample.

        Returns:
            Tuple of (graph_data, reward)
        """
        # Select random frame and bboxes
        data_dir, frame_id, original_bboxes = self.select_random_frame()

        # Limit number of entities
        if len(original_bboxes) > self.max_entities:
            original_bboxes = random.sample(original_bboxes, self.max_entities)

        # Get original dimensions and scale bboxes
        orig_w, orig_h = self.data_loader.get_image_dimensions(data_dir, frame_id)
        scaled_bboxes = scale_bboxes(original_bboxes, (orig_w, orig_h), self.image_size)

        # Create node features
        node_features = create_node_features(
            scaled_bboxes, self.image_size, (orig_w, orig_h)
        )

        # Create edges
        edge_index = create_edges(
            node_features, self.connect_threshold, self.image_size
        )

        agent_x, agent_y = self.generate_agent_position()

        # Generate synthetic agent position and check collision
        reward = self._compute_reward(scaled_bboxes, agent_x, agent_y)

        # Create PyTorch Geometric Data object
        graph_data = self._create_graph_data(node_features, edge_index)

        return graph_data, reward

    def _create_graph_data(
        self, node_features: torch.Tensor, edge_index: torch.Tensor
    ) -> Data:
        """Create PyTorch Geometric Data object."""
        if len(node_features) == 0:
            # Handle empty graphs with dummy node
            return create_empty_graph(node_feature_dim=6)

        return Data(
            x=node_features,
            edge_index=edge_index,
            num_nodes=len(node_features),
        )
