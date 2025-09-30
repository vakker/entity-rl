import random
import time
from typing import List, Optional, Tuple

import torch
from torch_geometric.data import Data

from .graph_utils import create_edges, create_empty_graph, create_node_features
from .mot_base import MOTBaseDataset
from .mot_data import MOTDataLoader, check_rectangle_overlap, scale_bboxes


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
        agent_radius: float = 0.02,
        num_samples_per_epoch: int = 1000,
        image_size: Tuple[int, int] = (100, 100),
        max_entities: int = 20,
        connect_threshold: float = 50.0,
        max_samples: Optional[int] = None,
        include_agent_node: bool = True,
        use_props: bool = False,
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
            max_samples: Maximum number of samples to load from data
            include_agent_node: Whether to include agent as a node in the graph
            use_props: Whether to use proposals for graph creation (always uses GT for rewards)
        """
        self.max_entities = max_entities
        self.connect_threshold = connect_threshold
        self.include_agent_node = include_agent_node
        self.use_props = use_props

        # Always load GT data for reward calculation
        self.gt_data_loader = MOTDataLoader(
            mot_data_dirs, use_gt=True, max_samples=max_samples
        )
        # FIXME: for compatibility
        self.gt_data = self.gt_data_loader.mot_data()

        # Initialize graph data loader based on use_props setting
        if use_props:
            # Use proposals for graph creation
            self.props_data_loader = MOTDataLoader(
                mot_data_dirs, use_gt=False, max_samples=max_samples
            )
            # FIXME: for compatibility
            self.props_data = self.props_data_loader.mot_data()

            entities = self.gt_data_loader.get_entities()
            print("#### GT")
            print(
                f"Max detections: {max(entities)}, Min detections: {min(entities)}, Avg detections: {sum(entities) / len(entities)}"
            )
            entities = self.props_data_loader.get_entities()
            print("#### Props")
            print(
                f"Max detections: {max(entities)}, Min detections: {min(entities)}, Avg detections: {sum(entities) / len(entities)}"
            )
        else:
            # Use GT for both graph and reward (standard mode)
            self.props_data_loader = None
            self.props_data = None

            entities = self.gt_data_loader.get_entities()
            print("#### GT")
            print(
                f"Max detections: {max(entities)}, Min detections: {min(entities)}, Avg detections: {sum(entities) / len(entities)}"
            )
            print("#### Props - none")

        # Initialize base class with simplified parameters
        super().__init__(
            mot_data_dirs,
            agent_radius,
            num_samples_per_epoch,
            image_size,
            use_gt=True,  # Always use GT for base class (reward calculation)
            max_samples=max_samples,
        )

    def select_random_frame_with_props(
        self,
    ) -> Tuple[str, int, List[Tuple], List[Tuple]]:
        """
        Select a random frame and return both proposals and GT bboxes.

        Returns:
            Tuple of (data_dir, frame_id, props_bboxes, gt_bboxes)
        """
        if not self.use_props:
            raise ValueError(
                "select_random_frame_with_props only available when use_props=True"
            )

        # Select from props data (which is the main data when use_props=True)
        data_dir = random.choice(list(self.props_data.keys()))
        frame_ids = list(self.props_data[data_dir].keys())
        frame_id = random.choice(frame_ids)

        props_bboxes = self.props_data[data_dir][frame_id]

        # Get corresponding GT data for the same frame
        if data_dir in self.gt_data and frame_id in self.gt_data[data_dir]:
            gt_bboxes = self.gt_data[data_dir][frame_id]
        else:
            raise ValueError(f"GT data not found for frame {frame_id} in {data_dir}")
            # gt_bboxes = []  # No GT data for this frame

        return data_dir, frame_id, props_bboxes, gt_bboxes

    def _generate_sample(self) -> Tuple[Data, int, torch.Tensor]:
        """
        Generate a single graph sample.

        Returns:
            Tuple of (graph_data, reward, agent_pos)
        """
        # Select frame based on use_props setting
        if self.use_props:
            # Use proposals for graph creation, GT for reward calculation
            self._timer.tic("select_random_frame_with_props")
            data_dir, frame_id, props_bboxes, gt_bboxes = (
                self.select_random_frame_with_props()
            )
            self._timer.toc("select_random_frame_with_props")
        else:
            # Use GT for both graph and reward
            self._timer.tic("select_random_frame")
            data_dir, frame_id, bboxes = self.select_random_frame()
            props_bboxes = bboxes
            gt_bboxes = bboxes
            self._timer.toc("select_random_frame")

        # Limit number of entities for graph (proposals)
        if len(props_bboxes) > self.max_entities:
            props_bboxes = random.sample(props_bboxes, self.max_entities)

        # Get original dimensions and scale both proposal and GT bboxes
        self._timer.tic("get_image_dimensions")
        orig_w, orig_h = self.gt_data_loader.get_image_dimensions(data_dir, frame_id)
        self._timer.toc("get_image_dimensions")

        self._timer.tic("scale_bboxes")
        if self.use_props:
            scaled_gt_bboxes = scale_bboxes(gt_bboxes, (orig_w, orig_h))
            scaled_props_bboxes = scale_bboxes(props_bboxes, (orig_w, orig_h))

        else:
            scaled_gt_bboxes = scale_bboxes(gt_bboxes, (orig_w, orig_h))
            scaled_props_bboxes = scaled_gt_bboxes

        self._timer.toc("scale_bboxes")

        # Generate synthetic agent position first
        self._timer.tic("generate_agent_position")
        agent_x, agent_y = self.generate_agent_position()
        self._timer.toc("generate_agent_position")

        # Create node features relative to agent position (like SPG environment)
        self._timer.tic("create_relative_node_features")
        node_features = self._create_relative_node_features(
            scaled_props_bboxes,
            agent_x,
            agent_y,
            self.agent_radius,
            self.include_agent_node,
        )
        self._timer.toc("create_relative_node_features")

        # Create edges
        self._timer.tic("create_edges")
        edge_index = create_edges(
            node_features,
            self.connect_threshold,
        )
        self._timer.toc("create_edges")

        # Compute reward based on agent collision with bounding boxes
        self._timer.tic("compute_reward")
        reward = self._compute_reward(scaled_gt_bboxes, agent_x, agent_y)
        self._timer.toc("compute_reward")

        # Create PyTorch Geometric Data object
        self._timer.tic("create_graph_data")
        graph_data = self._create_graph_data(node_features, edge_index)
        self._timer.toc("create_graph_data")

        # Create agent_pos tensor matching MOTVisDataset format
        self._timer.tic("create_agent_pos")
        agent_pos = torch.tensor(
            [agent_x, agent_y, self.agent_radius, self.agent_radius],
            dtype=torch.float32,
        )
        self._timer.toc("create_agent_pos")

        # print(graph_data.x, reward)
        return graph_data, reward, agent_pos

    def _create_relative_node_features(
        self,
        scaled_bboxes,
        agent_x: float,
        agent_y: float,
        agent_radius: float,
        include_agent_node: bool,
    ) -> torch.Tensor:
        """
        Create node features relative to agent position (like SPG environment).

        The SPG environment creates features as:
        - Agent node: [0, cos(angle), sin(angle), ...entity_type]
        - Detection nodes: [distance, cos(angle), sin(angle), ...entity_type]

        For MOT, we'll create:
        - Agent node: [0, 1, 0, 1] (distance=0, cos=1, sin=0, entity_type=1 for agent)
        - Obstacle nodes: [distance, cos(rel_angle), sin(rel_angle), 0] (entity_type=0 for obstacles)

        Args:
            include_agent_node: Whether to include the agent node in the graph
        """
        # return torch.tensor([[agent_x]], dtype=torch.float32)
        features = []

        # Add agent node first (like SPG does) - only if requested
        if include_agent_node:
            # Agent is at distance 0 from itself, facing "right" (angle=0)
            # agent_feature = [0.0, 1.0, 0.0, 1.0]  # [distance, cos, sin, is_agent]
            agent_feature = [0.0, 0.0, agent_radius, agent_radius, 1.0]
            # agent_feature = [agent_x, 0.0]
            features.append(agent_feature)

        # Add obstacle nodes relative to agent position
        for x, y, w, h, _ in scaled_bboxes:
            # NOTE: debugging
            # features.append([x, 1.0])
            # break

            # Calculate obstacle center
            obs_center_x = x + w / 2
            obs_center_y = y + h / 2

            # Calculate relative position from agent to obstacle center
            rel_x = obs_center_x - agent_x
            rel_y = obs_center_y - agent_y

            # Calculate distance (normalized to image diagonal)
            distance = (rel_x**2 + rel_y**2) ** 0.5

            # Calculate angle from agent to obstacle
            # if distance > 1e-6:
            #     cos_angle = rel_x / distance
            #     sin_angle = rel_y / distance
            # else:
            #     cos_angle = 1.0
            #     sin_angle = 0.0

            # Obstacle feature: [distance, cos(angle), sin(angle), is_agent=0]
            # obstacle_feature = [norm_distance, cos_angle, sin_angle, 0.0]
            obstacle_feature = [rel_x, rel_y, w, h, 0.0]
            features.append(obstacle_feature)

        return torch.tensor(features, dtype=torch.float32)

    def _create_graph_data(
        self, node_features: torch.Tensor, edge_index: torch.Tensor
    ) -> Data:
        """Create PyTorch Geometric Data object."""
        # Handle empty graph case (when include_agent_node=False and no entities)
        if len(node_features) == 0:
            return create_empty_graph(node_feature_dim=5)

        return Data(
            x=node_features,
            edge_index=edge_index,
            num_nodes=len(node_features),
        )
