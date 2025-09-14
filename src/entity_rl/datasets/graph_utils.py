"""
Graph utilities for GNN-based MOT datasets.

This module provides utilities for creating graph representations from bounding boxes.
"""

import gymnasium as gym
import numpy as np
import torch
from torch_geometric.data import Batch, Data
from typing import Dict, List, Tuple


def create_node_features(
    bboxes: List[Tuple[int, int, int, int, int]],
    target_size: Tuple[int, int],
    original_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Create node features from bounding boxes.

    Args:
        bboxes: List of (x, y, w, h, track_id) tuples
        target_size: Target image size (width, height)
        original_size: Original image size (width, height)

    Returns:
        Node feature tensor of shape (num_nodes, 6)
        Features: [center_x, center_y, width, height, area, aspect_ratio]
    """
    if not bboxes:
        return torch.zeros((0, 6))

    features = []
    orig_w, orig_h = original_size
    target_w, target_h = target_size

    for x, y, w, h, _ in bboxes:
        # Normalize coordinates to [0, 1] based on target image size
        norm_x = (x / orig_w) * (target_w / target_w)
        norm_y = (y / orig_h) * (target_h / target_h)
        norm_w = (w / orig_w) * (target_w / target_w)
        norm_h = (h / orig_h) * (target_h / target_h)

        # Center coordinates
        center_x = norm_x + norm_w / 2
        center_y = norm_y + norm_h / 2

        # Calculate area and aspect ratio
        area = norm_w * norm_h
        aspect_ratio = norm_w / (norm_h + 1e-6)

        # Feature vector
        node_feature = [center_x, center_y, norm_w, norm_h, area, aspect_ratio]
        features.append(node_feature)

    return torch.tensor(features, dtype=torch.float32)


def create_edges(
    node_features: torch.Tensor, connect_threshold: float, image_size: Tuple[int, int]
) -> torch.Tensor:
    """
    Create edge connectivity based on spatial proximity.

    Args:
        node_features: Node features tensor (num_nodes, feature_dim)
        connect_threshold: Distance threshold for connecting nodes
        image_size: Image size for normalizing threshold

    Returns:
        Edge index tensor of shape (2, num_edges)
    """
    if len(node_features) <= 1:
        return torch.zeros((2, 0), dtype=torch.long)

    # Extract center coordinates (first 2 features)
    centers = node_features[:, :2]  # Shape: (num_nodes, 2)
    num_nodes = centers.shape[0]
    edge_indices = []

    # Normalize threshold to [0, 1] space
    normalized_threshold = connect_threshold / max(image_size)

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            dist = torch.norm(centers[i] - centers[j])
            if dist < normalized_threshold:
                edge_indices.extend([[i, j], [j, i]])  # Undirected edges

    if not edge_indices:
        return torch.zeros((2, 0), dtype=torch.long)

    return torch.tensor(edge_indices).t().contiguous()


def collate_graph_batch(batch):
    """
    Custom collate function for graph data.

    Args:
        batch: List of (obs_dict, reward) tuples where obs_dict contains graph data

    Returns:
        Tuple of (batched_obs_dict, reward_tensor)
    """
    obs_list, rewards = zip(*batch)

    # Create list of Data objects for batching
    data_list = []
    for obs in obs_list:
        data = Data(x=obs['x'], edge_index=obs['edge_index'])
        data_list.append(data)

    # Batch graphs using PyTorch Geometric's Batch
    batched_graphs = Batch.from_data_list(data_list)

    # Create batched observation dict
    batched_obs = {
        'x': batched_graphs.x,
        'edge_index': batched_graphs.edge_index,
        'batch': batched_graphs.batch
    }

    # Convert rewards to tensor
    reward_tensor = torch.tensor(rewards, dtype=torch.long)

    return batched_obs, reward_tensor


def create_graph_observation_space(node_feature_dim: int = 6) -> gym.spaces.Dict:
    """
    Create observation space for graph data.

    Args:
        node_feature_dim: Dimension of node features

    Returns:
        Dictionary observation space for graph data
    """
    return gym.spaces.Dict({
        'x': gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(None, node_feature_dim), dtype=np.float32
        ),
        'edge_index': gym.spaces.Box(
            low=0, high=np.inf,
            shape=(2, None), dtype=np.int64
        ),
        'batch': gym.spaces.Box(
            low=0, high=np.inf,
            shape=(None,), dtype=np.int64
        )
    })


def create_empty_graph(node_feature_dim: int = 6) -> Data:
    """
    Create an empty graph with a single dummy node.

    Args:
        node_feature_dim: Dimension of node features

    Returns:
        Empty graph data object
    """
    return Data(
        x=torch.zeros((1, node_feature_dim)),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        num_nodes=1
    )