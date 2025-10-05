from typing import List, Tuple

import gymnasium as gym
import numpy as np
import torch
from torch_geometric.data import Batch, Data


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
        norm_x = x / orig_w
        norm_y = y / orig_h
        norm_w = w / orig_w
        norm_h = h / orig_h

        # Center coordinates
        center_x = norm_x + norm_w / 2
        center_y = norm_y + norm_h / 2

        # Calculate area and aspect ratio
        area = norm_w * norm_h
        aspect_ratio = norm_w / (norm_h + 1e-6)

        # Feature vector
        node_feature = [center_x, center_y, norm_w, norm_h]
        # node_feature = [center_x, center_y, norm_w, norm_h, area, aspect_ratio]
        features.append(node_feature)

    return torch.tensor(features, dtype=torch.float32)


def create_edges(
    node_features: torch.Tensor,
    connect_threshold: float,
) -> torch.Tensor:
    """
    Create edge connectivity based on spatial proximity.

    Uses vectorized distance computation for efficiency.

    Args:
        node_features: Node features tensor (num_nodes, feature_dim)
        connect_threshold: Distance threshold for connecting nodes

    Returns:
        Edge index tensor of shape (2, num_edges)
    """
    if len(node_features) <= 1:
        return torch.zeros((2, 0), dtype=torch.long)

    # Extract center coordinates (first 2 features)
    centers = node_features[:, :2]  # Shape: (num_nodes, 2)
    num_nodes = centers.shape[0]

    # Vectorized distance computation - much faster than nested loops
    # Compute pairwise distances using broadcasting
    diff = centers.unsqueeze(1) - centers.unsqueeze(0)  # Shape: (N, N, 2)
    distances = torch.norm(diff, dim=2)  # Shape: (N, N)

    # Create adjacency mask (distances < threshold)
    adj_mask = distances < connect_threshold

    # Get indices where mask is True, excluding self-loops
    edge_indices = torch.nonzero(adj_mask & (distances > 0), as_tuple=False)

    if edge_indices.shape[0] == 0:
        return torch.zeros((2, 0), dtype=torch.long)

    # Transpose to get shape (2, num_edges)
    return edge_indices.t().contiguous()


def collate_graph_batch(batch):
    """
    Custom collate function for graph data and optional image data.

    Supports both:
    - Graph-only mode: obs_dict contains {x, edge_index, batch}
    - Image+Graph mode: obs_dict contains {image, x, edge_index, batch}

    Args:
        batch: List of (obs_dict, reward, agent_pos) tuples

    Returns:
        Tuple of (batched_obs_dict, reward_tensor, agent_pos_tensor)
    """
    obs_list, rewards, agent_pos = zip(*batch)

    # Check if we have image data
    has_images = "image" in obs_list[0]

    # Create list of Data objects for batching graphs
    data_list = []

    for obs in obs_list:
        data = Data(x=obs["x"], edge_index=obs["edge_index"])
        data_list.append(data)

    # Batch graphs using PyTorch Geometric's Batch
    batched_graphs = Batch.from_data_list(data_list)

    # Create batched observation dict
    batched_obs = {
        "x": batched_graphs.x,
        "edge_index": batched_graphs.edge_index,
        "batch": batched_graphs.batch,
    }

    # If we have images, batch them as well
    if has_images:
        # Stack images into batch: (B, C, H, W)
        images = [obs["image"] for obs in obs_list]
        batched_obs["image"] = torch.stack(images)

    # Convert rewards to tensor
    reward_tensor = torch.tensor(rewards, dtype=torch.float32)
    agent_pos = torch.stack(agent_pos)

    return batched_obs, reward_tensor, agent_pos


def create_graph_observation_space(
    node_feature_dim: int = 6, max_elements: int = 80
) -> gym.spaces.Dict:
    """
    Create observation space for graph data, matching SPG environment format.

    Args:
        node_feature_dim: Dimension of node features
        max_elements: Maximum number of elements/nodes

    Returns:
        Dictionary observation space for graph data
    """
    from ray.rllib.utils.spaces.repeated import Repeated

    return gym.spaces.Dict(
        {
            "x": Repeated(
                gym.spaces.Box(-1, 1, shape=(node_feature_dim,), dtype=np.float32),
                max_elements,
            ),
            "edge_index": Repeated(
                gym.spaces.Box(0, max_elements, shape=(2,), dtype=np.int64),
                max_elements**2,
            ),
        }
    )


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
        num_nodes=1,
    )
