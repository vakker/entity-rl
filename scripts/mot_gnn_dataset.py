"""
MOT-based dataset for GNN training using ground truth bounding boxes.

This dataset creates graph representations directly from MOT ground truth annotations,
bypassing entity extraction and focusing on the entity-based reasoning component.
"""

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class MOTGNNDataset(Dataset):
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
        mot_data_dirs: List[str],
        agent_radius: int = 15,
        num_samples_per_epoch: int = 1000,
        image_size: Tuple[int, int] = (100, 100),
        max_entities: int = 20,
        connect_threshold: float = 50.0,
        use_gt: bool = True,
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
        self.mot_data_dirs = mot_data_dirs
        self.agent_radius = agent_radius
        self.num_samples_per_epoch = num_samples_per_epoch
        self.image_size = image_size
        self.max_entities = max_entities
        self.connect_threshold = connect_threshold
        self.use_gt = use_gt

        # Load MOT data
        self.mot_data = self._load_mot_data()

        if not self.mot_data:
            raise ValueError("No valid MOT data found in the provided directories")

        print(f"Loaded MOT data from {len(self.mot_data_dirs)} directories")
        print(
            f"Total frames available: {sum(len(frames) for frames in self.mot_data.values())}"
        )

    def _load_mot_data(self) -> Dict[str, Dict[int, List[Tuple[int, int, int, int, int]]]]:
        """
        Load MOT annotation data.

        Returns:
            Dictionary mapping data_dir -> {frame_id: [(x, y, w, h, track_id), ...]}
        """
        mot_data = {}

        for data_dir in self.mot_data_dirs:
            data_path = Path(data_dir)

            # Determine annotation file path
            if self.use_gt:
                ann_file = data_path / "gt" / "gt.txt"
            else:
                ann_file = data_path / "det" / "det.txt"

            img_dir = data_path / "img1"

            if not ann_file.exists() or not img_dir.exists():
                print(f"Warning: Missing files for {data_dir}")
                continue

            # Parse MOT annotations
            frame_data = {}
            try:
                with open(ann_file, "r") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) < 6:
                            continue

                        frame_id = int(row[0])
                        track_id = int(row[1]) if len(row) > 1 else -1
                        x, y, w, h = map(int, row[2:6])

                        # Filter invalid boxes
                        if w <= 0 or h <= 0:
                            continue

                        if frame_id not in frame_data:
                            frame_data[frame_id] = []
                        frame_data[frame_id].append((x, y, w, h, track_id))

            except Exception as e:
                print(f"Error reading {ann_file}: {e}")
                continue

            # Only keep frames with corresponding images
            valid_frame_data = {}
            for frame_id, bboxes in frame_data.items():
                img_path = img_dir / f"{frame_id:06d}.jpg"
                if img_path.exists() and len(bboxes) > 0:
                    valid_frame_data[frame_id] = bboxes

            if valid_frame_data:
                mot_data[str(data_dir)] = valid_frame_data
                print(f"Loaded {len(valid_frame_data)} frames from {data_dir}")

        return mot_data

    def _create_node_features(
        self,
        bboxes: List[Tuple[int, int, int, int, int]],
        orig_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Create node features from bounding boxes.

        Args:
            bboxes: List of (x, y, w, h, track_id) tuples
            orig_size: Original image size (width, height)

        Returns:
            Node feature tensor of shape (num_nodes, feature_dim)
        """
        if not bboxes:
            # Return empty tensor if no bboxes
            return torch.zeros((0, 6))

        features = []
        orig_w, orig_h = orig_size
        target_w, target_h = self.image_size

        for x, y, w, h, track_id in bboxes:
            # Normalize coordinates to [0, 1] based on target image size
            norm_x = (x / orig_w) * (target_w / target_w)  # Relative to target size
            norm_y = (y / orig_h) * (target_h / target_h)
            norm_w = (w / orig_w) * (target_w / target_w)
            norm_h = (h / orig_h) * (target_h / target_h)

            # Center coordinates
            center_x = norm_x + norm_w / 2
            center_y = norm_y + norm_h / 2

            # Feature vector: [center_x, center_y, width, height, area, aspect_ratio]
            area = norm_w * norm_h
            aspect_ratio = norm_w / (norm_h + 1e-6)

            node_feature = [center_x, center_y, norm_w, norm_h, area, aspect_ratio]
            features.append(node_feature)

        return torch.tensor(features, dtype=torch.float32)

    def _create_edges(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        Create edge connectivity based on spatial proximity.

        Args:
            node_features: Node features tensor (num_nodes, feature_dim)

        Returns:
            Edge index tensor of shape (2, num_edges)
        """
        if len(node_features) <= 1:
            return torch.zeros((2, 0), dtype=torch.long)

        # Extract center coordinates (first 2 features)
        centers = node_features[:, :2]  # Shape: (num_nodes, 2)

        # Calculate pairwise distances
        num_nodes = centers.shape[0]
        edge_indices = []

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                dist = torch.norm(centers[i] - centers[j])
                # Connect nodes if they're within threshold (normalized distance)
                if dist < (self.connect_threshold / max(self.image_size)):
                    edge_indices.extend([[i, j], [j, i]])  # Undirected edges

        if not edge_indices:
            return torch.zeros((2, 0), dtype=torch.long)

        return torch.tensor(edge_indices).t().contiguous()

    def _check_agent_collision(
        self,
        agent_x: int,
        agent_y: int,
        bboxes: List[Tuple[int, int, int, int, int]]
    ) -> bool:
        """Check if agent collides with any bounding box."""
        agent_left = agent_x - self.agent_radius
        agent_right = agent_x + self.agent_radius
        agent_top = agent_y - self.agent_radius
        agent_bottom = agent_y + self.agent_radius

        for x, y, w, h, _ in bboxes:
            if (agent_left < x + w and
                agent_right > x and
                agent_top < y + h and
                agent_bottom > y):
                return True

        return False

    def _generate_sample(self) -> Tuple[Data, int]:
        """
        Generate a single graph sample.

        Returns:
            Tuple of (graph_data, reward)
        """
        # Randomly select data and frame
        data_dir = random.choice(list(self.mot_data.keys()))
        frame_ids = list(self.mot_data[data_dir].keys())
        frame_id = random.choice(frame_ids)

        # Get bounding boxes for this frame
        bboxes = self.mot_data[data_dir][frame_id]

        # Load image to get original dimensions
        img_path = Path(data_dir) / "img1" / f"{frame_id:06d}.jpg"
        img = cv2.imread(str(img_path))
        if img is None:
            orig_h, orig_w = 480, 640  # Fallback dimensions
        else:
            orig_h, orig_w = img.shape[:2]

        # Limit number of entities
        if len(bboxes) > self.max_entities:
            bboxes = random.sample(bboxes, self.max_entities)

        # Scale bboxes to target image size
        scale_x = self.image_size[0] / orig_w
        scale_y = self.image_size[1] / orig_h

        scaled_bboxes = []
        for x, y, w, h, track_id in bboxes:
            scaled_x = int(x * scale_x)
            scaled_y = int(y * scale_y)
            scaled_w = int(w * scale_x)
            scaled_h = int(h * scale_y)
            scaled_bboxes.append((scaled_x, scaled_y, scaled_w, scaled_h, track_id))

        # Create node features
        node_features = self._create_node_features(scaled_bboxes, (orig_w, orig_h))

        # Create edges
        edge_index = self._create_edges(node_features)

        # Generate synthetic agent position and check collision
        if len(scaled_bboxes) > 0:
            agent_x = random.randint(self.agent_radius, self.image_size[0] - self.agent_radius)
            agent_y = random.randint(self.agent_radius, self.image_size[1] - self.agent_radius)
            has_collision = self._check_agent_collision(agent_x, agent_y, scaled_bboxes)
            reward = -1 if has_collision else 1
        else:
            # No entities, always safe
            reward = 1

        # Create PyTorch Geometric Data object
        graph_data = Data(
            x=node_features,
            edge_index=edge_index,
            num_nodes=len(node_features) if len(node_features) > 0 else 1
        )

        # Handle empty graphs by adding a dummy node
        if len(node_features) == 0:
            graph_data.x = torch.zeros((1, 6))
            graph_data.edge_index = torch.zeros((2, 0), dtype=torch.long)
            graph_data.num_nodes = 1

        return graph_data, reward

    def __len__(self) -> int:
        return self.num_samples_per_epoch

    def __getitem__(self, idx: int) -> Tuple[Data, int]:
        """Get a sample."""
        return self._generate_sample()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MOT GNN Dataset")
    parser.add_argument(
        "--mot-dirs", nargs="+", required=True, help="List of MOT data directories"
    )
    parser.add_argument(
        "--num-samples", type=int, default=5, help="Number of test samples"
    )
    parser.add_argument(
        "--max-entities", type=int, default=10, help="Maximum entities per sample"
    )
    parser.add_argument(
        "--connect-threshold", type=float, default=50.0, help="Distance threshold for edges"
    )

    args = parser.parse_args()

    dataset = MOTGNNDataset(
        mot_data_dirs=args.mot_dirs,
        num_samples_per_epoch=args.num_samples,
        max_entities=args.max_entities,
        connect_threshold=args.connect_threshold
    )

    print(f"Dataset length: {len(dataset)}")

    # Test samples
    for i in range(min(3, args.num_samples)):
        graph_data, reward = dataset[i]
        print(f"Sample {i}:")
        print(f"  Nodes: {graph_data.num_nodes}, Edges: {graph_data.edge_index.shape[1]}")
        print(f"  Node features shape: {graph_data.x.shape}")
        print(f"  Reward: {reward}")
        print()