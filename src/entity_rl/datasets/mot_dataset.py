import os
import random
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from entity_rl.utils import TicToc

from .graph_utils import create_edges, create_empty_graph
from .mot_data import (
    MOTDataLoader,
    check_rectangle_overlap,
    load_and_resize_image,
    scale_bboxes,
)

TIMERS_ENABLED = False


class MOTDataset(Dataset):
    """
    Unified MOT dataset supporting multiple output modes.

    Supports two output modes:
    - 'graph': Returns graph representation (nodes + edges) for GNN training
    - 'image': Returns image with agent drawn for vision-based training

    Each sample consists of:
    - Output data (graph or image depending on mode)
    - Reward signal based on synthetic agent placement
    - Agent position tensor
    """

    def __init__(
        self,
        mot_data_dirs: List[str],
        return_image: bool = False,
        agent_radius: float = 0.02,
        num_samples_per_epoch: int = 1000,
        image_size: Tuple[int, int] = (100, 100),
        max_samples: Optional[int] = None,
        task_type: str = "regression",
        # Graph-specific parameters
        max_entities: int = 20,
        connect_threshold: float = 50.0,
        include_agent_node: bool = True,
        visible_ann_filename: Optional[str] = None,
        gt_ann_filename: Optional[str] = None,
        use_precomputed_features: bool = False,
        feature_filename: Optional[str] = None,
        # Obstacle separation
        separate_obstacles: bool = False,
        # Prefetching parameters
        image_cache_size: int = 500,
    ):
        """
        Initialize unified MOT dataset.

        Args:
            mot_data_dirs: List of MOT data directories
            return_image: Whether to return image output
            agent_radius: Radius of synthetic agent for collision detection
            num_samples_per_epoch: Number of samples per epoch
            image_size: Target image size (width, height)
            max_samples: Maximum number of samples to load from data
            task_type: Task type - 'regression' or 'classification' (default: 'regression')
            max_entities: [Graph mode] Maximum entities per sample
            connect_threshold: [Graph mode] Distance threshold for graph edges
            include_agent_node: [Graph mode] Whether to include agent as graph node
            visible_ann_filename: Visible annotation filename relative to sequence
                directory. If None or 'gt.txt', GT is visible; otherwise the
                specified file is used.
            separate_obstacles: Whether to separate obstacles (class 0) from other entities
                (class 2) for collision detection. When True, only obstacles count for reward.
        """

        if task_type not in ["regression", "classification"]:
            raise ValueError(
                f"task_type must be 'regression' or 'classification', got '{task_type}'"
            )

        self.task_type = task_type
        self.return_image = return_image

        # Graph-specific parameters
        self.max_entities = max_entities
        self.connect_threshold = connect_threshold
        self.include_agent_node = include_agent_node
        self.use_precomputed_features = use_precomputed_features
        self.separate_obstacles = separate_obstacles

        # Always load GT data for reward calculation
        if gt_ann_filename is None:
            gt_ann_filename = "gt.csv"
        self.gt_data_loader = MOTDataLoader(
            mot_data_dirs,
            ann_filename=gt_ann_filename,
            max_samples=max_samples,
        )
        self.gt_data = self.gt_data_loader.mot_data()

        entities_gt = self.gt_data_loader.get_entities()
        print("#### GT")
        print(
            f"Max detections: {max(entities_gt)}, "
            f"Min detections: {min(entities_gt)}, "
            f"Avg detections: {sum(entities_gt) / len(entities_gt):.1f}"
        )

        # Determine visible annotations for graph
        self.visible_data_loader = None
        self.visible_data = None
        is_gt_visible = visible_ann_filename is None or visible_ann_filename in ["gt.txt", "gt.csv"]

        if is_gt_visible:
            self.visible_data_loader = None
            self.visible_data = self.gt_data
            print("#### Visible: GT")
        else:
            # Derive default features filename for CSV visible annotations when requested
            if use_precomputed_features and feature_filename is None:
                if isinstance(
                    visible_ann_filename, str
                ) and visible_ann_filename.endswith(".csv"):
                    feature_filename = visible_ann_filename.replace(
                        ".csv", "_features.npz"
                    )

            self.visible_data_loader = MOTDataLoader(
                mot_data_dirs,
                ann_filename=visible_ann_filename,
                max_samples=max_samples,
                feature_filename=feature_filename,
            )
            self.visible_data = self.visible_data_loader.mot_data()

            entities_visible = self.visible_data_loader.get_entities()
            print("#### Visible: Custom")
            print(
                f"Max detections: {max(entities_visible)}, "
                f"Min detections: {min(entities_visible)}, "
                f"Avg detections: {sum(entities_visible) / len(entities_visible):.1f}"
            )

        # Store parameters
        self.mot_data_dirs = mot_data_dirs
        self.agent_radius = agent_radius
        self.image_size = image_size

        # Calculate num_samples_per_epoch
        total_frames = self.gt_data_loader.get_total_frames()
        self.num_samples_per_epoch = min(total_frames, num_samples_per_epoch)

        print(f"Total frames available: {total_frames}")

        # Validate parameters
        self._validate_parameters()

        # Initialize timer
        self._timer = TicToc(enabled=TIMERS_ENABLED)

        # Initialize image cache (LRU cache using OrderedDict)
        self.image_cache_size = image_cache_size if return_image else 0
        self._image_cache: OrderedDict[Tuple[str, int], np.ndarray] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0

    def _validate_parameters(self) -> None:
        """Validate initialization parameters."""
        if not self.mot_data_dirs:
            raise ValueError("mot_data_dirs cannot be empty")

        if self.agent_radius <= 0:
            raise ValueError("agent_radius must be positive")

        if self.num_samples_per_epoch <= 0:
            raise ValueError("num_samples_per_epoch must be positive")

        if len(self.image_size) != 2 or any(s <= 0 for s in self.image_size):
            raise ValueError("image_size must be a tuple of two positive integers")

        # Check that agent can fit in image
        if (
            self.agent_radius * 2 >= self.image_size[0]
            or self.agent_radius * 2 >= self.image_size[1]
        ):
            raise ValueError("agent_radius too large for image_size")

    def select_random_frame(
        self,
        data_dir=None,
        frame_id=None,
    ) -> Tuple[str, int, List[Tuple], List[Tuple]]:
        """
        Select a random frame from the loaded data.

        Returns:
            Tuple of (data_dir, frame_id, bboxes)
        """
        if data_dir is None:
            data_dir = random.choice(list(self.gt_data.keys()))

        if frame_id is None:
            frame_ids = list(self.gt_data[data_dir].keys())
            frame_id = random.choice(frame_ids)

        gt_bboxes = self.gt_data[data_dir][frame_id]

        orig_w, orig_h = self.gt_data_loader.get_image_dimensions(data_dir, frame_id)
        scaled_gt_bboxes = scale_bboxes(gt_bboxes, (orig_w, orig_h))

        # Use GT when visible annotations are GT; otherwise use loaded visible annotations
        if self.visible_data is self.gt_data or self.visible_data_loader is None:
            visible_bboxes = gt_bboxes
            scaled_visible_bboxes = scaled_gt_bboxes
        else:
            visible_bboxes = self.visible_data[data_dir][frame_id]
            scaled_visible_bboxes = scale_bboxes(visible_bboxes, (orig_w, orig_h))

        return data_dir, frame_id, scaled_visible_bboxes, scaled_gt_bboxes

    def generate_agent_position(self) -> Tuple[float, float]:
        """
        Generate a random agent position within the image bounds.

        Returns:
            Tuple of (agent_x, agent_y)
        """
        agent_x = random.random()
        agent_y = random.random()

        return agent_x, agent_y

    def _compute_reward(self, scaled_bboxes, agent_x, agent_y) -> float:
        """
        Compute reward/label based on agent collision with bounding boxes.

        When separate_obstacles=True, only obstacles (class_id=0) count as collisions.
        When separate_obstacles=False, all entities count as collisions.

        Returns:
            - Regression: -1.0 (collision) or 1.0 (safe)
            - Classification: 0.0 (collision) or 1.0 (safe) for BCE loss
        """
        if len(scaled_bboxes) > 0:
            # Filter to obstacles only if separate_obstacles is enabled
            if self.separate_obstacles:
                # Only count obstacles (class_id=0) for collision
                obstacle_bboxes = [bbox for bbox in scaled_bboxes if bbox[5] == 0]
                has_collision = check_rectangle_overlap(
                    agent_x, agent_y, self.agent_radius, obstacle_bboxes
                )
            else:
                # All entities count for collision
                has_collision = check_rectangle_overlap(
                    agent_x, agent_y, self.agent_radius, scaled_bboxes
                )

            if self.task_type == "classification":
                return 0.0 if has_collision else 1.0  # BCE targets (float)
            else:  # regression
                return -1.0 if has_collision else 1.0  # Regression targets
        else:
            # No entities, always safe
            return 1.0

    def calculate_accuracy(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> int:
        """
        Calculate accuracy matches based on task type.

        Args:
            predictions: Model predictions
                - Regression: raw values (B,)
                - Classification: logits (B,) from BCEWithLogitsLoss
            labels: Ground truth labels
                - Regression: -1.0 or 1.0 (B,)
                - Classification: 0.0 or 1.0 (B,)

        Returns:
            Number of correct predictions (int for summing across batches)
        """
        if self.task_type == "classification":
            # For classification with BCEWithLogitsLoss:
            # logit >= 0 -> class 1 (safe), logit < 0 -> class 0 (collision)
            preds = (predictions >= 0).float()
            matches = (preds == labels).sum().item()
        else:  # regression
            # Threshold at 0: >= 0 is safe (1.0), < 0 is collision (-1.0)
            preds = torch.where(predictions >= 0, 1.0, -1.0)
            matches = (preds == labels).sum().item()

        return int(matches)

    def __len__(self) -> int:
        """Return the number of samples per epoch."""
        return self.num_samples_per_epoch

    def __getitem__(
        self, idx: int
    ) -> Tuple[Dict[str, torch.Tensor], Union[int, float], torch.Tensor]:
        """
        Get a sample by index.

        Args:
            idx: Sample index (ignored, samples are generated randomly)

        Returns:
            Tuple of (obs_dict, label, agent_pos) where:
            - obs_dict: Dict with "x" and "edge_index" for graph data
            - label: Classification label (int) or regression target (float)
            - agent_pos: Agent position tensor
        """
        self._timer.reset()
        self._timer.tic("generate_sample")
        data_dict, reward = self._generate_sample()
        self._timer.toc("generate_sample")

        # Extract graph and agent position
        graph_data = data_dict["graph"]
        agent_pos = data_dict["agent_pos"]

        # Format observation dict for model
        if self.return_image:
            # Image mode: return image as main observation
            obs_dict = {
                "image": data_dict["image"],
                "x": graph_data.x,
                "edge_index": graph_data.edge_index,
                "batch": torch.zeros(graph_data.num_nodes, dtype=torch.long),
            }
        else:
            # Graph mode: return only graph data
            obs_dict = {
                "x": graph_data.x,
                "edge_index": graph_data.edge_index,
                "batch": torch.zeros(graph_data.num_nodes, dtype=torch.long),
            }

        # Convert reward to appropriate type based on task
        if self.task_type == "classification":
            label = int(reward)  # Already 0 or 1
        else:  # regression
            label = float(reward)  # Already -1.0 or 1.0

        self._timer.print_stats(title="generate_sample")
        return obs_dict, label, agent_pos

    @property
    def total_frames(self) -> int:
        """Get total number of frames across all datasets."""
        return self.gt_data_loader.get_total_frames()

    def get_cache_stats(self) -> Dict[str, Union[int, float]]:
        """
        Get image cache statistics.

        Returns:
            Dictionary with cache stats (hits, misses, size, hit_rate)
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_size": len(self._image_cache),
            "cache_capacity": self.image_cache_size,
            "hit_rate": hit_rate,
        }

    def print_cache_stats(self) -> None:
        """Print cache statistics."""
        stats = self.get_cache_stats()
        print(f"\n=== Image Cache Statistics ===")
        print(f"Hits: {stats['cache_hits']}")
        print(f"Misses: {stats['cache_misses']}")
        print(f"Hit Rate: {stats['hit_rate']:.2%}")
        print(f"Cache Size: {stats['cache_size']}/{stats['cache_capacity']}")
        print("=" * 30)

    def _generate_sample(
        self,
        data_dir: str | None = None,
        frame_id: int | None = None,
        agent_pos: Tuple[float, float, float, float] | None = None,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Generate a single sample.

        Returns:
            Tuple of (data, reward, agent_pos) where:
            - data: Graph (Data) if output_mode='graph', Image (Tensor) if output_mode='image'
            - reward: int (0 for collision, +1 for safe)
            - agent_pos: Tensor [agent_x, agent_y, agent_radius, agent_radius]
        """
        self._timer.tic("select_random_frame")
        data_dir, frame_id, visible_bboxes, gt_bboxes = self.select_random_frame(
            data_dir=data_dir,
            frame_id=frame_id,
        )
        self._timer.toc("select_random_frame")

        # Limit entities for graph mode
        if len(visible_bboxes) > self.max_entities:
            visible_bboxes = random.sample(visible_bboxes, self.max_entities)

        if agent_pos is not None:
            agent_x, agent_y = agent_pos[0], agent_pos[1]
            assert agent_pos[2] == self.agent_radius
            assert agent_pos[3] == self.agent_radius
        else:
            self._timer.tic("generate_agent_position")
            agent_x, agent_y = self.generate_agent_position()
            self._timer.toc("generate_agent_position")

        self._timer.tic("compute_reward")
        reward = self._compute_reward(gt_bboxes, agent_x, agent_y)
        self._timer.toc("compute_reward")

        if self.use_precomputed_features and self.visible_data_loader:
            self._timer.tic("get_features")
            precomputed_features = self.visible_data_loader.get_features(
                data_dir, frame_id
            )
            self._timer.toc("get_features")
        else:
            precomputed_features = None

        data: Dict[str, Any] = {
            "graph": self._create_graph(
                visible_bboxes, agent_x, agent_y, precomputed_features
            ),
            "agent_pos": torch.tensor(
                [agent_x, agent_y, self.agent_radius, self.agent_radius],
                dtype=torch.float32,
            ),
        }

        if self.return_image:
            self._timer.tic("create_image")
            data["image"] = self._create_image(data_dir, frame_id, agent_x, agent_y)
            self._timer.toc("create_image")

        return data, reward

    def get_sample(
        self, data_dir: str, frame_id: int, agent_pos: Tuple[float, float, float, float]
    ) -> Tuple[Dict[str, Any], float]:
        return self._generate_sample(data_dir, frame_id, agent_pos)

    def _create_graph(
        self,
        scaled_bboxes: List[Tuple],
        agent_x: float,
        agent_y: float,
        precomputed_features: torch.Tensor | None = None,
    ) -> Data:
        """
        Create graph representation for GNN.

        Args:
            scaled_bboxes: List of scaled bounding boxes (x, y, w, h, track_id)
            agent_x: Agent x position (normalized)
            agent_y: Agent y position (normalized)
            precomputed_features: Precomputed features (torch tensor)

        Returns:
            PyTorch Geometric Data object
        """
        # Create node features
        self._timer.tic("create_relative_node_features")
        node_features = self._create_relative_node_features(
            scaled_bboxes,
            agent_x,
            agent_y,
            self.agent_radius,
            self.include_agent_node,
            precomputed_features=precomputed_features,
        )
        self._timer.toc("create_relative_node_features")

        # Create edges
        self._timer.tic("create_edges")
        edge_index = create_edges(node_features, self.connect_threshold)
        self._timer.toc("create_edges")

        # Create graph data
        self._timer.tic("create_graph_data")
        graph_data = self._create_graph_data(node_features, edge_index)
        self._timer.toc("create_graph_data")

        return graph_data

    def _create_relative_node_features(
        self,
        scaled_bboxes: List[Tuple],
        agent_x: float,
        agent_y: float,
        agent_radius: float,
        include_agent_node: bool,
        precomputed_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Create node features relative to agent position (vectorized).

        Node features format: [rel_x, rel_y, w, h, is_agent, is_obstacle, is_other] + precomputed_features (if available)

        Args:
            scaled_bboxes: Scaled bounding boxes (x, y, w, h, track_id, class_id)
            agent_x: Agent x position
            agent_y: Agent y position
            agent_radius: Agent radius
            include_agent_node: Whether to include agent node
            precomputed_features: Optional precomputed features tensor (already on device)

        Returns:
            Node feature tensor (num_nodes, feature_dim)

        Raises:
            ValueError: If precomputed_features length doesn't match scaled_bboxes
        """
        num_obstacles = len(scaled_bboxes)

        # Validate precomputed features if provided
        if (
            precomputed_features is not None
            and len(precomputed_features) != num_obstacles
        ):
            raise ValueError(
                f"Precomputed features length ({len(precomputed_features)}) "
                f"doesn't match bboxes length ({num_obstacles})"
            )

        # Determine dimensions upfront - now 7 base features instead of 5
        base_dim = 7
        feature_dim = base_dim
        if precomputed_features is not None:
            feature_dim += precomputed_features.shape[1]

        num_nodes = num_obstacles + (1 if include_agent_node else 0)

        # Pre-allocate full tensor
        self._timer.tic("allocate_tensor")
        all_features = torch.zeros(num_nodes, feature_dim, dtype=torch.float32)
        self._timer.toc("allocate_tensor")

        # Handle agent node first (if requested)
        start_idx = 0
        if include_agent_node:
            all_features[0, 2] = agent_radius  # w
            all_features[0, 3] = agent_radius  # h
            all_features[0, 4] = 1.0  # is_agent
            # is_obstacle=0, is_other=0 for agent (already initialized)
            start_idx = 1

        # Fill obstacle nodes (if any)
        if num_obstacles > 0:
            self._timer.tic("process_bboxes")
            # Vectorize bbox processing
            bboxes_array = np.array(scaled_bboxes, dtype=np.float32)  # (N, 6)
            x, y, w, h = (
                bboxes_array[:, 0],
                bboxes_array[:, 1],
                bboxes_array[:, 2],
                bboxes_array[:, 3],
            )
            class_ids = bboxes_array[:, 5]  # Extract class_id

            # Calculate centers and relative positions (vectorized)
            obs_center_x = x + w / 2
            obs_center_y = y + h / 2
            rel_x = obs_center_x - agent_x
            rel_y = obs_center_y - agent_y

            # Assign base features directly to preallocated tensor
            all_features[start_idx:, 0] = torch.from_numpy(rel_x)
            all_features[start_idx:, 1] = torch.from_numpy(rel_y)
            all_features[start_idx:, 2] = torch.from_numpy(w)
            all_features[start_idx:, 3] = torch.from_numpy(h)
            # is_agent=0 already set by zeros initialization

            # Set one-hot encoding for obstacle/other
            # is_obstacle (class_id == 0)
            all_features[start_idx:, 5] = torch.from_numpy((class_ids == 0).astype(np.float32))
            # is_other (class_id != 0, everything else)
            all_features[start_idx:, 6] = torch.from_numpy((class_ids != 0).astype(np.float32))

            self._timer.toc("process_bboxes")

            # Copy precomputed features if available
            if precomputed_features is not None:
                self._timer.tic("copy_precomputed")
                all_features[start_idx:, base_dim:] = precomputed_features
                self._timer.toc("copy_precomputed")

        return all_features

    def _create_graph_data(
        self, node_features: torch.Tensor, edge_index: torch.Tensor
    ) -> Data:
        """Create PyTorch Geometric Data object."""
        # Handle empty graph case
        if len(node_features) == 0:
            node_feature_dim = 7 + (12550 if self.use_precomputed_features else 0)
            return create_empty_graph(node_feature_dim=node_feature_dim)

        return Data(
            x=node_features,
            edge_index=edge_index,
            num_nodes=len(node_features),
        )

    def _get_cached_image(self, data_dir: str, frame_id: int) -> Optional[np.ndarray]:
        """
        Get image from cache if available (LRU).

        Args:
            data_dir: MOT data directory
            frame_id: Frame ID

        Returns:
            Cached image or None if not in cache
        """
        cache_key = (data_dir, frame_id)

        if cache_key in self._image_cache:
            # Move to end (most recently used)
            self._image_cache.move_to_end(cache_key)
            self._cache_hits += 1
            return self._image_cache[cache_key]

        self._cache_misses += 1
        return None

    def _cache_image(self, data_dir: str, frame_id: int, img: np.ndarray) -> None:
        """
        Add image to cache, evicting oldest if full (LRU).

        Args:
            data_dir: MOT data directory
            frame_id: Frame ID
            img: Image to cache
        """
        if self.image_cache_size == 0:
            return

        cache_key = (data_dir, frame_id)

        # Remove oldest if cache is full
        if len(self._image_cache) >= self.image_cache_size:
            self._image_cache.popitem(last=False)  # Remove oldest (first item)

        # Add new image
        self._image_cache[cache_key] = img

    def _create_image(
        self, data_dir: str, frame_id: int, agent_x: float, agent_y: float
    ) -> torch.Tensor:
        """
        Create image with agent drawn, using cache when available.

        Args:
            data_dir: MOT data directory
            frame_id: Frame ID
            agent_x: Agent x position (normalized)
            agent_y: Agent y position (normalized)

        Returns:
            Image tensor (H, W, 3) with agent drawn
        """
        # Try to get from cache first
        img = self._get_cached_image(data_dir, frame_id)

        if img is None:
            # Load and resize image
            self._timer.tic("load_and_resize_image")
            img = load_and_resize_image(data_dir, frame_id, self.image_size)
            self._timer.toc("load_and_resize_image")

            if img is None:
                raise RuntimeError(f"Failed to load image: {data_dir}/{frame_id}")

            # Cache the loaded image
            self._cache_image(data_dir, frame_id, img)

        # Draw agent on image (always need a copy since we modify it)
        self._timer.tic("draw_agent")
        agent_image = self._draw_agent(img.copy(), agent_x, agent_y)
        self._timer.toc("draw_agent")

        return torch.from_numpy(agent_image.astype(np.uint8))

    def _draw_agent(
        self, img: np.ndarray, agent_x: float, agent_y: float
    ) -> np.ndarray:
        """
        Draw agent as a red rectangle on the image.

        Args:
            img: Image to draw on
            agent_x: Agent x position (normalized 0-1)
            agent_y: Agent y position (normalized 0-1)

        Returns:
            Image with agent drawn
        """
        # Calculate rectangle bounds
        agent_left = max(0, agent_x - self.agent_radius)
        agent_right = min(1.0, agent_x + self.agent_radius)
        agent_top = max(0, agent_y - self.agent_radius)
        agent_bottom = min(1.0, agent_y + self.agent_radius)

        # Convert to pixel coordinates
        agent_left = int(agent_left * img.shape[1])
        agent_right = int(agent_right * img.shape[1])
        agent_top = int(agent_top * img.shape[0])
        agent_bottom = int(agent_bottom * img.shape[0])

        # Draw agent as red rectangle
        img[agent_top:agent_bottom, agent_left:agent_right] = [255, 0, 0]

        return img
