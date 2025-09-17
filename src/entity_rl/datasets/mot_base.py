import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from torch.utils.data import Dataset

from .mot_data import MOTDataLoader, check_rectangle_overlap


class MOTBaseDataset(Dataset, ABC):
    """
    Abstract base class for MOT-based datasets.

    This class provides common initialization, validation, and utilities
    that are shared across different MOT dataset implementations.
    """

    def __init__(
        self,
        mot_data_dirs: List[str],
        agent_radius: int = 15,
        num_samples_per_epoch: int = 1000,
        image_size: Tuple[int, int] = (100, 100),
        use_gt: bool = True,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize base MOT dataset.

        Args:
            mot_data_dirs: List of MOT data directories
            agent_radius: Radius of synthetic agent for collision detection
            num_samples_per_epoch: Number of samples to generate per epoch
            image_size: Target image size (width, height)
            use_gt: Whether to use ground truth (gt.txt) or detections (det.txt)
        """
        self.mot_data_dirs = mot_data_dirs
        self.agent_radius = agent_radius
        self.image_size = image_size
        self.use_gt = use_gt

        # Load MOT data
        self.data_loader = MOTDataLoader(mot_data_dirs, use_gt)
        self.mot_data = self.data_loader.load_mot_data(max_rows=max_samples)

        if not self.mot_data:
            raise ValueError("No valid MOT data found in the provided directories")

        total_frames = sum(len(frames) for frames in self.mot_data.values())
        entities = []
        for frames in self.mot_data.values():
            for frame in frames.values():
                entities.append(len(frame))

        self.num_samples_per_epoch = min(total_frames, num_samples_per_epoch)

        print(f"Loaded MOT data from {len(self.mot_data_dirs)} directories")
        print(f"Total frames available: {total_frames}")

        print(
            f"Max detections: {max(entities)}, Min detections: {min(entities)}, Avg detections: {sum(entities) / len(entities)}"
        )
        # Validate parameters
        self._validate_parameters()

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

    @abstractmethod
    def _generate_sample(self) -> Tuple[Any, int]:
        """
        Generate a single sample.

        Returns:
            Tuple of (sample_data, reward)
        """
        pass

    def select_random_frame(self) -> Tuple[str, int, List[Tuple]]:
        """
        Select a random frame from the loaded data.

        Returns:
            Tuple of (data_dir, frame_id, bboxes)
        """
        # Randomly select data directory and frame
        data_dir = random.choice(list(self.mot_data.keys()))
        frame_ids = list(self.mot_data[data_dir].keys())
        frame_id = random.choice(frame_ids)
        bboxes = self.mot_data[data_dir][frame_id]

        return data_dir, frame_id, bboxes

    def generate_agent_position(self) -> Tuple[float, float]:
        """
        Generate a random agent position within the image bounds.

        Returns:
            Tuple of (agent_x, agent_y)
        """
        agent_x = random.random()
        agent_y = random.random()

        return agent_x, agent_y

    def _compute_reward(self, scaled_bboxes, agent_x, agent_y) -> int:
        """Compute reward based on agent collision with bounding boxes."""
        if len(scaled_bboxes) > 0:
            has_collision = check_rectangle_overlap(
                agent_x, agent_y, self.agent_radius, scaled_bboxes
            )
            return -1 if has_collision else 1
        else:
            # No entities, always safe
            return 1

    def __len__(self) -> int:
        """Return the number of samples per epoch."""
        return self.num_samples_per_epoch

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        """
        Get a sample by index.

        Args:
            idx: Sample index (ignored, samples are generated randomly)

        Returns:
            Generated sample
        """
        # start_time = time.time()
        sample = self._generate_sample()
        # print(f"Generated sample in {time.time() - start_time:.2f} seconds")
        return sample

    @property
    def total_frames(self) -> int:
        """Get total number of frames across all datasets."""
        return sum(len(frames) for frames in self.mot_data.values())

    @property
    def dataset_info(self) -> Dict[str, int]:
        """Get information about loaded datasets."""
        info = {}
        for data_dir, frames in self.mot_data.items():
            info[data_dir] = len(frames)
        return info
