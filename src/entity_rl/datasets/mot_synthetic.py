"""
MOT-based synthetic dataset for ENROS training.

This dataset creates synthetic agent-environment interactions using MOT data,
placing agent blobs on real tracking scenes.
"""

import numpy as np
import torch
from typing import Tuple

from .mot_base import MOTBaseDataset
from .mot_data import check_rectangle_overlap, load_and_resize_image, scale_bboxes


class MOTSyntheticDataset(MOTBaseDataset):
    """
    Dataset that generates synthetic agent-environment data from MOT tracking data.

    Each sample consists of:
    - An image with a rectangular agent blob overlaid at a random position
    - A reward signal based on whether the agent overlaps with tracked objects
    """

    def _load_data(self):
        """Load MOT data for synthetic dataset."""
        return self.data_loader.load_mot_data()

    def _generate_sample(self) -> Tuple[torch.Tensor, int]:
        """
        Generate a single synthetic sample.

        Returns:
            Tuple of (image_tensor, reward) where:
            - image_tensor: torch.Tensor of shape (H, W, 3), dtype uint8
            - reward: int (-1 for collision, +1 for safe)
        """
        # Select random frame and bboxes
        data_dir, frame_id, original_bboxes = self.select_random_frame()

        # Load and resize image
        img = load_and_resize_image(data_dir, frame_id, self.image_size)

        if img is None:
            raise RuntimeError("Failed to load or resize image")
        else:
            # Get original dimensions and scale bboxes
            orig_w, orig_h = self.data_loader.get_image_dimensions(data_dir, frame_id)
            scaled_bboxes = scale_bboxes(original_bboxes, (orig_w, orig_h), self.image_size)

        # Generate random agent position
        agent_x, agent_y = self.generate_agent_position()

        # Draw agent blob on the image
        agent_image = self._draw_agent(img.copy(), agent_x, agent_y)

        # Determine reward based on overlap
        reward = self._compute_reward(scaled_bboxes, agent_x, agent_y)

        return torch.from_numpy(agent_image.astype(np.uint8)), reward

    def _draw_agent(self, img: np.ndarray, agent_x: int, agent_y: int) -> np.ndarray:
        """
        Draw agent as a red rectangle on the image.

        Args:
            img: Image to draw on
            agent_x, agent_y: Agent center position

        Returns:
            Image with agent drawn
        """
        # Calculate rectangle bounds
        agent_left = max(0, agent_x - self.agent_radius)
        agent_right = min(self.image_size[0], agent_x + self.agent_radius)
        agent_top = max(0, agent_y - self.agent_radius)
        agent_bottom = min(self.image_size[1], agent_y + self.agent_radius)

        # Draw agent as red rectangle
        img[agent_top:agent_bottom, agent_left:agent_right] = [255, 0, 0]

        return img
