"""
MOT-based synthetic dataset generator for ENROS training.

This module creates synthetic agent-environment interactions using MOT (Multiple Object Tracking) data.
For each sample:
1. Pick a random MOT image and its corresponding bounding boxes
2. Place a circular "agent" blob at a random position
3. Assign reward based on overlap with bounding boxes:
   - Reward = +1 if agent doesn't overlap with any bounding box (safe position)
   - Reward = -1 if agent overlaps with any bounding box (collision)
"""

import argparse
import csv
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset


class MOTSyntheticDataset(Dataset):
    """
    Dataset that generates synthetic agent-environment data from MOT tracking data.

    Each sample consists of:
    - An image with a circular agent blob overlaid at a random position
    - A reward signal based on whether the agent overlaps with tracked objects
    """

    def __init__(
        self,
        mot_data_dirs: List[str],
        agent_radius: int = 15,
        num_samples_per_epoch: int = 1000,
        image_size: Tuple[int, int] = (100, 100),
        use_gt: bool = True,
    ):
        """
        Initialize the MOT synthetic dataset.

        Args:
            mot_data_dirs: List of MOT data directories (e.g., ["data/MOT17/train/MOT17-02-FRCNN"])
            agent_radius: Radius of the circular agent blob in pixels
            num_samples_per_epoch: Number of synthetic samples to generate per epoch
            image_size: Target size to resize images to (height, width)
            use_gt: Whether to use ground truth annotations (gt.txt) or detections (det.txt)
        """
        self.mot_data_dirs = mot_data_dirs
        self.agent_radius = agent_radius
        self.num_samples_per_epoch = num_samples_per_epoch
        self.image_size = image_size
        self.use_gt = use_gt

        # Load MOT data
        self.mot_data = self._load_mot_data()

        if not self.mot_data:
            raise ValueError("No valid MOT data found in the provided directories")

        print(f"Loaded MOT data from {len(self.mot_data_dirs)} directories")
        print(
            f"Total frames available: {sum(len(frames) for frames in self.mot_data.values())}"
        )

    def _load_mot_data(self) -> Dict[str, Dict[int, List[Tuple[int, int, int, int]]]]:
        """
        Load MOT annotation data from all specified directories.

        Returns:
            Dictionary mapping data_dir -> {frame_id: [(x, y, w, h), ...]}
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

            if not ann_file.exists():
                print(f"Warning: Annotation file not found: {ann_file}")
                continue

            if not img_dir.exists():
                print(f"Warning: Image directory not found: {img_dir}")
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
                        x, y, w, h = map(int, row[2:6])

                        # Filter out invalid bounding boxes
                        if w <= 0 or h <= 0:
                            continue

                        if frame_id not in frame_data:
                            frame_data[frame_id] = []
                        frame_data[frame_id].append((x, y, w, h))

            except Exception as e:
                print(f"Error reading {ann_file}: {e}")
                continue

            # Only include frames that have corresponding image files
            valid_frame_data = {}
            for frame_id, bboxes in frame_data.items():
                img_path = img_dir / f"{frame_id:06d}.jpg"
                if img_path.exists():
                    valid_frame_data[frame_id] = bboxes

            if valid_frame_data:
                mot_data[str(data_dir)] = valid_frame_data
                print(f"Loaded {len(valid_frame_data)} frames from {data_dir}")

        return mot_data

    def _check_overlap(
        self, agent_x: int, agent_y: int, bboxes: List[Tuple[int, int, int, int]]
    ) -> bool:
        """
        Check if the rectangular agent overlaps with any bounding box.

        Args:
            agent_x, agent_y: Center of the agent rectangle
            bboxes: List of bounding boxes as (x, y, w, h)

        Returns:
            True if agent overlaps with any bounding box, False otherwise
        """
        # Agent bounding box (centered at agent_x, agent_y)
        agent_size = self.agent_radius * 2  # Use radius as half-width/height
        agent_left = agent_x - self.agent_radius
        agent_top = agent_y - self.agent_radius
        agent_right = agent_x + self.agent_radius
        agent_bottom = agent_y + self.agent_radius

        for x, y, w, h in bboxes:
            # Check if rectangles overlap
            if (agent_left < x + w and
                agent_right > x and
                agent_top < y + h and
                agent_bottom > y):
                return True

        return False

    def _generate_sample(self) -> Tuple[np.ndarray, int]:
        """
        Generate a single synthetic sample.

        Returns:
            Tuple of (image, reward) where:
            - image: np.ndarray of shape (height, width, 3)
            - reward: int (-1 for collision, +1 for safe)
        """
        # Randomly select a data directory and frame
        data_dir = random.choice(list(self.mot_data.keys()))
        frame_ids = list(self.mot_data[data_dir].keys())
        frame_id = random.choice(frame_ids)

        # Load image
        img_path = Path(data_dir) / "img1" / f"{frame_id:06d}.jpg"
        img = cv2.imread(str(img_path))

        if img is None:
            # Fallback to a black image if loading fails
            img = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
            scaled_bboxes = []
        else:
            # Get original dimensions for scaling bboxes
            orig_h, orig_w = img.shape[:2]

            # Resize image
            img = cv2.resize(img, self.image_size)

            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Scale bounding boxes to match resized image
            original_bboxes = self.mot_data[data_dir][frame_id]
            scale_x = self.image_size[0] / orig_w
            scale_y = self.image_size[1] / orig_h

            scaled_bboxes = []
            for x, y, w, h in original_bboxes:
                scaled_x = int(x * scale_x)
                scaled_y = int(y * scale_y)
                scaled_w = int(w * scale_x)
                scaled_h = int(h * scale_y)
                scaled_bboxes.append((scaled_x, scaled_y, scaled_w, scaled_h))

        # Generate random agent position
        agent_x = random.randint(
            self.agent_radius, self.image_size[0] - self.agent_radius
        )
        agent_y = random.randint(
            self.agent_radius, self.image_size[1] - self.agent_radius
        )

        # Draw agent blob on the image
        agent_image = img.copy()

        # Draw agent as a red rectangle
        agent_left = max(0, agent_x - self.agent_radius)
        agent_right = min(self.image_size[0], agent_x + self.agent_radius)
        agent_top = max(0, agent_y - self.agent_radius)
        agent_bottom = min(self.image_size[1], agent_y + self.agent_radius)

        agent_image[agent_top:agent_bottom, agent_left:agent_right] = [255, 0, 0]  # Red color

        # Determine reward based on overlap
        has_overlap = self._check_overlap(agent_x, agent_y, scaled_bboxes)
        reward = -1 if has_overlap else 1

        return agent_image.astype(np.uint8), reward

    def __len__(self) -> int:
        return self.num_samples_per_epoch

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a synthetic sample.

        Args:
            idx: Sample index (ignored, samples are generated randomly)

        Returns:
            Tuple of (image_tensor, reward) where:
            - image_tensor: torch.Tensor of shape (H, W, 3), dtype uint8
            - reward: int reward signal (-1 or +1)
        """
        image, reward = self._generate_sample()
        return torch.from_numpy(image), reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MOT Synthetic Dataset")
    parser.add_argument(
        "--mot-dirs", nargs="+", required=True, help="List of MOT data directories"
    )
    parser.add_argument(
        "--num-samples", type=int, default=5, help="Number of test samples to generate"
    )
    parser.add_argument(
        "--agent-radius", type=int, default=15, help="Radius of agent blob in pixels"
    )
    parser.add_argument(
        "--image-size",
        nargs=2,
        type=int,
        default=[100, 100],
        help="Target image size as width height",
    )
    parser.add_argument(
        "--use-detections",
        action="store_true",
        help="Use detection files (det.txt) instead of ground truth (gt.txt)",
    )

    args = parser.parse_args()

    print(f"Using MOT directories: {args.mot_dirs}")

    # Test the dataset
    dataset = MOTSyntheticDataset(
        mot_data_dirs=args.mot_dirs,
        agent_radius=args.agent_radius,
        num_samples_per_epoch=args.num_samples,
        image_size=tuple(args.image_size),
        use_gt=not args.use_detections,
    )

    print(f"Dataset length: {len(dataset)}")

    # Generate test samples
    for i in range(args.num_samples):
        img, reward = dataset[i]
        print(f"Sample {i}: shape={img.shape}, reward={reward}")

        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title(f"Sample {i}, Reward: {reward}")
        plt.axis("off")
        plt.savefig(f"sample_{i}_reward_{reward}.png")
        plt.close()

    print("Test completed. Sample images saved.")

