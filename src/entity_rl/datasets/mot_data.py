"""
MOT data loading utilities for ENROS datasets.

This module provides common functionality for loading and processing MOT annotation files.
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np


class MOTDataLoader:
    """Base class for loading MOT (Multiple Object Tracking) data."""

    def __init__(self, mot_data_dirs: List[str], use_gt: bool = True):
        """
        Initialize MOT data loader.

        Args:
            mot_data_dirs: List of MOT data directories
            use_gt: Whether to use ground truth (gt.txt) or detections (det.txt)
        """
        self.mot_data_dirs = mot_data_dirs
        self.use_gt = use_gt

    def load_mot_data(self, include_track_id: bool = False) -> Dict[str, Dict[int, List[Tuple]]]:
        """
        Load MOT annotation data from all specified directories.

        Args:
            include_track_id: Whether to include track IDs in the output

        Returns:
            Dictionary mapping data_dir -> {frame_id: [(x, y, w, h, track_id?), ...]}
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
            frame_data = self._parse_mot_annotations(ann_file, include_track_id)

            # Only include frames that have corresponding image files
            valid_frame_data = self._validate_frames(frame_data, img_dir)

            if valid_frame_data:
                mot_data[str(data_dir)] = valid_frame_data
                print(f"Loaded {len(valid_frame_data)} frames from {data_dir}")

        return mot_data

    def _parse_mot_annotations(
        self, ann_file: Path, include_track_id: bool = False
    ) -> Dict[int, List[Tuple]]:
        """
        Parse MOT annotation file.

        Args:
            ann_file: Path to annotation file
            include_track_id: Whether to include track IDs

        Returns:
            Dictionary mapping frame_id to list of bounding boxes
        """
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

                    # Filter out invalid bounding boxes
                    if w <= 0 or h <= 0:
                        continue

                    if frame_id not in frame_data:
                        frame_data[frame_id] = []

                    if include_track_id:
                        frame_data[frame_id].append((x, y, w, h, track_id))
                    else:
                        frame_data[frame_id].append((x, y, w, h))

        except Exception as e:
            print(f"Error reading {ann_file}: {e}")

        return frame_data

    def _validate_frames(
        self, frame_data: Dict[int, List[Tuple]], img_dir: Path
    ) -> Dict[int, List[Tuple]]:
        """
        Validate that frames have corresponding image files.

        Args:
            frame_data: Raw frame data from annotations
            img_dir: Directory containing images

        Returns:
            Filtered frame data with only valid frames
        """
        valid_frame_data = {}

        for frame_id, bboxes in frame_data.items():
            img_path = img_dir / f"{frame_id:06d}.jpg"
            if img_path.exists() and len(bboxes) > 0:
                valid_frame_data[frame_id] = bboxes

        return valid_frame_data

    def get_image_dimensions(self, data_dir: str, frame_id: int) -> Tuple[int, int]:
        """
        Get original image dimensions for a specific frame.

        Args:
            data_dir: MOT data directory
            frame_id: Frame ID

        Returns:
            Tuple of (width, height)
        """
        img_path = Path(data_dir) / "img1" / f"{frame_id:06d}.jpg"
        img = cv2.imread(str(img_path))

        if img is None:
            return 640, 480  # Fallback dimensions

        h, w = img.shape[:2]
        return w, h


def scale_bboxes(
    bboxes: List[Tuple],
    original_size: Tuple[int, int],
    target_size: Tuple[int, int],
    include_track_id: bool = False,
) -> List[Tuple]:
    """
    Scale bounding boxes to match target image size.

    Args:
        bboxes: List of bounding boxes
        original_size: Original image size (width, height)
        target_size: Target image size (width, height)
        include_track_id: Whether bboxes include track IDs

    Returns:
        List of scaled bounding boxes
    """
    if not bboxes:
        return []

    orig_w, orig_h = original_size
    target_w, target_h = target_size

    scale_x = target_w / orig_w
    scale_y = target_h / orig_h

    scaled_bboxes = []
    for bbox in bboxes:
        if include_track_id:
            x, y, w, h, track_id = bbox
            scaled_x = int(x * scale_x)
            scaled_y = int(y * scale_y)
            scaled_w = int(w * scale_x)
            scaled_h = int(h * scale_y)
            scaled_bboxes.append((scaled_x, scaled_y, scaled_w, scaled_h, track_id))
        else:
            x, y, w, h = bbox
            scaled_x = int(x * scale_x)
            scaled_y = int(y * scale_y)
            scaled_w = int(w * scale_x)
            scaled_h = int(h * scale_y)
            scaled_bboxes.append((scaled_x, scaled_y, scaled_w, scaled_h))

    return scaled_bboxes


def check_rectangle_overlap(
    agent_x: int,
    agent_y: int,
    agent_radius: int,
    bboxes: List[Tuple],
    include_track_id: bool = False,
) -> bool:
    """
    Check if a rectangular agent overlaps with any bounding box.

    Args:
        agent_x, agent_y: Center of the agent rectangle
        agent_radius: Half-width/height of the agent
        bboxes: List of bounding boxes
        include_track_id: Whether bboxes include track IDs

    Returns:
        True if agent overlaps with any bounding box, False otherwise
    """
    # Agent bounding box (centered at agent_x, agent_y)
    agent_left = agent_x - agent_radius
    agent_top = agent_y - agent_radius
    agent_right = agent_x + agent_radius
    agent_bottom = agent_y + agent_radius

    for bbox in bboxes:
        if include_track_id:
            x, y, w, h, _ = bbox
        else:
            x, y, w, h = bbox

        # Check if rectangles overlap
        if (
            agent_left < x + w
            and agent_right > x
            and agent_top < y + h
            and agent_bottom > y
        ):
            return True

    return False


def load_and_resize_image(
    data_dir: str, frame_id: int, target_size: Tuple[int, int]
) -> Optional[np.ndarray]:
    """
    Load and resize an image from MOT data.

    Args:
        data_dir: MOT data directory
        frame_id: Frame ID
        target_size: Target image size (width, height)

    Returns:
        Resized image as RGB numpy array, or None if loading fails
    """
    img_path = Path(data_dir) / "img1" / f"{frame_id:06d}.jpg"
    img = cv2.imread(str(img_path))

    if img is None:
        return None

    # Resize image
    img = cv2.resize(img, target_size)

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img