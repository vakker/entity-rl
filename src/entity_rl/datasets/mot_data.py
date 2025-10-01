import configparser
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


class MOTDataLoader:
    """Base class for loading MOT (Multiple Object Tracking) data."""

    def __init__(
        self,
        mot_data_dirs: List[str],
        use_gt: bool = True,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize MOT data loader.

        Args:
            mot_data_dirs: List of MOT data directories
            use_gt: Whether to use ground truth (gt.txt) or detections (det.txt)
        """
        self.mot_data_dirs = mot_data_dirs
        print(f"Loaded MOT data from {len(self.mot_data_dirs)} directories")
        self.use_gt = use_gt

        # Cache for image dimensions (per directory)
        self._dimension_cache: Dict[str, Tuple[int, int]] = {}
        self._load_all_metadata()

        self._mot_data = self.load_mot_data(max_rows=max_samples)

    def mot_data(self):
        return self._mot_data

    def get_total_frames(self):
        all_frames = []
        for frames in self._mot_data.values():
            all_frames.append(len(list(frames.keys())))

        return sum(all_frames)

    def get_entities(self):
        entities = []
        for frames in self._mot_data.values():
            for frame in frames.values():
                entities.append(len(frame))

        return entities

    def _load_all_metadata(self) -> None:
        """Load metadata (image dimensions) for all MOT directories."""
        for data_dir in self.mot_data_dirs:
            dimensions = self._load_sequence_metadata(data_dir)
            if dimensions:
                self._dimension_cache[str(data_dir)] = dimensions

    def _load_sequence_metadata(self, data_dir: str) -> Optional[Tuple[int, int]]:
        """
        Load sequence metadata from seqinfo.ini file.

        Args:
            data_dir: MOT data directory

        Returns:
            Tuple of (width, height) or None if metadata unavailable
        """
        data_path = Path(data_dir)
        seqinfo_path = data_path / "seqinfo.ini"

        if seqinfo_path.exists():
            try:
                config = configparser.ConfigParser()
                config.read(seqinfo_path)

                if "Sequence" in config:
                    width = config.getint("Sequence", "imWidth")
                    height = config.getint("Sequence", "imHeight")
                    return (width, height)
            except Exception as e:
                print(f"Warning: Could not parse seqinfo.ini in {data_dir}: {e}")

        # Fallback: try to get dimensions from first image
        img_dir = data_path / "img1"
        if img_dir.exists():
            # Find first image file
            for img_file in sorted(img_dir.glob("*.jpg")):
                dimensions = self._get_image_dimensions_from_header(img_file)
                if dimensions:
                    return dimensions
                break  # Only try first image

        return None

    def _get_image_dimensions_from_header(
        self, img_path: Path
    ) -> Optional[Tuple[int, int]]:
        """
        Get image dimensions by reading only the header (fast).

        Args:
            img_path: Path to image file

        Returns:
            Tuple of (width, height) or None if failed
        """
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                return (width, height)
        except Exception as e:
            print(f"Warning: Could not read image header from {img_path}: {e}")
            return None

    def load_mot_data(
        self, max_rows: Optional[int] = None
    ) -> Dict[str, Dict[int, List[Tuple[int, int, int, int, int]]]]:
        """
        Load MOT annotation data from all specified directories.

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
                ann_file = data_path / "prop.csv"

            img_dir = data_path / "img1"

            if not ann_file.exists():
                print(f"Warning: Annotation file not found: {ann_file}")
                continue

            if not img_dir.exists():
                print(f"Warning: Image directory not found: {img_dir}")
                continue

            # Parse MOT annotations
            frame_data = self._parse_mot_annotations(ann_file, max_rows)

            # Only include frames that have corresponding image files
            valid_frame_data = self._validate_frames(frame_data, img_dir)

            if valid_frame_data:
                mot_data[str(data_dir)] = valid_frame_data
                print(f"Loaded {len(valid_frame_data)} frames from {data_dir}")

        return mot_data

    def _parse_mot_annotations(
        self, ann_file: Path, max_rows: None
    ) -> Dict[int, List[Tuple[float, float, float, float, int]]]:
        """
        Parse MOT annotation file.

        Args:
            ann_file: Path to annotation file

        Returns:
            Dictionary mapping frame_id to list of (x, y, w, h, track_id) tuples
        """
        frame_data = {}

        try:
            with open(ann_file, "r") as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    # Skip header
                    if i == 0 and ann_file.suffix == ".csv":
                        continue

                    if len(row) < 6:
                        continue

                    frame_id = int(row[0])
                    if max_rows is not None and frame_id > max_rows:
                        continue

                    track_id = int(row[1]) if len(row) > 1 else -1
                    x, y, w, h = map(float, row[2:6])

                    # Filter out invalid bounding boxes
                    if w <= 0 or h <= 0:
                        continue

                    if frame_id not in frame_data:
                        frame_data[frame_id] = []

                    frame_data[frame_id].append((x, y, w, h, track_id))

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

        Uses cached metadata from seqinfo.ini or first image header for speed.
        Only falls back to loading full image if cache miss occurs.

        Args:
            data_dir: MOT data directory
            frame_id: Frame ID (unused if metadata cached)

        Returns:
            Tuple of (width, height)
        """
        # Try to use cached dimensions first
        data_dir_str = str(data_dir)
        if data_dir_str in self._dimension_cache:
            return self._dimension_cache[data_dir_str]

        # Cache miss - try to load metadata now
        dimensions = self._load_sequence_metadata(data_dir_str)
        if dimensions:
            self._dimension_cache[data_dir_str] = dimensions
            return dimensions

        # Last resort: load full image (slow, old behavior)
        img_path = Path(data_dir) / "img1" / f"{frame_id:06d}.jpg"
        img = cv2.imread(str(img_path))

        if img is None:
            raise Exception(f"Failed to load image {img_path}")

        h, w = img.shape[:2]
        # Cache for future use
        self._dimension_cache[data_dir_str] = (w, h)
        return w, h


def scale_bboxes(
    bboxes: List[Tuple[int, int, int, int, int]],
    original_size: Tuple[int, int],
) -> List[Tuple[float, float, float, float, int]]:
    """
    Scale bounding boxes to match target image size.

    Args:
        bboxes: List of (x, y, w, h, track_id) tuples
        original_size: Original image size (width, height)

    Returns:
        List of scaled (x, y, w, h, track_id) tuples
    """
    if not bboxes:
        return []

    orig_w, orig_h = original_size
    target_w, target_h = 1.0, 1.0

    scale_x = target_w / orig_w
    scale_y = target_h / orig_h

    scaled_bboxes = []
    for x, y, w, h, track_id in bboxes:
        scaled_x = x * scale_x
        scaled_y = y * scale_y
        scaled_w = w * scale_x
        scaled_h = h * scale_y
        scaled_bboxes.append((scaled_x, scaled_y, scaled_w, scaled_h, track_id))

    return scaled_bboxes


def check_rectangle_overlap(
    agent_x: float,
    agent_y: float,
    agent_radius: float,
    bboxes: List[Tuple[float, float, float, float, int]],
) -> bool:
    """
    Check if a rectangular agent overlaps with any bounding box.

    Args:
        agent_x, agent_y: Center of the agent rectangle
        agent_radius: Half-width/height of the agent
        bboxes: List of (x, y, w, h, track_id) tuples

    Returns:
        True if agent overlaps with any bounding box, False otherwise
    """
    # Agent bounding box (centered at agent_x, agent_y)
    agent_left = agent_x - agent_radius
    agent_top = agent_y - agent_radius
    agent_right = agent_x + agent_radius
    agent_bottom = agent_y + agent_radius

    for x, y, w, h, _ in bboxes:
        # NOTE: debuggin
        # if agent_x < x:
        #     return True
        #
        # else:
        #     return False

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
