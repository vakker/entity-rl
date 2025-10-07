"""
MOT (Multiple Object Tracking) Data Visualizer

This script visualizes MOT tracking data by drawing bounding boxes and track IDs
on video frames. It supports both ground truth and detection result formats.

Usage:
    python mot_visualizer.py --mot_file data/MOT16/train/MOT16-02/gt/gt.txt \
                           --image_dir data/MOT16/train/MOT16-02/img1 \
                           --output_dir output_videos \
                           --mode gt \
                           --fps 25 \
                           --show_occluder

MOT file format (CSV):
    frame_id, track_id, x, y, width, height, confidence, class_id, visibility

Author: Generated for entity-rl project
"""

import argparse
import glob
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np


class MOTVisualizer:
    """
    Visualizer for Multiple Object Tracking (MOT) data.

    Handles loading MOT tracking data and generating visualizations with
    bounding boxes and track IDs overlaid on video frames.
    """

    # Color palette for different track IDs
    COLORS = [
        (1.0, 0.0, 0.0),  # Red
        (0.0, 1.0, 0.0),  # Green
        (0.0, 0.0, 1.0),  # Blue
        (1.0, 1.0, 0.0),  # Yellow
        (1.0, 0.0, 1.0),  # Magenta
        (0.0, 1.0, 1.0),  # Cyan
        (0.5, 0.5, 0.5),  # Gray
        (1.0, 0.5, 0.0),  # Orange
        (0.5, 0.0, 0.5),  # Purple
        (0.0, 0.5, 0.5),  # Teal
    ]

    def __init__(
        self,
        mot_file: Union[str, Path],
        image_dir: Union[str, Path],
        output_dir: Union[str, Path],
        mode: str = "gt",
        fps: int = 25,
        show_occluder: bool = True,
        im_scale: float = 1.0,
    ):
        """
        Initialize the MOT visualizer.

        Args:
            mot_file: Path to MOT tracking data file (CSV format)
            image_dir: Directory containing video frames
            output_dir: Directory to save output videos
            mode: Visualization mode ('gt' for ground truth, 'det' for detections)
            fps: Frames per second for output video
            show_occluder: Whether to highlight occluded objects
            im_scale: Image scaling factor
        """
        self.mot_file = Path(mot_file)
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.mode = mode
        self.fps = fps
        self.show_occluder = show_occluder
        self.im_scale = im_scale

        self.colors = self.COLORS
        self.res_file: Optional[np.ndarray] = None

        # Validate inputs
        self._validate_inputs()

        # Load MOT data
        self.res_file = self._load_mot_data()

    def _validate_inputs(self) -> None:
        """Validate input parameters and file paths."""
        if not self.mot_file.exists():
            raise FileNotFoundError(f"MOT file not found: {self.mot_file}")

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        if self.mode not in ["gt", "det"]:
            raise ValueError(f"Invalid mode: {self.mode}. Must be 'gt' or 'det'")

        if self.fps <= 0:
            raise ValueError(f"FPS must be positive: {self.fps}")

    def _load_mot_data(self) -> Optional[np.ndarray]:
        """
        Load MOT tracking data from file.

        Returns:
            Numpy array with tracking data, or None if loading fails
        """
        try:
            # Try comma delimiter first
            data = np.genfromtxt(self.mot_file, delimiter=",")

            # If that fails, try space delimiter
            if data.ndim == 1:
                data = np.genfromtxt(self.mot_file, delimiter=" ")

            # If still 1D, file might be malformed
            if data.ndim == 1:
                print(f"Warning: Cannot parse {self.mot_file}, skipping")
                return None

            # Remove rows with NaN values
            nan_mask = np.sum(np.isnan(data), axis=1) == 0
            data = data[nan_mask]

            print(f"Loaded {len(data)} tracking entries from {self.mot_file}")
            return data

        except Exception as e:
            print(f"Error loading MOT data: {e}")
            return None

    def _get_frame_paths(self) -> List[Path]:
        """Get sorted list of frame image paths."""
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        frame_paths = []

        for ext in image_extensions:
            frame_paths.extend(self.image_dir.glob(ext))

        return sorted(frame_paths)

    def _draw_bounding_boxes(self, image: np.ndarray, frame_id: int) -> np.ndarray:
        """
        Draw bounding boxes and track IDs on the given frame.

        Args:
            image: Input image frame
            frame_id: Frame number to visualize

        Returns:
            Image with bounding boxes drawn
        """
        if self.res_file is None:
            return image

        # Find detections for this frame
        frame_mask = self.res_file[:, 0] == frame_id
        frame_detections = self.res_file[frame_mask]

        # Calculate max confidence for normalization
        max_conf = 1.0
        if self.mode == "det" and len(frame_detections) > 0:
            max_conf = np.max(frame_detections[:, 6])

        for detection in frame_detections:
            track_id = int(detection[1])
            x = int((detection[2] - 1) * self.im_scale)
            y = int((detection[3] - 1) * self.im_scale)
            width = int(detection[4] * self.im_scale)
            height = int(detection[5] * self.im_scale)

            # Normalize confidence to [0, 5] range
            confidence = detection[6] if len(detection) > 6 else 1.0
            conf_level = int((confidence / max_conf) * 5)

            # Get class label
            class_id = int(detection[7]) if len(detection) > 7 else 0

            # Select color based on obstacle status (class_id)
            # Red for obstacles (class_id == 0), Green for others (class_id != 0)
            if class_id == 0:
                color = (0, 0, 255)  # Red for obstacles
            else:
                color = (0, 255, 0)  # Green for non-obstacles

            pt1 = (x, y)
            pt2 = (x + width, y + height)

            # Handle occluders (special case for ground truth)
            if self.mode == "gt" and self.show_occluder and class_id in [9, 10, 11, 13]:
                # Draw semi-transparent overlay for occluded objects
                overlay = image.copy()
                alpha = 0.7
                occluder_color = (int(0.7 * 255), int(0.7 * 255), int(0.7 * 255))
                cv2.rectangle(overlay, pt1, pt2, occluder_color, -1)
                image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
            else:
                # Draw bounding box
                cv2.rectangle(image, pt1, pt2, color, 2)

                # Draw label (track ID + class for GT, class + confidence for detections)
                if self.mode == "gt":
                    # Ground truth: show track_id (class_id)
                    label_text = f"{track_id} (C{class_id})"
                else:
                    # Detections: show class_id (confidence)
                    label_text = f"C{class_id} ({confidence:.2f})"

                cv2.putText(
                    image,
                    label_text,
                    pt1,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

        return image

    def generate_video(
        self,
        display_time: bool = True,
        display_name: str = "mot_visualization",
        show_occluder: Optional[bool] = None,
        max_frames: Optional[int] = None,
    ) -> str:
        """
        Generate video with MOT visualizations.

        Args:
            display_time: Whether to display frame timestamps
            display_name: Name for the output video file
            show_occluder: Override occluder display setting
            max_frames: Maximum number of frames to process (None = all frames)

        Returns:
            Path to the generated video file
        """
        if show_occluder is not None:
            self.show_occluder = show_occluder

        # Get frame paths
        frame_paths = self._get_frame_paths()
        if not frame_paths:
            raise ValueError(f"No image files found in {self.image_dir}")

        # Limit frames if max_frames is specified
        if max_frames is not None and max_frames > 0:
            frame_paths = frame_paths[:max_frames]

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Read first frame to get video dimensions
        first_frame = cv2.imread(str(frame_paths[0]))
        if first_frame is None:
            raise ValueError(f"Could not read first frame: {frame_paths[0]}")

        height, width = first_frame.shape[:2]

        # Create video writer
        output_path = self.output_dir / f"{display_name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(output_path), fourcc, self.fps, (width, height)
        )

        print(f"Generating video: {output_path}")
        print(f"Processing {len(frame_paths)} frames...")

        # Process each frame
        for i, frame_path in enumerate(frame_paths):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                print(f"Warning: Could not read frame {frame_path}")
                continue

            frame_id = i + 1  # MOT frames are 1-indexed

            # Draw bounding boxes
            frame = self._draw_bounding_boxes(frame, frame_id)

            # Add timestamp if requested
            if display_time:
                timestamp = f"Frame: {frame_id}"
                cv2.putText(
                    frame,
                    timestamp,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )

            # Write frame to video
            video_writer.write(frame)

            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(frame_paths)} frames")

        video_writer.release()
        print(f"Video generation complete: {output_path}")

        return str(output_path)


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Visualize MOT (Multiple Object Tracking) data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --mot_file data/MOT16/train/MOT16-02/gt/gt.txt \\
           --image_dir data/MOT16/train/MOT16-02/img1 \\
           --output_dir output_videos \\
           --mode gt

  %(prog)s --mot_file results/tracking_output.txt \\
           --image_dir frames/ \\
           --output_dir videos/ \\
           --mode det \\
           --fps 30 \\
           --no-occluder
        """,
    )

    parser.add_argument(
        "--mot_file",
        type=str,
        required=True,
        help="Path to MOT tracking data file (CSV format)",
    )

    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing video frame images",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save output video",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["gt", "det"],
        default="gt",
        help="Visualization mode: 'gt' for ground truth, 'det' for detections",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="Frames per second for output video (default: 25)",
    )

    parser.add_argument(
        "--show_occluder",
        action="store_true",
        default=True,
        help="Highlight occluded objects (default: True)",
    )

    parser.add_argument(
        "--no_occluder",
        action="store_false",
        dest="show_occluder",
        help="Don't highlight occluded objects",
    )

    parser.add_argument(
        "--im_scale",
        type=float,
        default=1.0,
        help="Image scaling factor (default: 1.0)",
    )

    parser.add_argument(
        "--display_name",
        type=str,
        default="mot_visualization",
        help="Name for output video file (default: mot_visualization)",
    )

    parser.add_argument(
        "--display_time",
        action="store_true",
        default=True,
        help="Display frame timestamps (default: True)",
    )

    parser.add_argument(
        "--no_time",
        action="store_false",
        dest="display_time",
        help="Don't display frame timestamps",
    )

    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (default: all frames)",
    )

    args = parser.parse_args()

    try:
        # Create visualizer
        visualizer = MOTVisualizer(
            mot_file=args.mot_file,
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            mode=args.mode,
            fps=args.fps,
            show_occluder=args.show_occluder,
            im_scale=args.im_scale,
        )

        # Generate video
        output_path = visualizer.generate_video(
            display_time=args.display_time,
            display_name=args.display_name,
            max_frames=args.max_frames,
        )

        print(f"Success! Video saved to: {output_path}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

