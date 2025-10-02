"""
ENROS MOT Inference and Visualization

This script loads a trained ENROS checkpoint and runs inference on MOT data,
visualizing the model's predictions.

Usage:
    python enros_mot_inference.py --checkpoint_dir experiments/mot_graph_20250101 \
                                  --mot_dir data/MOT16/train/MOT16-02 \
                                  --output_dir output_videos \
                                  --fps 25

Author: Generated for entity-rl project
"""

import argparse
import random
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import gymnasium as gym
import numpy as np
import torch
import yaml
from torch_geometric.data import Batch
from tqdm import tqdm

from entity_rl import utils
from entity_rl.datasets import MOTDataset
from entity_rl.datasets.graph_utils import create_graph_observation_space
from entity_rl.datasets.mot_data import MOTDataLoader, scale_bboxes
from entity_rl.models.enros import ENROSPolicy


class ENROSMOTVisualizer:
    """Visualizer for ENROS predictions on MOT data."""

    # Color palette
    COLORS = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
    ]

    def __init__(
        self,
        checkpoint_dir: Path,
        mot_dir: Path,
        output_dir: Path,
        device: str = "cuda",
        fps: int = 25,
        agent_radius: float = 0.02,
        max_entities: int = 100,
        connect_threshold: float = 50.0,
        use_props: bool = False,
        include_agent_node: bool = False,
    ):
        """
        Initialize the ENROS MOT visualizer.

        Args:
            checkpoint_dir: Path to experiment directory (will auto-detect config and best model)
            mot_dir: MOT sequence directory
            output_dir: Directory to save output videos
            device: Device for inference ('cuda' or 'cpu')
            fps: Frames per second for output video
            agent_radius: Radius of synthetic agent
            max_entities: Maximum number of entities to process
            connect_threshold: Distance threshold for graph edges
            use_props: Whether to use proposals (False = use GT)
            include_agent_node: Whether to include agent node in graph
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.mot_dir = Path(mot_dir)
        self.output_dir = Path(output_dir)
        self.device = torch.device(device)
        self.fps = fps

        # Dataset parameters
        self.agent_radius = agent_radius
        self.max_entities = max_entities
        self.connect_threshold = connect_threshold
        self.use_props = use_props
        self.include_agent_node = include_agent_node

        # Auto-detect config and checkpoint
        self.config_path = self._find_config()
        self.checkpoint_path = self._find_best_checkpoint()

        # Validate inputs
        self._validate_inputs()

        # Load model
        print("Loading model...")
        self.model = self._load_model()
        self.model.eval()

        # Create dataset (reuse training dataset logic)
        print(f"Creating dataset from {mot_dir}...")
        self.dataset = MOTDataset(
            mot_data_dirs=[str(mot_dir)],
            agent_radius=agent_radius,
            num_samples_per_epoch=1000,  # Not used for inference
            image_size=(100, 100),  # Not used for inference
            max_entities=max_entities,
            connect_threshold=connect_threshold,
            max_samples=None,
            use_props=use_props,
            include_agent_node=include_agent_node,
        )

        # Get frame info
        self.data_dir = list(self.dataset.gt_data.keys())[0]
        self.frame_ids = sorted(self.dataset.gt_data[self.data_dir].keys())
        print(f"Found {len(self.frame_ids)} frames")

    def _find_config(self) -> Path:
        """Find config file in checkpoint directory."""
        for config_name in ["conf.yaml", "config.yaml"]:
            config_path = self.checkpoint_dir / config_name
            if config_path.exists():
                print(f"Found config: {config_path}")
                return config_path

        raise FileNotFoundError(
            f"No config file (conf.yaml or config.yaml) found in {self.checkpoint_dir}"
        )

    def _find_best_checkpoint(self) -> Path:
        """Find best model checkpoint in directory."""
        # Look for best metric checkpoints (saved by save_best_models)
        metric_priority = [
            "latest.pt",
        ]

        # Check in checkpoints subdirectory first
        checkpoints_dir = self.checkpoint_dir / "checkpoints"
        if checkpoints_dir.exists():
            for ckpt_name in metric_priority:
                ckpt_path = checkpoints_dir / ckpt_name
                if ckpt_path.exists():
                    print(f"Found checkpoint: {ckpt_path}")
                    return ckpt_path

        raise FileNotFoundError(
            f"No checkpoint file (.pt) found in {self.checkpoint_dir}"
        )

    def _validate_inputs(self):
        """Validate input paths."""
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint directory not found: {self.checkpoint_dir}"
            )
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        if not self.mot_dir.exists():
            raise FileNotFoundError(f"MOT directory not found: {self.mot_dir}")

    def _load_model(self) -> ENROSPolicy:
        """Load ENROS model from checkpoint."""
        # Load config
        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        # Extract model config (handle both direct and nested structures)
        if "base" in config:
            model_config = config["base"]["model"]
        else:
            model_config = config["model"]

        # Create observation space for graph data
        obs_space = create_graph_observation_space(node_feature_dim=5)
        action_space = gym.spaces.MultiDiscrete([3, 3])

        # Create model
        model = ENROSPolicy(
            obs_space,
            action_space,
            num_outputs=6,
            model_config=model_config,
            name="enros_mot_inference",
        )

        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Handle different checkpoint formats
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.to(self.device)

        print(f"Loaded checkpoint from {self.checkpoint_path}")
        print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

        return model

    def _run_inference(
        self, graph_data, agent_pos: torch.Tensor
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Run model inference and extract attention weights.

        Args:
            graph_data: Graph data from dataset
            agent_pos: Agent position tensor [x, y, w, h]

        Returns:
            Tuple of (value_prediction, attention_weights)
            - value_prediction: Tensor with predicted value
            - attention_weights: numpy array of attention weights per node (or None)
        """
        # Prepare observation (same as training)
        batch = Batch.from_data_list([graph_data])
        obs_batch = {
            "x": batch.x.to(self.device),
            "edge_index": batch.edge_index.to(self.device),
            "batch": batch.batch.to(self.device),
        }

        agent_pos = agent_pos.unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            model_input = {"obs": obs_batch, "agent_pos": agent_pos}
            _ = self.model(model_input)
            value = self.model.value_function()

        # Extract attention weights from pooling layer if available
        attention_weights = None
        try:
            # Navigate to GNN encoder's aggregation layer
            encoder = self.model._encoder
            attention_weights = (
                encoder._stages[1]._encoder[0]._aggr.attention_acts.numpy()
            )
            attention_weights = attention_weights - attention_weights.min()
            attention_weights = attention_weights / attention_weights.max()
            # print(attention_weights.min(), attention_weights.max())
        except Exception as e:
            print(f"Warning: Could not extract attention weights: {e}")

        return value.cpu(), attention_weights

    def _draw_predictions(
        self,
        image: np.ndarray,
        bboxes: List[Tuple],
        agent_pos: Tuple[float, float, float, float],
        value_pred: float,
        frame_id: int,
        attention_weights: np.ndarray = None,
    ) -> np.ndarray:
        """
        Draw bounding boxes, agent, and predictions on image.

        Args:
            image: Input image
            bboxes: List of scaled bboxes (x, y, w, h, track_id) in normalized coords
            agent_pos: Agent position (x, y, radius_x, radius_y) in normalized coords
            value_pred: Predicted value (classification: 0=collision, 1=safe)
            frame_id: Frame number
            attention_weights: Optional attention weights per node

        Returns:
            Image with visualizations
        """
        h, w = image.shape[:2]
        agent_x, agent_y, agent_w, agent_h = agent_pos

        # Determine offset for attention weights
        # If include_agent_node is True, first node is agent, rest are bboxes
        attention_offset = 1 if self.include_agent_node else 0

        # Draw bounding boxes
        for bbox_idx, (x, y, bbox_w, bbox_h, track_id) in enumerate(bboxes):
            # Convert normalized coords to pixels
            px = int(x * w)
            py = int(y * h)
            pw = int(bbox_w * w)
            ph = int(bbox_h * h)

            # Color based on attention weight if available
            if attention_weights is not None and bbox_idx + attention_offset < len(
                attention_weights
            ):
                att_val = attention_weights[bbox_idx + attention_offset]
                # Map attention to color: low attention = blue (cold), high = red (hot)
                # Use matplotlib colormap for hot (0=black, 1=red)
                intensity = np.clip(att_val, 0, 1)
                # Hot colormap: interpolate from dark blue (low) to bright red (high)
                color = (
                    int(intensity * 255),  # Blue channel
                    int(intensity * 255),  # Green channel
                    int(intensity * 255),  # Red channel
                )
            else:
                color = (200, 200, 200)  # Gray (default)

            thickness = 3 if attention_weights is not None else 2
            cv2.rectangle(image, (px, py), (px + pw, py + ph), color, thickness)

            # Draw attention value if available
            if attention_weights is not None and bbox_idx + attention_offset < len(
                attention_weights
            ):
                att_val = attention_weights[bbox_idx + attention_offset]
                cv2.putText(
                    image,
                    f"{att_val[0]:.3f}",
                    (px, py - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                )
            else:
                cv2.putText(
                    image,
                    f"ID:{int(track_id)}",
                    (px, py - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )

        # Draw agent
        agent_px = int(agent_x * w)
        agent_py = int(agent_y * h)
        agent_radius_px = int(agent_w * min(w, h))

        # Color based on prediction (green=safe, red=collision)
        # Value is classification: 0=collision, 1=safe
        if value_pred > 0.5:
            agent_color = (0, 255, 0)  # Green - safe
            status = "SAFE"
        else:
            agent_color = (0, 0, 255)  # Red - collision
            status = "COLLISION"

        cv2.circle(image, (agent_px, agent_py), agent_radius_px, agent_color, -1)
        cv2.circle(image, (agent_px, agent_py), agent_radius_px, (255, 255, 255), 2)

        # Draw prediction text
        pred_text = f"Prediction: {status} (value={value_pred:.3f})"
        cv2.putText(
            image,
            pred_text,
            (10, h - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Draw frame number
        cv2.putText(
            image,
            f"Frame: {frame_id}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        # Draw attention colorbar legend if attention weights are available
        if attention_weights is not None:
            # Draw colorbar in bottom-right corner
            bar_width = 200
            bar_height = 20
            bar_x = w - bar_width - 20
            bar_y = h - bar_height - 60

            # Draw gradient bar
            for i in range(bar_width):
                intensity = i / bar_width
                color = (
                    int(intensity * 255),  # Blue
                    int(intensity * 128),  # Green
                    int(255 * intensity),  # Red
                )
                cv2.line(
                    image, (bar_x + i, bar_y), (bar_x + i, bar_y + bar_height), color, 1
                )

            # Draw border
            cv2.rectangle(
                image,
                (bar_x, bar_y),
                (bar_x + bar_width, bar_y + bar_height),
                (255, 255, 255),
                1,
            )

            # Labels
            cv2.putText(
                image,
                "0.0",
                (bar_x - 5, bar_y + bar_height + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )
            cv2.putText(
                image,
                "0.5",
                (bar_x + bar_width // 2 - 10, bar_y + bar_height + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )
            cv2.putText(
                image,
                "1.0",
                (bar_x + bar_width - 15, bar_y + bar_height + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )
            cv2.putText(
                image,
                "Attention",
                (bar_x + bar_width // 2 - 30, bar_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        return image

    def generate_video(
        self, output_name: str = "enros_mot_inference", num_frames: int = None
    ) -> str:
        """
        Generate video with ENROS predictions.

        Args:
            output_name: Name for output video file
            num_frames: Number of frames to process (None = all frames)

        Returns:
            Path to generated video file
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get image dimensions from first frame
        first_frame_path = self.mot_dir / "img1" / f"{self.frame_ids[0]:06d}.jpg"
        first_image = cv2.imread(str(first_frame_path))
        if first_image is None:
            raise ValueError(f"Could not read first frame: {first_frame_path}")

        height, width = first_image.shape[:2]

        # Get original dimensions for bbox scaling
        orig_w, orig_h = self.dataset.gt_data_loader.get_image_dimensions(
            self.data_dir, self.frame_ids[0]
        )

        # Create video writer
        output_path = self.output_dir / f"{output_name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(output_path), fourcc, self.fps, (width, height)
        )

        print(f"Generating video: {output_path}")

        # Limit frames if specified
        frames_to_process = (
            self.frame_ids[:num_frames] if num_frames else self.frame_ids
        )
        print(f"Processing {len(frames_to_process)} frames...")

        # Track accuracy
        total_matches = 0
        num_samples = 0

        # Process each frame
        for frame_id in tqdm(frames_to_process, desc="Processing frames"):
            # Load image
            img_path = self.mot_dir / "img1" / f"{frame_id:06d}.jpg"
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"Warning: Could not read frame {img_path}")
                continue

            # Generate random agent position for this frame
            # agent_x = random.random()
            # agent_y = random.random()
            agent_x = 0.25
            agent_y = 0.6
            agent_pos_tuple = (agent_x, agent_y, self.agent_radius, self.agent_radius)

            # Get sample for this specific frame with this agent position
            data_dict, reward_gt = self.dataset.get_sample(
                self.data_dir, frame_id, agent_pos_tuple
            )
            graph_data = data_dict["graph"]
            agent_pos = data_dict["agent_pos"]

            # Run inference and extract attention weights
            value, attention_weights = self._run_inference(graph_data, agent_pos)

            # Calculate accuracy (same as training/evaluation)
            # Dataset returns: 0 (collision) or 1 (safe)
            reward_class = reward_gt  # Already 0 or 1
            # Get prediction class: value >= 0.5 -> safe (1), else collision (0)
            # NOTE: threshold depends on reward range
            pred_class = 1 if value.item() >= 0 else -1
            # Track match
            match = 1 if pred_class == reward_class else 0
            total_matches += match
            num_samples += 1

            # Get bboxes for visualization (scaled to [0,1])
            bboxes = self.dataset.gt_data[self.data_dir][frame_id]
            scaled_bboxes = scale_bboxes(bboxes, (orig_w, orig_h))

            # Draw visualizations with attention weights
            frame = self._draw_predictions(
                frame,
                scaled_bboxes,
                agent_pos.numpy(),
                value.item(),
                frame_id,
                attention_weights,
            )

            video_writer.write(frame)

        video_writer.release()
        print(f"Video generation complete: {output_path}")

        # Print accuracy summary
        accuracy = total_matches / num_samples if num_samples > 0 else 0.0
        print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Correct predictions: {total_matches}/{num_samples}")

        return str(output_path)


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Run ENROS inference on MOT data with visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --checkpoint_dir experiments/mot_graph_20250101 \\
           --mot_dir data/MOT16/train/MOT16-02 \\
           --output_dir output_videos

  %(prog)s --checkpoint_dir path/to/experiment \\
           --mot_dir path/to/MOT/sequence \\
           --output_dir videos \\
           --device cuda \\
           --fps 30 \\
           --num_frames 100
        """,
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to experiment directory (will auto-detect config and best model)",
    )

    parser.add_argument(
        "--mot_dir",
        type=str,
        required=True,
        help="Path to MOT sequence directory",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save output videos",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference (default: cuda if available)",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="Frames per second for output video (default: 25)",
    )

    parser.add_argument(
        "--agent_radius",
        type=float,
        default=0.02,
        help="Agent radius for collision detection (default: 0.02)",
    )

    parser.add_argument(
        "--max_entities",
        type=int,
        default=100,
        help="Maximum number of entities to process (default: 100)",
    )

    parser.add_argument(
        "--connect_threshold",
        type=float,
        default=50.0,
        help="Distance threshold for graph edges (default: 50.0)",
    )

    parser.add_argument(
        "--use_props",
        action="store_true",
        help="Use proposals instead of ground truth",
    )

    parser.add_argument(
        "--include_agent_node",
        action="store_true",
        help="Include agent as a node in the graph",
    )

    parser.add_argument(
        "--output_name",
        type=str,
        default="enros_mot_inference",
        help="Name for output video file (default: enros_mot_inference)",
    )

    parser.add_argument(
        "--num_frames",
        type=int,
        default=None,
        help="Number of frames to process (default: all frames)",
    )

    args = parser.parse_args()

    # Create visualizer
    visualizer = ENROSMOTVisualizer(
        checkpoint_dir=Path(args.checkpoint_dir),
        mot_dir=Path(args.mot_dir),
        output_dir=Path(args.output_dir),
        device=args.device,
        fps=args.fps,
        agent_radius=args.agent_radius,
        max_entities=args.max_entities,
        connect_threshold=args.connect_threshold,
        use_props=args.use_props,
        include_agent_node=args.include_agent_node,
    )

    # Generate video
    output_path = visualizer.generate_video(
        output_name=args.output_name,
        num_frames=args.num_frames,
    )

    print(f"Success! Video saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
