import argparse
import csv
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from entity_rl import utils
from entity_rl.models.enros import ENROSPolicy


def load_model_from_config(cfg_path, obs_shape, device="cuda"):
    """Load ENROS model with DINO from config file."""
    print(f"Loading config from: {cfg_path}")
    conf = utils.load_dict(cfg_path)["base"]

    obs_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
    action_space = gym.spaces.MultiDiscrete([3, 3])

    model = ENROSPolicy(
        obs_space,
        action_space,
        num_outputs=6,
        model_config=conf["model"],
        name="enros",
    )

    print("Model created, parameters:")
    print(model.show_trainable_params())

    model.to(device)
    model.eval()

    return model


def preprocess_image(image_path, target_size=(210, 160)):
    """Load and preprocess image for DINO.

    Returns:
        tuple: (input_tensor, img_array, orig_size) where orig_size is (width, height)
    """
    if isinstance(image_path, str):
        img = Image.open(image_path).convert("RGB")
    else:
        img = image_path

    # Get original dimensions
    orig_width, orig_height = img.size

    # Resize to target size
    img = img.resize(target_size)
    img_array = np.array(img)

    # Stack as single frame (DINO expects stack_depth=1)
    img_stacked = img_array  # Shape: (H, W, 3)

    # Add batch dimension and convert to tensor
    img_tensor = torch.tensor(img_stacked, dtype=torch.float32).unsqueeze(
        0
    )  # (1, H, W, 3)

    return img_tensor, img_array, (orig_width, orig_height)


def extract_dino_detections_and_features(model, input_tensor, device):
    """Extract DINO detections and query features from the model.

    Returns:
        tuple: (bboxes, scores, labels, query_features)
            - bboxes: numpy array (N, 4) in xyxy format
            - scores: numpy array (N,) objectness scores
            - labels: numpy array (N,) predicted class labels
            - query_features: numpy array (N, 256) query embeddings from decoder
    """
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        # Run forward pass
        model_input = {"obs": input_tensor}

        # Get entity encoder (DINO)
        entity_encoder = model._encoder._stages[0]
        _ = entity_encoder(model_input['obs'])

        if (
            not hasattr(entity_encoder, "dino_outputs")
            or entity_encoder.dino_outputs is None
        ):
            raise RuntimeError(
                "No DINO outputs found. Make sure you're using DINOEncoder."
            )

        # Extract detections and query features from the stored outputs
        # dino_outputs is a list (one per frame in stack, usually just 1)
        dino_output = entity_encoder.dino_outputs[0]  # First frame

        # Extract from DINO outputs dict
        features = dino_output["features"]  # (B, N, 256) query embeddings
        bboxes = dino_output["bboxes"]      # (B, N, 4) predicted bboxes
        scores = dino_output["scores"]      # (B, N, 1) objectness scores
        labels = dino_output["labels"]      # (B, N) predicted class labels

        # Get the first batch item
        batch_bboxes = bboxes[0].cpu().numpy()  # (N, 4) in xyxy format
        batch_scores = scores[0].cpu().numpy().squeeze(-1)  # (N,) - squeeze only last dim
        batch_labels = labels[0].cpu().numpy()  # (N,)

        # Get query features for the first batch item
        batch_features = features[0].cpu().numpy()  # (N, 256)

        # Filter out zero detections (padding)
        valid_mask = np.abs(batch_bboxes).sum(axis=1) > 1e-6
        if valid_mask.any():
            batch_bboxes = batch_bboxes[valid_mask]
            batch_scores = batch_scores[valid_mask]
            batch_labels = batch_labels[valid_mask]
            batch_features = batch_features[valid_mask]

        return batch_bboxes, batch_scores, batch_labels, batch_features


def save_detections_to_csv(output_path, all_detections):
    """Save DINO detections to CSV file in MOT format."""
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write header (MOT format: frame_id, track_id, x, y, width, height, confidence, class_id, visibility)
        writer.writerow(
            [
                "frame_id",
                "track_id",
                "x",
                "y",
                "width",
                "height",
                "confidence",
                "class_id",
                "visibility",
            ]
        )

        for frame_id, (bboxes, scores, labels) in all_detections.items():
            for bbox, score, label in zip(bboxes, scores, labels):
                x1, y1, x2, y2 = bbox
                # Convert xyxy to xywh
                x = x1
                y = y1
                width = x2 - x1
                height = y2 - y1
                # MOT format: frame_id, track_id (-1 for detections), x, y, width, height, confidence, class_id, visibility (1.0)
                writer.writerow([frame_id, -1, x, y, width, height, score, int(label), 1.0])


def save_features_to_npz(output_path, all_features):
    """Save DINO query features to NPZ file."""
    # Convert dict of arrays to a format suitable for np.savez (keys must be strings)
    features_dict = {f"frame_{frame_id}": features for frame_id, features in all_features.items()}
    np.savez_compressed(output_path, **features_dict)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess data using DINO entity encoder (saves detections and query features)"
    )
    parser.add_argument(
        "--cfg",
        default="configs/dino-preprocess.yaml",
        help="Path to training config file (default: configs/dino-preprocess.yaml)",
    )
    parser.add_argument(
        "--input-dir", required=True, help="Directory containing input images"
    )
    parser.add_argument(
        "--output", required=True, help="Path to save preprocessed detections (CSV) and features (NPZ)"
    )
    parser.add_argument("--device", default="cuda", help="Device to run on")
    parser.add_argument(
        "--score-threshold", type=float, default=0.0, help="Minimum score threshold"
    )
    parser.add_argument(
        "--size", nargs=2, type=int, default=[500, 500], help="Target image size (H W)"
    )
    parser.add_argument(
        "--max-frames", type=int, help="Maximum number of frames to process"
    )

    args = parser.parse_args()

    # Load model
    obs_shape = (args.size[0], args.size[1], 3)  # (H, W, 3)
    model = load_model_from_config(args.cfg, obs_shape, args.device)

    # Get input images
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Find all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = sorted(
        [
            f
            for f in input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
    )

    if not image_files:
        raise FileNotFoundError(f"No image files found in {input_dir}")

    if args.max_frames:
        image_files = image_files[: args.max_frames]

    print(f"Found {len(image_files)} images to process")

    # Process images
    all_detections = {}
    all_features = {}
    failed_frames = []

    for i, img_path in enumerate(tqdm(image_files, desc="Processing images")):
        # Extract frame ID from filename (assuming format like 000001.jpg)
        try:
            frame_id = int(img_path.stem)
        except ValueError:
            print(f"Warning: Could not parse frame ID from {img_path.name}, using index {i+1}")
            frame_id = i + 1

        # Preprocess image
        # PIL resize expects (W, H), so swap args.size which is [H, W]
        input_tensor, _, orig_size = preprocess_image(str(img_path), (args.size[1], args.size[0]))
        orig_width, orig_height = orig_size
        target_height, target_width = args.size

        # Extract detections and query features
        bboxes, scores, labels, features = extract_dino_detections_and_features(model, input_tensor, args.device)

        # Filter by score threshold
        if args.score_threshold > 0:
            valid_mask = scores > args.score_threshold
            bboxes = bboxes[valid_mask]
            scores = scores[valid_mask]
            labels = labels[valid_mask]
            features = features[valid_mask]

        # Scale bounding boxes back to original image size
        if len(bboxes) > 0:
            scale_x = orig_width / target_width
            scale_y = orig_height / target_height
            bboxes = bboxes.copy()
            bboxes[:, [0, 2]] *= scale_x  # x1, x2
            bboxes[:, [1, 3]] *= scale_y  # y1, y2

        all_detections[frame_id] = (bboxes, scores, labels)
        all_features[frame_id] = features

        # except Exception as e:
        #     print(f"Error processing frame {frame_id}: {e}")
        #     failed_frames.append((frame_id, str(e)))
        #     # Add empty detections for this frame
        #     all_detections[frame_id] = (np.empty((0, 4)), np.empty((0,)), np.empty((0,)))
        #     all_features[frame_id] = np.empty((0, 256))

    # Save results
    save_detections_to_csv(args.output, all_detections)
    print(f"Saved {len(all_detections)} frames to {args.output}")

    # Save query features
    features_output = args.output.replace('.csv', '_features.npz')
    save_features_to_npz(features_output, all_features)
    print(f"Saved query features to {features_output}")

    # Report statistics
    total_detections = sum(len(dets[0]) for dets in all_detections.values())
    avg_detections = total_detections / len(all_detections) if all_detections else 0
    print(f"\nStatistics:")
    print(f"  Total detections: {total_detections}")
    print(f"  Average detections per frame: {avg_detections:.2f}")

    if failed_frames:
        print(f"\nFailed frames ({len(failed_frames)}):")
        for frame_id, error in failed_frames[:10]:  # Show first 10
            print(f"  Frame {frame_id}: {error}")
        if len(failed_frames) > 10:
            print(f"  ... and {len(failed_frames) - 10} more")


if __name__ == "__main__":
    main()
