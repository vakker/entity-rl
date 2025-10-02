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
    """Load ENROS model from config file."""
    print(f"Loading config from: {cfg_path}")
    conf = utils.load_dict(cfg_path)["base"]
    # conf["model"]["custom_model_config"]["encoder"]["include_agent_pos"] = False
    # conf["model"]["custom_model_config"]["encoder"]["scene"] = {
    #     "name": "NoOpSceneEncoder"
    # }

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
    """Load and preprocess image for RPN.

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

    # Stack as single frame (RPN expects stack_depth=1)
    img_stacked = img_array  # Shape: (H, W, 3)

    # Add batch dimension and convert to tensor
    img_tensor = torch.tensor(img_stacked, dtype=torch.float32).unsqueeze(
        0
    )  # (1, H, W, 3)

    return img_tensor, img_array, (orig_width, orig_height)


def extract_rpn_proposals(model, input_tensor, device):
    """Extract RPN proposals from the model."""
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        # Run forward pass
        model_input = {"obs": input_tensor}

        # Get entity encoder (RPN)
        entity_encoder = model._encoder._stages[0]
        _ = entity_encoder(model_input['obs'])

        if (
            not hasattr(entity_encoder, "rpn_outputs")
            or entity_encoder.rpn_outputs is None
        ):
            raise RuntimeError(
                "No RPN outputs found. Make sure you're using RPNEncoder."
            )

        # Extract proposals from the stored outputs
        proposals = entity_encoder.rpn_outputs["proposals"]

        # Get the first batch item
        if len(proposals) > 0:
            batch_proposals = proposals[0]  # First batch item
            bboxes = batch_proposals.bboxes.cpu().numpy()  # (N, 4) in xyxy format
            scores = batch_proposals.scores.cpu().numpy()  # (N,)

            return bboxes, scores
        else:
            return np.empty((0, 4)), np.empty((0,))


def save_proposals_to_csv(output_path, all_proposals):
    """Save RPN proposals to CSV file in MOT format."""
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

        for frame_id, (bboxes, scores) in all_proposals.items():
            for bbox, score in zip(bboxes, scores):
                x1, y1, x2, y2 = bbox
                # Convert xyxy to xywh
                x = x1
                y = y1
                width = x2 - x1
                height = y2 - y1
                # MOT format: frame_id, track_id (-1 for proposals), x, y, width, height, confidence, class_id (1), visibility (1.0)
                writer.writerow([frame_id, -1, x, y, width, height, score, 1, 1.0])


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess data using RPN entity encoder"
    )
    parser.add_argument(
        "--cfg",
        default="configs/rpn-preprocess.yaml",
        help="Path to training config file (default: configs/rpn-preprocess.yaml)",
    )
    parser.add_argument(
        "--input-dir", required=True, help="Directory containing input images"
    )
    parser.add_argument(
        "--output", required=True, help="Path to save preprocessed data (CSV)"
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
    all_proposals = {}
    failed_frames = []

    for i, img_path in enumerate(tqdm(image_files, desc="Processing images")):
        # Extract frame ID from filename (assuming format like 000001.jpg)
        frame_id = int(img_path.stem)

        # Preprocess image
        # PIL resize expects (W, H), so swap args.size which is [H, W]
        input_tensor, _, orig_size = preprocess_image(str(img_path), (args.size[1], args.size[0]))
        orig_width, orig_height = orig_size
        target_height, target_width = args.size

        # Extract proposals
        bboxes, scores = extract_rpn_proposals(model, input_tensor, args.device)

        # Filter by score threshold
        if args.score_threshold > 0:
            valid_mask = scores > args.score_threshold
            bboxes = bboxes[valid_mask]
            scores = scores[valid_mask]

        # Scale bounding boxes back to original image size
        if len(bboxes) > 0:
            scale_x = orig_width / target_width
            scale_y = orig_height / target_height
            bboxes = bboxes.copy()
            bboxes[:, [0, 2]] *= scale_x  # x1, x2
            bboxes[:, [1, 3]] *= scale_y  # y1, y2

        all_proposals[frame_id] = (bboxes, scores)

    # Save results
    save_proposals_to_csv(args.output, all_proposals)
    print(f"Saved {len(all_proposals)} frames to {args.output}")

    if failed_frames:
        print(f"Failed to process {len(failed_frames)} frames")

    # Print statistics
    total_proposals = sum(len(bboxes) for bboxes, _ in all_proposals.values())
    print(f"Total proposals extracted: {total_proposals}")
    print(f"Average proposals per frame: {total_proposals / len(all_proposals):.2f}")


if __name__ == "__main__":
    main()
