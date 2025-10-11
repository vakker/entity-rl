import argparse
import os

import cv2
import gymnasium as gym
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from entity_rl import utils
from entity_rl.models.enros import ENROSPolicy


def load_model_from_config(cfg_path, device="cuda"):
    """Load ENROS model from config file."""
    print(f"Loading config from: {cfg_path}")
    conf = utils.load_dict(cfg_path)["base"]

    obs_space = gym.spaces.Box(low=0, high=255, shape=(500, 500, 3), dtype=np.uint8)
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
    """Load and preprocess image for RPN."""
    if isinstance(image_path, str):
        img = Image.open(image_path).convert("RGB")
    else:
        img = image_path

    # Resize to target size
    img = img.resize(target_size)
    img_array = np.array(img)

    # Stack as single frame (RPN expects stack_depth=1)
    img_stacked = img_array  # Shape: (H, W, 3)

    # Add batch dimension and convert to tensor
    img_tensor = torch.tensor(img_stacked, dtype=torch.float32).unsqueeze(
        0
    )  # (1, H, W, 3)

    return img_tensor, img_array


def visualize_proposals(
    image, proposals, scores, max_proposals=20, score_threshold=0.1
):
    """Visualize RPN proposals on the image."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)

    # Filter proposals by score
    valid_mask = scores.flatten() > score_threshold
    if valid_mask.sum() == 0:
        print(f"No proposals above threshold {score_threshold}")
        valid_mask = scores.flatten() > 0  # Show all non-zero proposals

    valid_proposals = proposals[valid_mask][:max_proposals]
    valid_scores = scores[valid_mask][:max_proposals]

    print(f"Showing {len(valid_proposals)} proposals (threshold: {score_threshold})")

    colors = plt.cm.rainbow(np.linspace(0, 1, len(valid_proposals)))

    for i, (bbox, score, color) in enumerate(
        zip(valid_proposals, valid_scores, colors)
    ):
        # bbox is in format [x1, y1, x2, y2] in pixel coordinates
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        # Create rectangle patch
        rect = patches.Rectangle(
            (x1, y1),
            width,
            height,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
            alpha=0.8,
        )
        ax.add_patch(rect)

        # Add score text
        ax.text(
            x1,
            y1 - 5,
            f"{score:.3f}",
            color=color,
            fontsize=8,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
        )

    ax.set_title(f"RPN Proposals (Top {len(valid_proposals)})")
    ax.axis("off")

    return fig


def extract_rpn_proposals(model, input_tensor, device):
    """Extract RPN proposals from the model."""
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        # Run forward pass
        model_input = {"obs": input_tensor}
        _ = model(model_input)

        # Get entity encoder (RPN)
        entity_encoder = model._encoder._stages[0]

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


def main():
    parser = argparse.ArgumentParser(description="Visualize RPN proposals")
    parser.add_argument("--cfg", required=True, help="Path to training config file")
    parser.add_argument(
        "--image",
        help="Path to input image (if not provided, will use webcam or generate random)",
    )
    parser.add_argument("--output", help="Path to save visualization (optional)")
    parser.add_argument("--device", default="cuda", help="Device to run on")
    parser.add_argument(
        "--max-proposals", type=int, default=20, help="Maximum proposals to show"
    )
    parser.add_argument(
        "--score-threshold", type=float, default=0.1, help="Minimum score threshold"
    )
    parser.add_argument(
        "--size", nargs=2, type=int, default=[500, 500], help="Target image size (H W)"
    )

    args = parser.parse_args()

    # Load model
    model = load_model_from_config(args.cfg, args.device)

    # Get input image
    if args.image:
        if not os.path.exists(args.image):
            raise FileNotFoundError(f"Image not found: {args.image}")
        input_tensor, original_image = preprocess_image(args.image, tuple(args.size))
        print(f"Loaded image: {args.image}")
    else:
        raise RuntimeError("No image provided")

    print(f"Input tensor shape: {input_tensor.shape}")

    # Extract proposals
    print("Extracting RPN proposals...")
    try:
        bboxes, scores = extract_rpn_proposals(model, input_tensor, args.device)
        print(f"Found {len(bboxes)} proposals")

        if len(bboxes) == 0:
            print("No proposals found!")
            return

        # Print some statistics
        print(f"Score range: {scores.min():.4f} - {scores.max():.4f}")
        print(
            f"Above threshold ({args.score_threshold}): {(scores > args.score_threshold).sum()}"
        )

    except Exception as e:
        print(f"Error extracting proposals: {e}")
        print("Make sure your config uses RPNEncoder")
        return

    # Visualize
    print("Creating visualization...")
    fig = visualize_proposals(
        original_image, bboxes, scores, args.max_proposals, args.score_threshold
    )

    # Save or show
    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to: {args.output}")
    else:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()

