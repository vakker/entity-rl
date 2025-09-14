"""
Training script for ENROS using MOT synthetic data.

This script uses the MOT synthetic dataset to train ENROS models on agent-environment
interactions derived from Multiple Object Tracking data.
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict

import gymnasium as gym
import numpy as np
import torch
from mot_synthetic_dataset import MOTSyntheticDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from entity_rl import utils
from entity_rl.models.enros import ENROSPolicy


class CustomDataset:
    """Adapter class to match the interface expected by the training loop."""

    def __init__(self, mot_dataset):
        self.mot_dataset = mot_dataset
        self.label_map = {0: 0, 1: 1, -1: 2}

    def __len__(self):
        return len(self.mot_dataset)

    def __getitem__(self, index):
        x, y = self.mot_dataset[index]
        # Convert reward to class label
        y_mapped = self.label_map[y]
        return x, y_mapped


def add_bbox(frame, bbox_pred):
    """Add bounding boxes to frame for visualization."""
    from mmdet.structures.bbox import bbox_cxcywh_to_xyxy
    from torchvision.utils import draw_bounding_boxes

    frame = frame.byte()
    img_shape = frame.shape[:2]
    det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
    det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
    det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
    det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
    det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])

    return draw_bounding_boxes(
        frame.permute(2, 0, 1), det_bboxes, colors="red"
    ).permute(1, 2, 0)


def process_outputs(obs_batch, bbox_preds_batch):
    """Process model outputs for visualization."""
    images = []
    for obs, bbox_preds in zip(obs_batch, bbox_preds_batch):
        frame = add_bbox(obs, bbox_preds).numpy()
        images.append(frame)
    return images


def train_model(
    model: ENROSPolicy,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device = torch.device("cuda"),
) -> None:
    """
    Main training loop for ENROS model.

    Args:
        model: ENROS policy model
        train_loader: Training data loader
        val_loader: Validation data loader
        args: Command line arguments
        device: Device to run training on
    """
    model.to(device)

    use_amp = model.use_amp
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(args.output_dir)

    global_step = 0

    # Initial validation
    model.eval()
    with torch.no_grad():
        val_loss = 0
        total_samples = 0
        for obs, rew in tqdm(val_loader, desc="Initial validation"):
            obs_orig = obs
            obs = obs.to(device)
            rew = rew.to(device)
            _ = model({"obs": obs})
            rew_pred = model.value_function()
            val_loss += loss_fn(rew_pred, rew).item()
            total_samples += 1

        val_loss /= total_samples
        writer.add_scalar("val_loss", val_loss, global_step)
        print(f"Initial validation loss: {val_loss:.4f}")

        # Save visualization samples
        if args.save_samples:
            save_samples(writer, obs_orig[:5], model, args, global_step)

    # Training loop
    model.train()
    for epoch in trange(args.epochs, desc="Training epochs"):
        epoch_loss = 0
        num_batches = 0

        for obs, rew in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            obs = obs.to(device)
            rew = rew.to(device)

            _ = model({"obs": obs})
            rew_pred = model.value_function()
            loss = loss_fn(rew_pred, rew)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Log metrics
            global_step += 1
            epoch_loss += loss.item()
            num_batches += 1
            writer.add_scalar("train_loss", loss.item(), global_step)

            # Log gradients periodically
            if args.log_gradients and global_step % 100 == 0:
                log_gradients(writer, model, global_step)

            optimizer.zero_grad(set_to_none=True)

        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            total_samples = 0
            correct_predictions = 0
            total_predictions = 0

            for obs, rew in tqdm(val_loader, desc="Validation"):
                obs_orig = obs
                obs = obs.to(device)
                rew = rew.to(device)
                _ = model({"obs": obs})
                rew_pred = model.value_function()
                val_loss += loss_fn(rew_pred, rew).item()
                total_samples += 1

                # Calculate accuracy
                pred_classes = torch.argmax(rew_pred, dim=1)
                correct_predictions += (pred_classes == rew).sum().item()
                total_predictions += rew.size(0)

            val_loss /= total_samples
            accuracy = correct_predictions / total_predictions
            writer.add_scalar("val_loss", val_loss, global_step)
            writer.add_scalar("val_accuracy", accuracy, global_step)
            print(f"Validation loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")

            # Save visualization samples
            if args.save_samples and (epoch + 1) % 5 == 0:
                save_samples(writer, obs_orig[:5], model, args, global_step)

        model.train()

    writer.close()
    print("Training completed!")


def save_samples(writer, obs_batch, model, args, global_step):
    """Save sample visualizations."""
    if not args.bbox:
        return

    try:
        # Get bounding box predictions if available
        bbox_preds = model._encoder._stages[0].gdino_outputs["bboxes"]
        images = process_outputs(obs_batch[:5], bbox_preds[:5])

        # Save individual frames
        frames_dir = Path(args.output_dir) / "bboxes"
        frames_dir.mkdir(exist_ok=True)

        for j, img in enumerate(images):
            from skimage import io as skio

            img_path = frames_dir / f"f-{global_step:03d}-{j:06d}.png"
            skio.imsave(str(img_path), img[:, :, :3], check_contrast=False)

        # Add to tensorboard
        writer.add_images(
            "bboxes",
            np.stack(images).transpose(0, 3, 1, 2)[:, :3],
            global_step=global_step,
        )
    except Exception as e:
        print(f"Warning: Could not save bbox visualizations: {e}")

    # Save original observations
    obs_frames_dir = Path(args.output_dir) / "obs_orig"
    obs_frames_dir.mkdir(exist_ok=True)

    for j, img in enumerate(obs_batch[:5]):
        from skimage import io as skio

        img_path = obs_frames_dir / f"f-{global_step:03d}-{j:06d}.png"
        skio.imsave(str(img_path), img.numpy(), check_contrast=False)


def log_gradients(writer, model, global_step):
    """Log gradient histograms to tensorboard."""
    try:
        if hasattr(model._encoder._stages[0], "_model"):
            text_embed = model._encoder._stages[0]._model.text_embed.weight
            if text_embed.grad is not None:
                writer.add_histogram(
                    "text_embed_grad",
                    text_embed.grad.data.cpu().numpy(),
                    global_step,
                )
    except Exception as e:
        print(f"Warning: Could not log gradients: {e}")


def main(args):
    """Main training function."""
    print(f"Using MOT directories: {args.mot_dirs}")
    print(f"Output directory: {args.output_dir}")
    print(f"Image size: {args.image_size}")
    print(f"Agent radius: {args.agent_radius}")

    # Create synthetic dataset
    full_dataset = MOTSyntheticDataset(
        mot_data_dirs=args.mot_dirs,
        agent_radius=args.agent_radius,
        num_samples_per_epoch=args.num_samples,
        image_size=tuple(args.image_size),
        use_gt=not args.use_detections,
    )

    print(f"Created dataset with {len(full_dataset)} samples per epoch")

    # Create train/val split using different random seeds
    train_dataset = MOTSyntheticDataset(
        mot_data_dirs=args.mot_dirs,
        agent_radius=args.agent_radius,
        num_samples_per_epoch=int(args.num_samples * 0.8),  # 80% for training
        image_size=tuple(args.image_size),
        use_gt=not args.use_detections,
    )

    val_dataset = MOTSyntheticDataset(
        mot_data_dirs=args.mot_dirs,
        agent_radius=args.agent_radius,
        num_samples_per_epoch=int(args.num_samples * 0.2),  # 20% for validation
        image_size=tuple(args.image_size),
        use_gt=not args.use_detections,
    )

    # Wrap with adapter
    train_wrapped = CustomDataset(train_dataset)
    val_wrapped = CustomDataset(val_dataset)

    # Create data loaders
    train_loader = DataLoader(
        train_wrapped,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
    )

    val_loader = DataLoader(
        val_wrapped,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=2,
    )

    # Set up model
    obs_space = gym.spaces.Box(
        low=0,
        high=255,
        shape=(args.image_size[1], args.image_size[0], 3),
        dtype=np.uint8,
    )
    action_space = gym.spaces.MultiDiscrete([3, 3])

    conf = utils.load_dict(args.cfg)["base"]

    model = ENROSPolicy(
        obs_space,
        action_space,
        num_outputs=6,
        model_config=conf["model"],
        name="enros",
    )

    print(f"Model created: {sum(p.numel() for p in model.parameters())} parameters")

    # Train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_model(model, train_loader, val_loader, args, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ENROS on MOT synthetic data")

    # Data arguments
    parser.add_argument(
        "--mot-dirs", nargs="+", required=True, help="List of MOT data directories"
    )
    parser.add_argument(
        "--use-detections",
        action="store_true",
        help="Use detection files instead of ground truth",
    )

    # Model arguments
    parser.add_argument("--cfg", required=True, help="Config file path")

    # Training arguments
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--num-samples", type=int, default=2000, help="Number of samples per epoch"
    )

    # Dataset arguments
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

    # Visualization arguments
    parser.add_argument("--bbox", action="store_true", help="Enable bbox visualization")
    parser.add_argument(
        "--save-samples", action="store_true", help="Save sample images"
    )
    parser.add_argument(
        "--log-gradients", action="store_true", help="Log gradient histograms"
    )

    args = parser.parse_args()
    main(args)

