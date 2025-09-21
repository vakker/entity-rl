"""
Training utilities for MOT-based ENROS training.

This module provides common training functionality shared across MOT training scripts.
"""

import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from entity_rl.models.enros import ENROSPolicy


class RewardLabelAdapter:
    """Adapter class to convert rewards to class labels for training."""

    def __init__(self, base_dataset):
        """
        Initialize adapter.

        Args:
            base_dataset: Base dataset to wrap
        """
        self.base_dataset = base_dataset
        self.label_map = {-1: 0.0, 1: 1.0}

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        sample, reward = self.base_dataset[index]
        # Convert reward to class label
        reward_class = self.label_map[reward]
        return sample, reward_class


def setup_tensorboard(output_dir: str) -> SummaryWriter:
    """
    Set up TensorBoard logging.

    Args:
        output_dir: Output directory for logs

    Returns:
        TensorBoard writer
    """
    os.makedirs(output_dir, exist_ok=True)
    return SummaryWriter(output_dir)


def move_dict_to_device(data_dict: Dict, device: torch.device) -> Dict:
    """Move dictionary of tensors to device."""
    return {k: v.to(device) for k, v in data_dict.items()}


def save_model_checkpoint(
    model: ENROSPolicy,
    optimizer,
    epoch: int,
    val_loss: float,
    val_accuracy: float,
    output_dir: str,
    filename: str = "best_model.pt",
) -> None:
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        val_loss: Validation loss
        val_accuracy: Validation accuracy
        output_dir: Output directory
        filename: Filename for checkpoint
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
    }

    checkpoint_path = os.path.join(output_dir, filename)
    torch.save(checkpoint, checkpoint_path)


def add_bbox(frame, bbox_pred):
    """
    Add bounding boxes to frame for visualization.

    Args:
        frame: Input frame tensor
        bbox_pred: Predicted bounding boxes

    Returns:
        Frame with bounding boxes drawn
    """
    try:
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
    except ImportError:
        print("Warning: mmdet not available for bbox visualization")
        return frame


def process_outputs(obs_batch, bbox_preds_batch):
    """
    Process model outputs for visualization.

    Args:
        obs_batch: Batch of observations
        bbox_preds_batch: Batch of bbox predictions

    Returns:
        List of processed images
    """
    images = []
    for obs, bbox_preds in zip(obs_batch, bbox_preds_batch):
        frame = add_bbox(obs, bbox_preds).numpy()
        images.append(frame)
    return images


def save_samples(
    writer: SummaryWriter,
    obs_batch,
    model: ENROSPolicy,
    output_dir: str,
    global_step: int,
    save_bbox: bool = False,
) -> None:
    """
    Save sample visualizations.

    Args:
        writer: TensorBoard writer
        obs_batch: Batch of observations
        model: ENROS model
        output_dir: Output directory
        global_step: Current global step
        save_bbox: Whether to save bbox visualizations
    """
    output_path = Path(output_dir)

    # Save bbox visualizations if requested and available
    if save_bbox:
        try:
            bbox_preds = model._encoder._stages[0].gdino_outputs["bboxes"]
            images = process_outputs(obs_batch[:5], bbox_preds[:5])

            # Save individual frames
            frames_dir = output_path / "bboxes"
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
    try:
        obs_frames_dir = output_path / "obs_orig"
        obs_frames_dir.mkdir(exist_ok=True)

        for j, obs in enumerate(obs_batch[:5]):
            from skimage import io as skio

            img_path = obs_frames_dir / f"f-{global_step:03d}-{j:06d}.png"
            if isinstance(obs, torch.Tensor):
                skio.imsave(str(img_path), obs.numpy(), check_contrast=False)
    except Exception as e:
        print(f"Warning: Could not save observation samples: {e}")


def log_gradients(writer: SummaryWriter, model: ENROSPolicy, global_step: int) -> None:
    """
    Log gradient histograms to tensorboard.

    Args:
        writer: TensorBoard writer
        model: ENROS model
        global_step: Current global step
    """
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


def create_optimizer(model: ENROSPolicy, learning_rate: float) -> torch.optim.Optimizer:
    """
    Create optimizer for model training.

    Args:
        model: ENROS model
        learning_rate: Learning rate

    Returns:
        Adam optimizer
    """
    return torch.optim.Adam(model.parameters(), lr=learning_rate)


def create_loss_function(device: torch.device):
    """
    Create loss function for training.

    Args:
        device: Device to place loss function on

    Returns:
        CrossEntropyLoss function
    """
    return torch.nn.MSELoss().to(device)
    # return torch.nn.CrossEntropyLoss().to(device)


def setup_amp_scaler(model: ENROSPolicy) -> torch.cuda.amp.GradScaler:
    """
    Set up automatic mixed precision scaler.

    Args:
        model: ENROS model

    Returns:
        GradScaler for AMP training
    """
    use_amp = getattr(model, "use_amp", False)
    return torch.cuda.amp.GradScaler(enabled=use_amp)
