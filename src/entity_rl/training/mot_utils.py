import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from entity_rl.models.enros import ENROSPolicy


class GNNDatasetAdapter(Dataset):
    """Adapter for GNN dataset to provide correct observation format."""

    def __init__(self, base_dataset):
        """
        Initialize adapter.

        Args:
            base_dataset: Base dataset to wrap
        """
        self.base_dataset = base_dataset
        self.label_map = {0: 0.0, 1: 1.0}

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        result = self.base_dataset[index]

        assert len(result) == 2
        data_dict, reward = result

        # Extract components
        graph_data = data_dict["graph"]
        agent_pos = data_dict["agent_pos"]

        # Convert reward to class label
        reward_class = self.label_map[reward]

        # Format as dict observation expected by ENROS
        obs_dict = {
            "x": graph_data.x,
            "edge_index": graph_data.edge_index,
            "batch": torch.zeros(graph_data.num_nodes, dtype=torch.long),
        }

        return obs_dict, reward_class, agent_pos


def create_timestamped_log_dir(base_output_dir: str, script_name: str) -> str:
    """
    Create timestamped log directory for experiment.

    Args:
        base_output_dir: Base output directory specified by user
        script_name: Name of the training script (e.g., 'mot_graph')

    Returns:
        Path to created timestamped log directory
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path(base_output_dir) / f"{script_name}_{timestamp}"

    # Create directory structure
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "tensorboard").mkdir(exist_ok=True)
    (log_dir / "checkpoints").mkdir(exist_ok=True)

    return str(log_dir)


def setup_tensorboard(output_dir: str) -> SummaryWriter:
    """
    Set up TensorBoard logging with timestamped directory.

    Args:
        output_dir: Output directory for logs

    Returns:
        TensorBoard writer
    """
    tb_dir = Path(output_dir) / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(str(tb_dir))


def save_training_config(
    output_dir: str, args: Any, config_data: Optional[Dict] = None
) -> None:
    """
    Save training configuration and parameters to JSON file.

    Args:
        output_dir: Output directory for experiment
        args: Parsed command line arguments
        config_data: Optional additional config data to save
    """
    config_path = Path(output_dir) / "training_params.json"

    # Convert args to dictionary
    if hasattr(args, "__dict__"):
        params = vars(args)
    else:
        params = args

    # Add additional config data if provided
    if config_data:
        params["model_config"] = config_data

    # Add timestamp
    params["timestamp"] = datetime.now().isoformat()

    # Convert Path objects to strings for JSON serialization
    for key, value in params.items():
        if isinstance(value, Path):
            params[key] = str(value)

    with open(config_path, "w") as f:
        json.dump(params, f, indent=2, default=str)


def copy_config_file(config_path: str, output_dir: str) -> None:
    """
    Copy the original config file to the experiment directory.

    Args:
        config_path: Path to the original config file
        output_dir: Output directory for experiment
    """
    if os.path.exists(config_path):
        dest_path = Path(output_dir) / "config.yaml"
        shutil.copy2(config_path, dest_path)


def save_experiment_metadata(output_dir: str) -> None:
    """
    Save experiment metadata including git commit hash and environment info.

    Args:
        output_dir: Output directory for experiment
    """
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "python_version": f"{subprocess.sys.version_info.major}.{subprocess.sys.version_info.minor}.{subprocess.sys.version_info.micro}",
    }

    # Try to get git information
    try:
        # Get current commit hash
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        metadata["git_commit"] = commit_hash

        # Get branch name
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        metadata["git_branch"] = branch

        # Check if there are uncommitted changes
        try:
            subprocess.check_output(
                ["git", "diff-index", "--quiet", "HEAD", "--"],
                stderr=subprocess.DEVNULL,
            )
            metadata["git_clean"] = True
        except subprocess.CalledProcessError:
            metadata["git_clean"] = False

    except (subprocess.CalledProcessError, FileNotFoundError):
        metadata["git_info"] = "Git not available or not in git repository"

    # Save PyTorch and CUDA versions
    metadata["torch_version"] = torch.__version__
    metadata["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        metadata["cuda_version"] = torch.version.cuda
        metadata["gpu_count"] = torch.cuda.device_count()
        if torch.cuda.device_count() > 0:
            metadata["gpu_name"] = torch.cuda.get_device_name(0)

    metadata_path = Path(output_dir) / "experiment_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def setup_experiment_logging(
    base_output_dir: str, script_name: str, args: Any, config_path: str = None
) -> tuple[str, SummaryWriter]:
    """
    Set up complete experiment logging infrastructure.

    Args:
        base_output_dir: Base output directory specified by user
        script_name: Name of the training script
        args: Parsed command line arguments
        config_path: Optional path to config file

    Returns:
        Tuple of (log_directory_path, tensorboard_writer)
    """
    # Create timestamped log directory
    log_dir = create_timestamped_log_dir(base_output_dir, script_name)

    # Setup TensorBoard
    writer = setup_tensorboard(log_dir)

    # Save all experiment metadata
    save_training_config(log_dir, args)
    save_experiment_metadata(log_dir)

    if config_path and os.path.exists(config_path):
        copy_config_file(config_path, log_dir)

    print(f"Experiment log directory: {log_dir}")
    return log_dir, writer


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
    additional_metrics: Optional[Dict[str, float]] = None,
) -> None:
    """
    Save model checkpoint with enhanced metadata.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        val_loss: Validation loss
        val_accuracy: Validation accuracy
        output_dir: Output directory
        filename: Filename for checkpoint
        additional_metrics: Optional additional metrics to save
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "timestamp": datetime.now().isoformat(),
    }

    # Add additional metrics if provided
    if additional_metrics:
        checkpoint.update(additional_metrics)

    # Save to checkpoints subdirectory
    checkpoints_dir = Path(output_dir) / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoints_dir / filename

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def save_best_models(
    model: ENROSPolicy,
    optimizer,
    epoch: int,
    current_metrics: Dict[str, float],
    best_metrics: Dict[str, float],
    output_dir: str,
) -> Dict[str, float]:
    """
    Save best model based on validation loss and update best_metrics tracking.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        current_metrics: Current epoch metrics (must include 'val_loss')
        best_metrics: Dictionary tracking best metrics so far
        output_dir: Output directory

    Returns:
        Updated best_metrics dictionary
    """
    updated_best_metrics = best_metrics.copy()

    # Check if we have new best validation loss
    val_loss = current_metrics.get("val_loss")
    if val_loss is None:
        raise ValueError("current_metrics must include 'val_loss'")

    best_val_loss = best_metrics.get("best_val_loss", float("inf"))

    if val_loss < best_val_loss:
        updated_best_metrics["best_val_loss"] = val_loss
        updated_best_metrics["best_val_loss_epoch"] = epoch

        # Save best model checkpoint
        save_model_checkpoint(
            model,
            optimizer,
            epoch,
            current_metrics.get("val_loss", 0.0),
            current_metrics.get("val_accuracy", 0.0),
            output_dir,
            "best.pt",
            current_metrics,
        )

    # Always save latest checkpoint
    save_model_checkpoint(
        model,
        optimizer,
        epoch,
        current_metrics.get("val_loss", 0.0),
        current_metrics.get("val_accuracy", 0.0),
        output_dir,
        "latest.pt",
        current_metrics,
    )

    return updated_best_metrics


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
