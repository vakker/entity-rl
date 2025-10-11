"""
Evaluate trained ENROS model on MOT graph data.

This script loads a trained checkpoint and runs evaluation on MOT data,
computing metrics like accuracy, loss, precision, recall, F1, and mAP.

Usage:
    python evaluate_mot_graph.py --checkpoint_dir experiments/mot_graph_20250101 \
                                 --mot_dirs data/MOT16/train/MOT16-02 data/MOT16/train/MOT16-04 \
                                 --batch_size 32 \
                                 --device cuda

Author: Generated for entity-rl project
"""

import argparse
from pathlib import Path

import gymnasium as gym
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from entity_rl import config as cfg_utils
from entity_rl import utils
from entity_rl.config import MOTGraphEvaluationConfig
from entity_rl.datasets import MOTDataset
from entity_rl.datasets.graph_utils import (
    collate_graph_batch,
    create_graph_observation_space,
)
from entity_rl.models.enros import ENROSPolicy
from entity_rl.training import (
    create_loss_function,
    evaluate_detection_batch,
    extract_detection_data_from_mot_sample,
)
from entity_rl.utils import TicToc


def find_config(checkpoint_dir: Path) -> Path:
    """Find config file in checkpoint directory."""
    for config_name in ["config.yaml", "conf.yaml"]:
        config_path = checkpoint_dir / config_name
        if config_path.exists():
            print(f"Found config: {config_path}")
            return config_path

    raise FileNotFoundError(
        f"No config file (config.yaml or conf.yaml) found in {checkpoint_dir}"
    )


def find_checkpoint(checkpoint_dir: Path) -> Path:
    """Find best model checkpoint in directory."""
    # Look for best metric checkpoints
    metric_priority = [
        # "best.pt",
        "latest.pt",
    ]

    # Check in checkpoints subdirectory first
    checkpoints_subdir = checkpoint_dir / "checkpoints"
    if checkpoints_subdir.exists():
        for ckpt_name in metric_priority:
            ckpt_path = checkpoints_subdir / ckpt_name
            if ckpt_path.exists():
                print(f"Found checkpoint: {ckpt_path}")
                return ckpt_path

    # Check root directory
    for ckpt_name in metric_priority:
        ckpt_path = checkpoint_dir / ckpt_name
        if ckpt_path.exists():
            print(f"Found checkpoint: {ckpt_path}")
            return ckpt_path

    raise FileNotFoundError(f"No checkpoint file (.pt) found in {checkpoint_dir}")


def load_checkpoint(
    checkpoint_path: Path,
    config_path: Path,
    device: torch.device,
    task_type: str = "regression",
):
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        config_path: Path to config file
        device: Device to load model on
        task_type: Task type ('regression' or 'classification')

    Returns:
        Loaded model in eval mode
    """
    print(f"Loading checkpoint from: {checkpoint_path}")

    # Load config
    conf = utils.load_dict(str(config_path))["base"]

    # Create observation space
    obs_space = create_graph_observation_space(node_feature_dim=5)
    action_space = gym.spaces.MultiDiscrete([3, 3])

    # Create model
    model = ENROSPolicy(
        obs_space,
        action_space,
        num_outputs=6,
        model_config=conf["model"],
        name="enros_gnn_eval",
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Model loaded: {sum(p.numel() for p in model.parameters())} parameters")

    return model


def evaluate(cfg):
    print(f"Evaluating on MOT directories: {cfg.mot_dirs}")
    print(f"Checkpoint directory: {cfg.checkpoint_dir}")

    checkpoint_dir = Path(cfg.checkpoint_dir)

    # Auto-detect config and checkpoint
    config_path = find_config(checkpoint_dir)
    checkpoint_path = find_checkpoint(checkpoint_dir)

    # Set up device
    device = torch.device(cfg.device)
    print(f"Using device: {device}")

    # Load model
    model = load_checkpoint(checkpoint_path, config_path, device, cfg.task_type)

    # Create dataset
    print("Creating evaluation dataset...")
    eval_dataset = MOTDataset(
        mot_data_dirs=cfg.mot_dirs,
        agent_radius=cfg.agent_radius,
        num_samples_per_epoch=cfg.num_samples,
        image_size=tuple(cfg.image_size),
        task_type=cfg.task_type,
        max_entities=cfg.max_entities,
        connect_threshold=cfg.connect_threshold,
        max_samples=cfg.max_samples,
        visible_ann_filename=cfg.ann_filename,
        include_agent_node=cfg.include_agent_node,
        use_precomputed_features=cfg.use_precomputed_features,
        feature_filename=cfg.feature_filename,
    )

    print(f"Evaluation samples: {len(eval_dataset)}")
    print(f"Task type: {cfg.task_type}")

    batch_size = min(cfg.batch_size, len(eval_dataset))

    # Create data loader
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_graph_batch,
    )

    # Set up loss
    loss_fn = create_loss_function(device, cfg.task_type)

    # Initialize metrics
    total_loss = 0
    total_matches = 0
    num_batches = 0
    num_samples = 0

    # For detection metrics
    pred_boxes_batch = []
    pred_scores_batch = []
    gt_boxes_batch = []

    # Initialize timer
    timer = TicToc(enabled=cfg.benchmark)

    print("\nRunning evaluation...")

    # Evaluation loop
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(
            tqdm(eval_loader, desc="Evaluating", disable=cfg.no_bar)
        ):
            timer.tic("data_to_gpu")

            assert len(batch_data) == 3
            obs_batch, reward_batch, agent_pos = batch_data

            agent_pos = agent_pos.to(device)
            obs_batch = {k: v.to(device) for k, v in obs_batch.items()}
            reward_batch = reward_batch.to(device)

            timer.toc("data_to_gpu")
            timer.tic("forward_pass")

            # Forward pass
            model_input = {"obs": obs_batch, "agent_pos": agent_pos}
            _ = model(model_input)
            reward_pred = model.value_function()
            # print(reward_pred)
            # reward_pred[:] = 1.0

            timer.toc("forward_pass")
            timer.tic("metrics")

            # Calculate accuracy using dataset method
            match = eval_dataset.calculate_accuracy(reward_pred, reward_batch)

            loss = loss_fn(reward_pred, reward_batch)
            total_loss += loss.item()
            total_matches += match
            # print(match)
            num_batches += 1
            num_samples += len(reward_batch)

            timer.toc("metrics")

            # FIXME:
            # Extract detection data for metrics (sample batches)
            # if (
            #     args.compute_detection_metrics
            #     and len(pred_boxes_batch) < args.max_detection_batches
            # ):
            #     for i in range(min(batch_size, len(eval_dataset))):
            #         sample_idx = batch_idx * batch_size + i
            #         if sample_idx < len(eval_dataset):
            #             data_dict, sample_reward = eval_dataset[sample_idx]
            #             graph_data = data_dict["graph"]
            #             sample_agent_pos = data_dict["agent_pos"]
            #
            #             boxes, scores = extract_detection_data_from_mot_sample(
            #                 graph_data,
            #                 sample_reward,
            #                 sample_agent_pos,
            #                 tuple(args.image_size),
            #             )
            #
            #             if len(boxes) > 0:
            #                 pred_boxes_batch.append(boxes)
            #                 pred_scores_batch.append(scores)
            #                 gt_boxes_batch.append(boxes)

    # Calculate final metrics
    avg_loss = total_loss / num_batches
    avg_accuracy = total_matches / num_samples

    # Calculate detection metrics if we have data
    detection_metrics = {}
    if cfg.compute_detection_metrics and len(pred_boxes_batch) > 0:
        try:
            detection_metrics = evaluate_detection_batch(
                pred_boxes_batch,
                pred_scores_batch,
                gt_boxes_batch,
                iou_threshold=cfg.iou_threshold,
            )
        except Exception as e:
            print(f"Warning: Could not calculate detection metrics: {e}")

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total samples evaluated: {num_samples}")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Average Accuracy: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")

    if detection_metrics:
        print("\nDetection Metrics:")
        print(f"  Precision: {detection_metrics.get('mean_precision', 0):.4f}")
        print(f"  Recall:    {detection_metrics.get('mean_recall', 0):.4f}")
        print(f"  F1 Score:  {detection_metrics.get('mean_f1_score', 0):.4f}")
        print(f"  mAP:       {detection_metrics.get('mean_ap', 0):.4f}")

    print("=" * 60)

    # Print timing if benchmark enabled
    if cfg.benchmark:
        timer.print_stats(title="Evaluation Timing")

    # Save results if output path provided
    if cfg.output:
        output_path = Path(cfg.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results = {
            "checkpoint_dir": str(checkpoint_dir),
            "num_samples": num_samples,
            "loss": avg_loss,
            "accuracy": avg_accuracy,
        }

        if detection_metrics:
            results["detection_metrics"] = detection_metrics

        import json

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained ENROS model on MOT graph data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python scripts/evaluate_mot_graph.py --checkpoint-dir experiments/mot_001 \\
      --mot-dirs data/MOT16/train/MOT16-02 data/MOT16/train/MOT16-04

  # With CLI overrides
  python scripts/evaluate_mot_graph.py --checkpoint-dir experiments/mot_001 \\
      --mot-dirs data/MOT16/train/MOT16-02 \\
      batch_size=64 device=cuda:1

  # Compute detection metrics and save results
  python scripts/evaluate_mot_graph.py --checkpoint-dir experiments/mot_001 \\
      --mot-dirs data/MOT16/train/MOT16-02 \\
      compute_detection_metrics=true output=results.json
        """,
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Path to experiment directory (will auto-detect config and best checkpoint)",
    )
    parser.add_argument(
        "--mot-dirs",
        nargs="+",
        required=True,
        help="MOT data directories to evaluate on",
    )

    # Optional overrides
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides in dotlist format (e.g., batch_size=64 device=cuda:1)",
    )

    args = parser.parse_args()

    # Load structured config with type safety and validation
    # Required fields (checkpoint_dir, mot_dirs) are set from CLI args
    try:
        # Initialize overrides list
        if args.overrides is None:
            args.overrides = []

        # Add required CLI arguments to overrides
        args.overrides.append(f"checkpoint_dir={args.checkpoint_dir}")
        args.overrides.append(f"mot_dirs={args.mot_dirs}")

        cfg = cfg_utils.load_structured_config(
            MOTGraphEvaluationConfig,
            config_path=None,  # No config file for evaluation
            overrides=args.overrides,
        )
    except Exception as e:
        print(f"\n❌ Configuration Error: {e}\n")
        print("Required parameters:")
        print("  - checkpoint_dir: Path to experiment directory")
        print("  - mot_dirs: List of MOT data directories")
        raise

    # Print config
    cfg_utils.print_config(cfg, "MOT Graph Evaluation Configuration")

    evaluate(cfg)
