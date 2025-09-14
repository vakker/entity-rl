#!/usr/bin/env python3
"""
Simplified training script for GNN-only ENROS using MOT ground truth data.

This script bypasses entity extraction and focuses on training the GNN component
using ground truth bounding boxes converted to graph representations.
"""

import argparse

import gymnasium as gym
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from entity_rl import utils
from entity_rl.datasets import MOTGNNDataset
from entity_rl.datasets.graph_utils import collate_graph_batch, create_graph_observation_space
from entity_rl.models.enros import ENROSPolicy
from entity_rl.training import (
    RewardLabelAdapter,
    create_loss_function,
    create_optimizer,
    evaluate_model,
    log_gradients,
    save_model_checkpoint,
    setup_tensorboard,
)


class GNNDatasetAdapter(RewardLabelAdapter):
    """Adapter for GNN dataset to provide correct observation format."""

    def __getitem__(self, index):
        graph_data, reward = self.base_dataset[index]

        # Convert reward to class label
        reward_class = self.label_map[reward]

        # Format as dict observation expected by ENROS
        obs_dict = {
            'x': graph_data.x,
            'edge_index': graph_data.edge_index,
            'batch': torch.zeros(graph_data.num_nodes, dtype=torch.long)
        }

        return obs_dict, reward_class


def main(args):
    """Main training function."""
    print(f"Using MOT directories: {args.mot_dirs}")
    print(f"Output directory: {args.output_dir}")

    # Create datasets
    train_dataset = MOTGNNDataset(
        mot_data_dirs=args.mot_dirs,
        agent_radius=args.agent_radius,
        num_samples_per_epoch=int(args.num_samples * 0.8),
        image_size=tuple(args.image_size),
        max_entities=args.max_entities,
        connect_threshold=args.connect_threshold,
        use_gt=not args.use_detections,
    )

    val_dataset = MOTGNNDataset(
        mot_data_dirs=args.mot_dirs,
        agent_radius=args.agent_radius,
        num_samples_per_epoch=int(args.num_samples * 0.2),
        image_size=tuple(args.image_size),
        max_entities=args.max_entities,
        connect_threshold=args.connect_threshold,
        use_gt=not args.use_detections,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Wrap with adapter
    train_wrapped = GNNDatasetAdapter(train_dataset)
    val_wrapped = GNNDatasetAdapter(val_dataset)

    # Create data loaders
    train_loader = DataLoader(
        train_wrapped,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        collate_fn=collate_graph_batch,
    )

    val_loader = DataLoader(
        val_wrapped,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=2,
        collate_fn=collate_graph_batch,
    )

    # Set up model with graph observation space
    obs_space = create_graph_observation_space(node_feature_dim=6)
    action_space = gym.spaces.MultiDiscrete([3, 3])

    # Load and modify config for GNN training
    conf = utils.load_dict(args.cfg)["base"]
    model_config = conf["model"]["custom_model_config"]

    # Ensure we're using EntityPassThrough + GNNEncoder
    if "combined" in model_config:
        print("Modifying config for GNN-only training")
        model_config = {
            "encoder": {
                "entity": {"name": "EntityPassThrough"},
                "scene": {
                    "name": "GNNEncoder",
                    "config": {"conv": {"activation": "ELU", "dims": [[6, 8], [8, 1]]}},
                },
            }
        }
        conf["model"]["custom_model_config"] = model_config

    model = ENROSPolicy(
        obs_space,
        action_space,
        num_outputs=6,
        model_config=conf["model"],
        name="enros_gnn",
    )

    print(f"Model created: {sum(p.numel() for p in model.parameters())} parameters")

    # Set up training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model.to(device)
    optimizer = create_optimizer(model, args.lr)
    loss_fn = create_loss_function(device)
    writer = setup_tensorboard(args.output_dir)

    # Training loop
    global_step = 0
    best_val_loss = float("inf")

    for epoch in trange(args.epochs, desc="Training epochs"):
        # Training
        model.train()
        epoch_loss = 0
        num_batches = 0

        for obs_batch, reward_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            # Move to device
            obs_batch = {k: v.to(device) for k, v in obs_batch.items()}
            reward_batch = reward_batch.to(device)

            # Forward pass
            _ = model({"obs": obs_batch})
            reward_pred = model.value_function()

            # Backward pass
            loss = loss_fn(reward_pred, reward_batch)
            optimizer.zero_grad()
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            # Logging
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            if global_step % args.log_interval == 0:
                writer.add_scalar("train_loss", loss.item(), global_step)

        # Validation
        val_loss, val_acc = evaluate_model(model, val_loader, loss_fn, device)
        writer.add_scalar("val_loss", val_loss, global_step)
        writer.add_scalar("val_accuracy", val_acc, global_step)

        print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model_checkpoint(
                model, optimizer, epoch, val_loss, val_acc, args.output_dir
            )

    writer.close()
    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GNN component of ENROS on MOT ground truth")

    # Data arguments
    parser.add_argument("--mot-dirs", nargs="+", required=True, help="MOT data directories")
    parser.add_argument("--use-detections", action="store_true", help="Use detections instead of ground truth")

    # Model arguments
    parser.add_argument("--cfg", required=True, help="Config file path")

    # Training arguments
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-samples", type=int, default=2000, help="Samples per epoch")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--log-interval", type=int, default=50, help="Logging interval")

    # Dataset arguments
    parser.add_argument("--agent-radius", type=int, default=15, help="Agent radius")
    parser.add_argument("--max-entities", type=int, default=20, help="Max entities per sample")
    parser.add_argument("--connect-threshold", type=float, default=50.0, help="Edge connection threshold")
    parser.add_argument("--image-size", nargs=2, type=int, default=[100, 100], help="Image size")

    main(parser.parse_args())