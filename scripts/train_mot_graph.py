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
from torch_geometric.data import Batch, Data
from tqdm import tqdm, trange

from entity_rl import utils
from entity_rl.datasets import MOTGraphDataset
from entity_rl.datasets.graph_utils import (
    collate_graph_batch,
    create_graph_observation_space,
)
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
            "x": graph_data.x,
            "edge_index": graph_data.edge_index,
            "batch": torch.zeros(graph_data.num_nodes, dtype=torch.long),
        }

        return obs_dict, reward_class


def main(args):
    """Main training function."""
    print(f"Using MOT directories: {args.mot_dirs}")
    print(f"Output directory: {args.output_dir}")

    tng_dirs = args.mot_dirs[: len(args.mot_dirs) // 2]
    val_dirs = args.mot_dirs[len(args.mot_dirs) // 2 :]

    assert len(tng_dirs) > 0
    assert len(val_dirs) > 0

    # Create datasets
    train_dataset = MOTGraphDataset(
        mot_data_dirs=tng_dirs,
        agent_radius=args.agent_radius,
        num_samples_per_epoch=args.num_samples,
        image_size=tuple(args.image_size),
        max_entities=args.max_entities,
        connect_threshold=args.connect_threshold,
        use_gt=not args.use_detections,
        max_samples=args.max_samples,
    )

    val_dataset = MOTGraphDataset(
        mot_data_dirs=val_dirs,
        agent_radius=args.agent_radius,
        num_samples_per_epoch=args.num_samples,
        image_size=tuple(args.image_size),
        max_entities=args.max_entities,
        connect_threshold=args.connect_threshold,
        use_gt=not args.use_detections,
        max_samples=args.max_samples,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Wrap with adapter
    train_wrapped = GNNDatasetAdapter(train_dataset)
    val_wrapped = GNNDatasetAdapter(val_dataset)

    batch_size = min(args.batch_size, len(train_wrapped))

    # Create data loaders
    train_loader = DataLoader(
        train_wrapped,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        collate_fn=collate_graph_batch,
    )

    val_loader = DataLoader(
        val_wrapped,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.num_workers,
        collate_fn=collate_graph_batch,
    )
    assert len(train_loader)

    # Set up model with graph observation space
    obs_space = create_graph_observation_space(node_feature_dim=5)
    action_space = gym.spaces.MultiDiscrete([3, 3])

    # Load and modify config for GNN training
    conf = utils.load_dict(args.cfg)["base"]
    model_config = conf["model"]["custom_model_config"]

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

    for epoch in trange(args.epochs, desc="Training epochs", disable=args.no_bar):
        # Training
        model.train()
        tng_loss = 0
        num_batches = 0
        matches = 0

        for obs_batch, reward_batch in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1} TNG",
            leave=False,
            disable=args.no_bar,
        ):
            # Move to device
            # obs_batch = Batch(**obs_batch).to(device)
            # print(obs_batch['x'])
            # Concat x to reward_batch
            # __import__('ipdb').set_trace()
            # print(torch.cat([obs_batch["x"], reward_batch.unsqueeze(1)], dim=1))
            # torch.all((obs_batch["x"][0::2]<=obs_batch["x"][1::2]) == reward_batch.bool())
            # if not torch.all(
            #     (obs_batch["x"][0::2, 0] >= obs_batch["x"][1::2, 0])
            #     == reward_batch.bool()
            # ):
            #     __import__("ipdb").set_trace()

            # if not torch.all((obs_batch["x"].squeeze()<=0.5) == reward_batch.bool()):
            #     __import__('ipdb').set_trace()

            obs_batch = {k: v.to(device) for k, v in obs_batch.items()}
            # Count each individual reward
            # __import__('ipdb').set_trace()
            unique_values, counts = torch.unique(reward_batch, return_counts=True)
            tqdm.write(
                f"Reward targ stats: {unique_values}, {counts/len(reward_batch)}"
            )
            reward_batch = reward_batch.to(device)

            # Forward pass
            _ = model({"obs": obs_batch})
            reward_pred = model.value_function()
            # __import__('ipdb').set_trace()
            preds = torch.zeros_like(reward_batch)
            preds[reward_pred >= 0.5] = 1.0
            preds[reward_pred < 0.5] = 0.0
            match = (preds == reward_batch).float().mean().item()
            matches += match

            # unique_values, counts = torch.unique(reward_pred, return_counts=True)
            # tqdm.write(f"Reward pred stats: {unique_values}, {counts/len(reward_pred)}")
            # print("reward_batch", reward_batch)
            # print("reward_pred", reward_pred)

            # Backward pass
            loss = loss_fn(reward_pred, reward_batch)
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            # Logging
            tng_loss += loss.item()
            num_batches += 1
            global_step += 1

            # if global_step % args.log_interval == 0:
            #     writer.add_scalar("train_loss", loss.item(), global_step)

        avg_tng_loss = tng_loss / num_batches
        avg_tng_acc = matches / num_batches
        tqdm.write(f"Epoch {epoch+1} - TNG Loss: {avg_tng_loss:.4f}")
        tqdm.write(f"Epoch {epoch+1} - TNG Acc: {avg_tng_acc:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        num_batches = 0
        matches = 0
        for obs_batch, reward_batch in tqdm(
            val_loader,
            desc=f"Epoch {epoch+1} VAL",
            leave=False,
            disable=args.no_bar,
        ):
            # Move to device
            obs_batch = {k: v.to(device) for k, v in obs_batch.items()}
            reward_batch = reward_batch.to(device)

            # Forward pass
            _ = model({"obs": obs_batch})
            reward_pred = model.value_function()
            preds = torch.zeros_like(reward_batch)
            preds[reward_pred >= 0.5] = 1.0
            preds[reward_pred < 0.5] = 0.0
            match = (preds == reward_batch).float().mean().item()
            # print(reward_pred)

            # Backward pass
            loss = loss_fn(reward_pred, reward_batch)
            val_loss += loss.item()
            matches += match
            num_batches += 1

        avg_val_loss = val_loss / num_batches
        avg_val_acc = matches / num_batches
        tqdm.write(f"Epoch {epoch+1} - VAL Loss: {avg_val_loss:.4f}")
        tqdm.write(f"Epoch {epoch+1} - VAL Acc: {avg_val_acc:.4f}")

        # val_loss, val_acc = evaluate_model(model, val_loader, loss_fn, device)
        # writer.add_scalar("val_loss", val_loss, global_step)
        # writer.add_scalar("val_accuracy", val_acc, global_step)
        #
        # print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        #
        # # Save best model
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     save_model_checkpoint(
        #         model, optimizer, epoch, val_loss, val_acc, args.output_dir
        #     )

    writer.close()
    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train GNN component of ENROS on MOT ground truth"
    )

    # Data arguments
    parser.add_argument(
        "--mot-dirs", nargs="+", required=True, help="MOT data directories"
    )
    parser.add_argument(
        "--use-detections",
        action="store_true",
        help="Use detections instead of ground truth",
    )
    parser.add_argument("--no-bar", action="store_true")
    parser.add_argument("--max-samples", type=int, help="Max samples to load")

    # Model arguments
    parser.add_argument("--cfg", required=True, help="Config file path")

    # Training arguments
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--num-samples", type=int, default=2000, help="Samples per epoch"
    )
    parser.add_argument(
        "--grad-clip", type=float, default=1.0, help="Gradient clipping"
    )
    parser.add_argument("--log-interval", type=int, default=50, help="Logging interval")

    # Dataset arguments
    parser.add_argument("--agent-radius", type=float, default=0.02, help="Agent radius")
    parser.add_argument(
        "--max-entities", type=int, default=100, help="Max entities per sample"
    )
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument(
        "--connect-threshold",
        type=float,
        default=50.0,
        help="Edge connection threshold",
    )
    parser.add_argument(
        "--image-size", nargs=2, type=int, default=[500, 500], help="Image size"
    )

    main(parser.parse_args())
