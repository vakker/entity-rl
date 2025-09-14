"""
Training script for GNN-only ENROS using MOT ground truth data.

This script bypasses entity extraction and focuses on training the GNN component
using ground truth bounding boxes converted to graph representations.
"""

import argparse
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from mot_gnn_dataset import MOTGNNDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Batch
from tqdm import tqdm, trange

from entity_rl import utils
from entity_rl.models.enros import ENROSPolicy


def collate_graph_batch(batch):
    """
    Custom collate function for graph data.

    Args:
        batch: List of (graph_data, reward) tuples

    Returns:
        Tuple of (batched_graphs, reward_tensor)
    """
    graphs, rewards = zip(*batch)

    # Batch graphs using PyTorch Geometric's Batch
    batched_graphs = Batch.from_data_list(graphs)

    # Convert rewards to tensor
    reward_tensor = torch.tensor(rewards, dtype=torch.long)

    return batched_graphs, reward_tensor


class GNNDatasetAdapter:
    """Adapter to make GNN dataset compatible with ENROS expected input format."""

    def __init__(self, gnn_dataset):
        self.gnn_dataset = gnn_dataset
        self.label_map = {0: 0, 1: 1, -1: 2}

    def __len__(self):
        return len(self.gnn_dataset)

    def __getitem__(self, idx):
        graph_data, reward = self.gnn_dataset[idx]

        # Convert reward to class label
        reward_class = self.label_map[reward]

        # Format as dict observation expected by ENROS
        obs_dict = {
            'x': graph_data.x,
            'edge_index': graph_data.edge_index,
            'batch': torch.zeros(graph_data.num_nodes, dtype=torch.long)  # Single graph
        }

        return obs_dict, reward_class


def train_gnn_model(
    model: ENROSPolicy,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device = torch.device('cuda')
) -> None:
    """
    Main training loop for GNN-only ENROS model.
    """
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(args.output_dir)

    global_step = 0
    best_val_loss = float('inf')

    # Initial validation
    model.eval()
    with torch.no_grad():
        val_loss, val_acc = evaluate_model(model, val_loader, loss_fn, device)
        writer.add_scalar("val_loss", val_loss, global_step)
        writer.add_scalar("val_accuracy", val_acc, global_step)
        print(f"Initial validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

    # Training loop
    model.train()
    for epoch in trange(args.epochs, desc="Training epochs"):
        epoch_loss = 0
        num_batches = 0
        correct_predictions = 0
        total_predictions = 0

        for batch_idx, (obs_batch, reward_batch) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            # Move data to device
            obs_batch = move_obs_to_device(obs_batch, device)
            reward_batch = reward_batch.to(device)

            # Forward pass - ENROS expects dict observation
            _ = model({"obs": obs_batch})
            reward_pred = model.value_function()

            # Compute loss
            loss = loss_fn(reward_pred, reward_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            # Track metrics
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            # Calculate batch accuracy
            pred_classes = torch.argmax(reward_pred, dim=1)
            correct_predictions += (pred_classes == reward_batch).sum().item()
            total_predictions += reward_batch.size(0)

            # Log training loss
            if global_step % args.log_interval == 0:
                writer.add_scalar("train_loss", loss.item(), global_step)

        # Epoch statistics
        avg_epoch_loss = epoch_loss / num_batches
        epoch_accuracy = correct_predictions / total_predictions
        writer.add_scalar("train_loss_epoch", avg_epoch_loss, epoch)
        writer.add_scalar("train_accuracy_epoch", epoch_accuracy, epoch)

        print(f"Epoch {epoch+1} - Loss: {avg_epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss, val_acc = evaluate_model(model, val_loader, loss_fn, device)
            writer.add_scalar("val_loss", val_loss, global_step)
            writer.add_scalar("val_accuracy", val_acc, global_step)

            print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                }, os.path.join(args.output_dir, 'best_model.pt'))
                print(f"Saved new best model with val_loss: {val_loss:.4f}")

        model.train()

    writer.close()
    print("Training completed!")


def evaluate_model(model, data_loader, loss_fn, device):
    """Evaluate model on given data loader."""
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    num_batches = 0

    for obs_batch, reward_batch in data_loader:
        obs_batch = move_obs_to_device(obs_batch, device)
        reward_batch = reward_batch.to(device)

        _ = model({"obs": obs_batch})
        reward_pred = model.value_function()

        # Loss
        loss = loss_fn(reward_pred, reward_batch)
        total_loss += loss.item()
        num_batches += 1

        # Accuracy
        pred_classes = torch.argmax(reward_pred, dim=1)
        correct_predictions += (pred_classes == reward_batch).sum().item()
        total_predictions += reward_batch.size(0)

    avg_loss = total_loss / num_batches
    accuracy = correct_predictions / total_predictions

    return avg_loss, accuracy


def move_obs_to_device(obs_batch, device):
    """Move observation batch to device."""
    return {
        'x': obs_batch['x'].to(device),
        'edge_index': obs_batch['edge_index'].to(device),
        'batch': obs_batch['batch'].to(device)
    }


def create_graph_observation_space(node_feature_dim: int = 6):
    """Create observation space for graph data."""
    return gym.spaces.Dict({
        'x': gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(None, node_feature_dim), dtype=np.float32
        ),
        'edge_index': gym.spaces.Box(
            low=0, high=np.inf,
            shape=(2, None), dtype=np.int64
        ),
        'batch': gym.spaces.Box(
            low=0, high=np.inf,
            shape=(None,), dtype=np.int64
        )
    })


def main(args):
    """Main training function."""
    print(f"Using MOT directories: {args.mot_dirs}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max entities per sample: {args.max_entities}")

    # Create datasets
    train_dataset = MOTGNNDataset(
        mot_data_dirs=args.mot_dirs,
        agent_radius=args.agent_radius,
        num_samples_per_epoch=int(args.num_samples * 0.8),
        image_size=tuple(args.image_size),
        max_entities=args.max_entities,
        connect_threshold=args.connect_threshold,
        use_gt=not args.use_detections
    )

    val_dataset = MOTGNNDataset(
        mot_data_dirs=args.mot_dirs,
        agent_radius=args.agent_radius,
        num_samples_per_epoch=int(args.num_samples * 0.2),
        image_size=tuple(args.image_size),
        max_entities=args.max_entities,
        connect_threshold=args.connect_threshold,
        use_gt=not args.use_detections
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Wrap with adapter
    train_wrapped = GNNDatasetAdapter(train_dataset)
    val_wrapped = GNNDatasetAdapter(val_dataset)

    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_wrapped,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        collate_fn=lambda batch: collate_graph_batch(batch)
    )

    val_loader = DataLoader(
        val_wrapped,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=2,
        collate_fn=lambda batch: collate_graph_batch(batch)
    )

    # Set up model with graph observation space
    obs_space = create_graph_observation_space(node_feature_dim=6)
    action_space = gym.spaces.MultiDiscrete([3, 3])

    # Load config and modify for GNN-only training
    conf = utils.load_dict(args.cfg)["base"]

    # Ensure we're using EntityPassThrough + GNNEncoder
    model_config = conf["model"]["custom_model_config"]
    if "combined" in model_config:
        print("Warning: Config uses combined encoder, switching to entity+scene for GNN training")
        # Modify config to use EntityPassThrough + GNNEncoder
        model_config = {
            "encoder": {
                "entity": {"name": "EntityPassThrough"},
                "scene": {
                    "name": "GNNEncoder",
                    "config": {
                        "conv": {
                            "activation": "ELU",
                            "dims": [[6, 8], [8, 1]]
                        }
                    }
                }
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

    # Train the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_gnn_model(model, train_loader, val_loader, args, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GNN component of ENROS on MOT ground truth")

    # Data arguments
    parser.add_argument(
        "--mot-dirs", nargs="+", required=True, help="List of MOT data directories"
    )
    parser.add_argument(
        "--use-detections", action="store_true", help="Use detections instead of ground truth"
    )

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
    parser.add_argument(
        "--image-size", nargs=2, type=int, default=[100, 100], help="Image size as width height"
    )

    args = parser.parse_args()
    main(args)