#!/usr/bin/env python3
"""
Simplified training script for ENROS using MOT synthetic data.

This script uses the MOT synthetic dataset to train ENROS models on agent-environment
interactions derived from Multiple Object Tracking data.
"""

import argparse
import os

import gymnasium as gym
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from entity_rl import utils
from entity_rl.datasets import MOTSyntheticDataset
from entity_rl.models.enros import ENROSPolicy
from entity_rl.training import (
    RewardLabelAdapter,
    create_loss_function,
    create_optimizer,
    evaluate_model,
    log_gradients,
    save_model_checkpoint,
    save_samples,
    setup_amp_scaler,
    setup_tensorboard,
)


def main(args):
    """Main training function."""
    print(f"Using MOT directories: {args.mot_dirs}")
    print(f"Output directory: {args.output_dir}")

    # Create datasets
    train_dataset = MOTSyntheticDataset(
        mot_data_dirs=args.mot_dirs,
        agent_radius=args.agent_radius,
        num_samples_per_epoch=int(args.num_samples * 0.8),
        image_size=tuple(args.image_size),
        use_gt=not args.use_detections,
    )

    val_dataset = MOTSyntheticDataset(
        mot_data_dirs=args.mot_dirs,
        agent_radius=args.agent_radius,
        num_samples_per_epoch=int(args.num_samples * 0.2),
        image_size=tuple(args.image_size),
        use_gt=not args.use_detections,
    )

    # Wrap with adapter
    train_wrapped = RewardLabelAdapter(train_dataset)
    val_wrapped = RewardLabelAdapter(val_dataset)

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

    # Set up training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model.to(device)
    optimizer = create_optimizer(model, args.lr)
    scaler = setup_amp_scaler(model)
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

        for obs, rew in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            obs = obs.to(device)
            rew = rew.to(device)

            _ = model({"obs": obs})
            rew_pred = model.value_function()
            loss = loss_fn(rew_pred, rew)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            writer.add_scalar("train_loss", loss.item(), global_step)

            if args.log_gradients and global_step % 100 == 0:
                log_gradients(writer, model, global_step)

            optimizer.zero_grad(set_to_none=True)

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

        # Save samples
        if args.save_samples and (epoch + 1) % 5 == 0:
            sample_batch, _ = next(iter(val_loader))
            save_samples(writer, sample_batch, model, args.output_dir, global_step, args.bbox)

    writer.close()
    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ENROS on MOT synthetic data")

    # Data arguments
    parser.add_argument("--mot-dirs", nargs="+", required=True, help="MOT data directories")
    parser.add_argument("--use-detections", action="store_true", help="Use detections instead of ground truth")

    # Model arguments
    parser.add_argument("--cfg", required=True, help="Config file path")

    # Training arguments
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--num-samples", type=int, default=2000, help="Samples per epoch")

    # Dataset arguments
    parser.add_argument("--agent-radius", type=int, default=15, help="Agent radius")
    parser.add_argument("--image-size", nargs=2, type=int, default=[100, 100], help="Image size")

    # Visualization arguments
    parser.add_argument("--bbox", action="store_true", help="Enable bbox visualization")
    parser.add_argument("--save-samples", action="store_true", help="Save sample images")
    parser.add_argument("--log-gradients", action="store_true", help="Log gradients")

    main(parser.parse_args())