import argparse

import gymnasium as gym
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from entity_rl import utils
from entity_rl.datasets import MOTDataset
from entity_rl.datasets.graph_utils import (
    collate_graph_batch,
    create_graph_observation_space,
)
from entity_rl.models.enros import ENROSPolicy
from entity_rl.training import (
    create_loss_function,
    create_optimizer,
    evaluate_detection_batch,
    extract_detection_data_from_mot_sample,
    save_best_models,
    setup_experiment_logging,
)
from entity_rl.utils import TicToc


def main(args):
    """Main training function."""
    print(f"Using MOT directories: {args.mot_dirs}")
    print(f"Output directory: {args.output_dir}")

    tng_dirs = args.mot_dirs[: len(args.mot_dirs) // 2]
    val_dirs = args.mot_dirs[len(args.mot_dirs) // 2 :]

    assert len(tng_dirs) > 0
    assert len(val_dirs) > 0

    # Create datasets
    train_dataset = MOTDataset(
        mot_data_dirs=tng_dirs,
        agent_radius=args.agent_radius,
        num_samples_per_epoch=args.num_samples,
        image_size=tuple(args.image_size),
        task_type=args.task_type,
        max_entities=args.max_entities,
        connect_threshold=args.connect_threshold,
        max_samples=args.max_samples,
        use_props=args.use_props,
        include_agent_node=args.include_agent_node,
    )

    val_dataset = MOTDataset(
        mot_data_dirs=val_dirs,
        agent_radius=args.agent_radius,
        num_samples_per_epoch=args.num_samples,
        image_size=tuple(args.image_size),
        task_type=args.task_type,
        max_entities=args.max_entities,
        connect_threshold=args.connect_threshold,
        max_samples=args.max_samples,
        use_props=args.use_props,
        include_agent_node=args.include_agent_node,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Task type: {args.task_type}")

    batch_size = min(args.batch_size, len(train_dataset))

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        collate_fn=collate_graph_batch,
    )

    val_loader = DataLoader(
        val_dataset,
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

    model = ENROSPolicy(
        obs_space,
        action_space,
        num_outputs=6,
        model_config=conf["model"],
        name="enros_gnn",
    )

    print(f"Model created: {sum(p.numel() for p in model.parameters())} parameters")

    # Set up training
    device = torch.device(args.device)
    print(f"Using device: {device}")

    model.to(device)
    optimizer = create_optimizer(model, args.lr)
    loss_fn = create_loss_function(device, args.task_type)

    # Setup experiment logging with timestamped directory
    output_dir, writer = setup_experiment_logging(
        args.output_dir, "mot_graph", args, args.cfg
    )

    # Training loop
    global_step = 0
    best_metrics = {}

    # Initialize timer (enabled/disabled based on --benchmark flag)
    timer = TicToc(enabled=args.benchmark)

    for epoch in trange(args.epochs, desc="Training epochs", disable=args.no_bar):
        timer.reset()

        # Training
        timer.tic("tng")
        model.train()
        timer.toc("tng")
        tng_loss = 0
        num_batches = 0
        matches = 0

        timer.tic("data_loading")

        for batch_data in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1} TNG",
            leave=False,
            disable=args.no_bar,
        ):
            assert (
                len(batch_data) == 3
            ), f"Expected 3 elements in batch_data, got {len(batch_data)}"
            obs_batch, reward_batch, agent_pos = batch_data

            timer.toc("data_loading")

            timer.tic("data_to_gpu")

            agent_pos = agent_pos.to(device)

            # Move to device
            obs_batch = {k: v.to(device) for k, v in obs_batch.items()}
            # Count each individual reward
            # unique_values, counts = torch.unique(reward_batch, return_counts=True)
            # tqdm.write(
            #     f"Reward targ stats: {unique_values}, {counts/len(reward_batch)}"
            # )
            reward_batch = reward_batch.to(device)

            timer.toc("data_to_gpu")
            timer.tic("forward_pass")

            # Forward pass
            model_input = {"obs": obs_batch, "agent_pos": agent_pos}
            _ = model(model_input)
            reward_pred = model.value_function()

            timer.toc("forward_pass")
            timer.tic("loss_compute")

            # Calculate accuracy using dataset method
            match = train_dataset.calculate_accuracy(reward_pred, reward_batch)
            matches += match

            # Backward pass
            loss = loss_fn(reward_pred, reward_batch)

            timer.toc("loss_compute")
            timer.tic("backward_pass")

            loss.backward()

            timer.toc("backward_pass")
            timer.tic("optimizer_step")

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            timer.toc("optimizer_step")

            # Logging
            tng_loss += loss.item()
            num_batches += 1
            global_step += 1

            # if global_step % args.log_interval == 0:
            #     writer.add_scalar("train_loss", loss.item(), global_step)

            # For the next data loading
            timer.tic("data_loading")

        avg_tng_loss = tng_loss / num_batches
        avg_tng_acc = matches / (num_batches * batch_size)

        # Log training metrics
        writer.add_scalar("loss/train", avg_tng_loss, epoch)
        writer.add_scalar("accuracy/train", avg_tng_acc, epoch)

        # Print timing statistics for training
        timer.print_stats(title=f"Epoch {epoch+1} Training Timing")

        if not epoch % args.val_int == 0:
            continue

        # Validation
        model.eval()
        val_loss = 0
        num_batches = 0
        matches = 0

        # For detection metrics
        pred_boxes_batch = []
        pred_scores_batch = []
        gt_boxes_batch = []

        timer.reset()

        with torch.no_grad():
            for batch_data in tqdm(
                val_loader,
                desc=f"Epoch {epoch+1} VAL",
                leave=False,
                disable=args.no_bar,
            ):
                timer.tic("val_data_loading")

                assert (
                    len(batch_data) == 3
                ), f"Expected 3 elements in batch_data, got {len(batch_data)}"
                obs_batch, reward_batch, agent_pos = batch_data

                timer.toc("val_data_loading")
                timer.tic("val_data_to_gpu")

                agent_pos = agent_pos.to(device)

                obs_batch = {k: v.to(device) for k, v in obs_batch.items()}
                reward_batch = reward_batch.to(device)

                timer.toc("val_data_to_gpu")
                timer.tic("val_forward_pass")

                # Forward pass
                model_input = {"obs": obs_batch, "agent_pos": agent_pos}
                _ = model(model_input)
                reward_pred = model.value_function()

                timer.toc("val_forward_pass")
                timer.tic("val_metrics")

                # Calculate accuracy using dataset method
                match = val_dataset.calculate_accuracy(reward_pred, reward_batch)
                matches += match

                loss = loss_fn(reward_pred, reward_batch)
                val_loss += loss.item()
                num_batches += 1

                timer.toc("val_metrics")

                # FIXME
                # Extract detection data for metrics (sample a few batches)
                # if (
                #     len(pred_boxes_batch) < 5
                # ):  # Only process first few batches for efficiency
                #     # Get graph data from the original dataset
                #     for i in range(min(batch_size, len(val_dataset))):
                #         sample_idx = (num_batches - 1) * batch_size + i
                #         if sample_idx < len(val_dataset):
                #             graph_data, sample_reward, sample_agent_pos = val_dataset[
                #                 sample_idx
                #             ]
                #
                #             # Extract detection data
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
                #                 # For this demo, use the same boxes as ground truth
                #                 # In practice, you'd load actual ground truth
                #                 gt_boxes_batch.append(boxes)

        # Calculate validation metrics
        avg_val_loss = val_loss / num_batches if num_batches > 0 else 0.0
        avg_val_acc = matches / (num_batches * batch_size) if num_batches > 0 else 0.0

        # Calculate detection metrics if we have data
        detection_metrics = {}
        if len(pred_boxes_batch) > 0:
            try:
                detection_metrics = evaluate_detection_batch(
                    pred_boxes_batch,
                    pred_scores_batch,
                    gt_boxes_batch,
                    iou_threshold=0.5,
                )
            except Exception as e:
                tqdm.write(f"Warning: Could not calculate detection metrics: {e}")

        # Log all metrics
        writer.add_scalar("loss/validation", avg_val_loss, epoch)
        writer.add_scalar("accuracy/validation", avg_val_acc, epoch)

        # Print timing statistics for validation
        timer.print_stats(title=f"Epoch {epoch+1} Validation Timing")

        if detection_metrics:
            writer.add_scalar(
                "detection/precision", detection_metrics.get("mean_precision", 0), epoch
            )
            writer.add_scalar(
                "detection/recall", detection_metrics.get("mean_recall", 0), epoch
            )
            writer.add_scalar(
                "detection/F1", detection_metrics.get("mean_f1_score", 0), epoch
            )
            writer.add_scalar(
                "detection/mAP", detection_metrics.get("mean_ap", 0), epoch
            )

        # Print epoch summary
        tqdm.write(
            f"Epoch {epoch+1} - TNG Loss: {avg_tng_loss:.4f}, TNG Acc: {avg_tng_acc:.4f}"
        )
        tqdm.write(
            f"Epoch {epoch+1} - VAL Loss: {avg_val_loss:.4f}, VAL Acc: {avg_val_acc:.4f}"
        )

        if detection_metrics:
            tqdm.write(
                f"Epoch {epoch+1} - Detection P/R/F1: {detection_metrics.get('mean_precision', 0):.3f}/{detection_metrics.get('mean_recall', 0):.3f}/{detection_metrics.get('mean_f1_score', 0):.3f}"
            )

        # Collect current metrics for model saving
        current_metrics = {
            "val_loss": avg_val_loss,
            "val_accuracy": avg_val_acc,
            "train_loss": avg_tng_loss,
            "train_accuracy": avg_tng_acc,
        }

        # Add detection metrics if available
        if detection_metrics:
            current_metrics.update(
                {
                    "val_precision": detection_metrics.get("mean_precision", 0),
                    "val_recall": detection_metrics.get("mean_recall", 0),
                    "val_f1_score": detection_metrics.get("mean_f1_score", 0),
                    "val_map": detection_metrics.get("mean_ap", 0),
                }
            )

        # Save best models based on different metrics
        best_metrics = save_best_models(
            model, optimizer, epoch, current_metrics, best_metrics, output_dir
        )

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
        "--use-props",
        action="store_true",
        help="Use proposals for graph creation (always uses GT for rewards)",
    )
    parser.add_argument("--no-bar", action="store_true")
    parser.add_argument("--max-samples", type=int, help="Max samples to load")

    # Model arguments
    parser.add_argument("--cfg", required=True, help="Config file path")
    parser.add_argument(
        "--task-type",
        type=str,
        default="regression",
        choices=["regression", "classification"],
        help="Task type: 'regression' or 'classification' (default: regression)",
    )

    # Training arguments
    parser.add_argument("--device", default="cuda:0", help="Device to train on")
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

    parser.add_argument("--val-int", type=int, default=10, help="Validation interval")
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
    parser.add_argument(
        "--include-agent-node",
        action="store_true",
        help="Include agent as a node in the graph (default: False)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Enable detailed timing benchmarks for each training component",
    )

    main(parser.parse_args())
