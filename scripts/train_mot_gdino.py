import argparse

import gymnasium as gym
import numpy as np
import torch
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy
from torch.utils.data import DataLoader
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm, trange

from entity_rl import utils
from entity_rl.datasets import MOTVisDataset
from entity_rl.models.enros import ENROSPolicy
from entity_rl.training import (
    RewardLabelAdapter,
    create_loss_function,
    create_optimizer,
    setup_tensorboard,
)


def add_bbox(frame, bbox_pred):
    """Add bounding boxes to frame for visualization."""
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
    """Process model outputs to create bbox visualization images."""
    images = []
    for obs, bbox_preds in zip(obs_batch, bbox_preds_batch):
        frames = []
        # Handle frame stacking - split channels into individual frames
        obs = [obs[:, :, 3 * i : 3 * (i + 1)] for i in range(obs.shape[2] // 3)]
        for frame, bbox_pred in zip(obs, bbox_preds):
            frame = add_bbox(frame, bbox_pred).numpy()
            frames.append(frame)

        img = torch.cat([torch.from_numpy(f) for f in frames], dim=2).numpy()
        images.append(img)

    return images


def main(args):
    """Main training function."""
    print(f"Using MOT directories: {args.mot_dirs}")
    print(f"Output directory: {args.output_dir}")

    tng_dirs = args.mot_dirs[: len(args.mot_dirs) // 2]
    val_dirs = args.mot_dirs[len(args.mot_dirs) // 2 :]

    assert len(tng_dirs) > 0
    assert len(val_dirs) > 0

    # Create MOT datasets
    train_dataset = MOTVisDataset(
        mot_data_dirs=tng_dirs,
        agent_radius=args.agent_radius,
        num_samples_per_epoch=args.num_samples,
        image_size=tuple(args.image_size),
        use_gt=not args.use_detections,
        max_samples=args.max_samples,
    )

    val_dataset = MOTVisDataset(
        mot_data_dirs=val_dirs,
        agent_radius=args.agent_radius,
        num_samples_per_epoch=args.num_samples,
        image_size=tuple(args.image_size),
        use_gt=not args.use_detections,
        max_samples=args.max_samples,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Wrap with reward label adapter
    train_wrapped = RewardLabelAdapter(train_dataset)
    val_wrapped = RewardLabelAdapter(val_dataset)

    batch_size = min(args.batch_size, len(train_wrapped))

    # Create data loaders
    train_loader = DataLoader(
        train_wrapped,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        val_wrapped,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.num_workers,
    )
    assert len(train_loader)

    # Get observation space from dataset
    sample_obs = train_wrapped[0][0]
    obs_space = gym.spaces.Box(
        low=0,
        high=255,
        shape=sample_obs.shape,
        dtype=np.uint8,
    )
    action_space = gym.spaces.MultiDiscrete([3, 3])

    # Load model config and create ENROS model
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
    device = torch.device(args.device)
    print(f"Using device: {device}")

    model.to(device)
    # use_amp = model.use_amp
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    # loss_fn = torch.nn.CrossEntropyLoss().cuda()

    optimizer = create_optimizer(model, args.lr)
    loss_fn = create_loss_function(device)
    writer = setup_tensorboard(args.output_dir)

    global_step = 0

    for epoch in trange(args.epochs, desc="Training epochs", disable=args.no_bar):
        # Training loop
        model.train()
        tng_loss = 0
        num_batches = 0
        matches = 0

        for batch_data in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}",
            disable=args.no_bar,
        ):
            # Unpack batch data - expect 4 elements (obs, reward, agent_x, agent_y)
            assert (
                len(batch_data) == 3
            ), f"Expected 3 elements in batch_data, got {len(batch_data)}"
            obs_batch, reward_batch, agent_pos = batch_data
            agent_pos = agent_pos.to(device)

            # Move to device
            obs_batch = obs_batch.to(device)
            # Count each individual reward
            unique_values, counts = torch.unique(reward_batch, return_counts=True)
            # tqdm.write(
            #     f"Reward targ stats: {unique_values}, {counts/len(reward_batch)}"
            # )
            # continue
            # It's double, but it should be float
            reward_batch = reward_batch.float().to(device)

            # Forward pass
            model_input = {"obs": obs_batch, "agent_pos": agent_pos}
            _ = model(model_input)
            reward_pred = model.value_function()
            # print(reward_pred)

            # Get matches
            preds = torch.zeros_like(reward_batch)
            preds[reward_pred >= 0.5] = 1.0
            preds[reward_pred < 0.5] = 0.0
            match = (preds == reward_batch).float().mean().item()
            matches += match

            # Backward pass
            loss = loss_fn(reward_pred, reward_batch)
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Log training loss
            tng_loss += loss.item()
            num_batches += 1
            global_step += 1

            # if global_step % args.log_interval == 0:
            #     writer.add_scalar("train_loss", loss.item(), global_step)

            # Log text embedding gradients if bbox visualization is enabled
            # if args.bbox:
            #     text_embed = model._encoder._stages[0]._model.text_embed.weight
            #     if text_embed.grad is not None:
            #         writer.add_histogram(
            #             "text_embed_grad",
            #             text_embed.grad.data.cpu().numpy(),
            #             global_step,
            #         )
            #

        avg_tng_loss = tng_loss / num_batches
        avg_tng_acc = matches / num_batches
        tqdm.write(f"Epoch {epoch+1} - TNG Loss: {avg_tng_loss:.4f}")
        tqdm.write(f"Epoch {epoch+1} - TNG Acc: {avg_tng_acc:.4f}")
        continue

        # Validation after each epoch
        model.eval()
        val_loss = 0
        num_batches = 0
        matches = 0
        with torch.no_grad():
            for batch_data in tqdm(
                val_loader,
                desc=f"Epoch {epoch+1} VAL",
                leave=False,
                disable=args.no_bar,
            ):
                # Unpack batch data - expect 4 elements (obs, reward, agent_x, agent_y)
                assert (
                    len(batch_data) == 3
                ), f"Expected 3 elements in batch_data, got {len(batch_data)}"
                obs_batch, reward_batch, agent_pos = batch_data
                agent_pos = agent_pos.to(device)

                obs_batch = obs_batch.to(device)
                reward_batch = reward_batch.to(device)

                model_input = {"obs": obs_batch, "agent_pos": agent_pos}
                _ = model(model_input)
                reward_pred = model.value_function()

                # Calculate accuracy
                preds = torch.zeros_like(reward_batch)
                preds[reward_pred >= 0.5] = 1.0
                preds[reward_pred < 0.5] = 0.0
                match = (preds == reward_batch).float().mean().item()
                matches += match

                loss = loss_fn(reward_pred, reward_batch)
                val_loss += loss.item()
                num_batches += 1

            avg_val_loss = val_loss / num_batches
            avg_val_acc = matches / num_batches
            # writer.add_scalar("val_loss", avg_val_loss, global_step)
            # writer.add_scalar("val_accuracy", avg_val_acc, global_step)

            avg_val_loss = val_loss / num_batches
            avg_val_acc = matches / num_batches
            tqdm.write(f"Epoch {epoch+1} - VAL Loss: {avg_val_loss:.4f}")
            tqdm.write(f"Epoch {epoch+1} - VAL Acc: {avg_val_acc:.4f}")

            # Save bbox visualizations
            # if args.bbox:
            #     obs_orig = obs_batch
            #     bbox_preds = model._encoder._stages[0].gdino_outputs["bboxes"]
            #     images = process_outputs(obs_orig[:5], bbox_preds[:5])
            #
            #     frames_dir = osp.join(args.output_dir, "bboxes")
            #     for j, img in enumerate(images):
            #         img_path = osp.join(frames_dir, f"f-{global_step:03d}-{j:06d}.png")
            #         skio.imsave(img_path, img[:, :, :3], check_contrast=False)
            #
            #     writer.add_images(
            #         "bboxes",
            #         torch.stack([torch.from_numpy(img) for img in images]).permute(
            #             0, 3, 1, 2
            #         )[:, :3],
            #         global_step=global_step,
            #     )
            #
            # # Save original observations
            # frames_dir = osp.join(args.output_dir, "obs_orig")
            # for j, img in enumerate(obs_orig[:5]):
            #     img_path = osp.join(frames_dir, f"f-{global_step:03d}-{j:06d}.png")
            #     skio.imsave(
            #         img_path,
            #         img[:, :, :3].numpy(),
            #         check_contrast=False,
            #     )

    writer.close()
    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ENROS with GDino on MOT synthetic data"
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
    parser.add_argument("--device", default="cuda:0", help="Device to train on")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
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
        "--image-size", nargs=2, type=int, default=[500, 500], help="Image size"
    )
    parser.add_argument(
        "--num-workers", type=int, default=2, help="Number of workers for data loading"
    )

    # Visualization arguments
    parser.add_argument("--bbox", action="store_true", help="Enable bbox visualization")

    main(parser.parse_args())
