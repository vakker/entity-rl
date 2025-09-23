import argparse
import os
from os import path as osp

import gymnasium as gym
import numpy as np
import torch
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy
from skimage import io as skio
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm, trange

from entity_rl import utils
from entity_rl.datasets import MOTVisDataset
from entity_rl.models.enros import ENROSPolicy
from entity_rl.training import RewardLabelAdapter


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

    # Split MOT directories for train/val
    tng_dirs = args.mot_dirs[: len(args.mot_dirs) // 2]
    val_dirs = args.mot_dirs[len(args.mot_dirs) // 2 :]

    assert len(tng_dirs) > 0, "Need at least one training directory"
    assert len(val_dirs) > 0, "Need at least one validation directory"

    # Create MOT datasets
    train_dataset = MOTVisDataset(
        mot_data_dirs=tng_dirs,
        agent_radius=args.agent_radius,
        num_samples_per_epoch=args.num_samples,
        image_size=tuple(args.image_size),
        use_gt=not args.use_detections,
    )

    val_dataset = MOTVisDataset(
        mot_data_dirs=val_dirs,
        agent_radius=args.agent_radius,
        num_samples_per_epoch=args.num_samples,
        image_size=tuple(args.image_size),
        use_gt=not args.use_detections,
    )

    # Wrap with reward label adapter
    train_wrapped = RewardLabelAdapter(train_dataset)
    val_wrapped = RewardLabelAdapter(val_dataset)

    print(f"Train samples: {len(train_wrapped)}, Val samples: {len(val_wrapped)}")

    # Get observation space from dataset
    sample_obs, _ = train_wrapped[0]
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
    model.cuda()

    use_amp = model.use_amp
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    loss_fn = torch.nn.CrossEntropyLoss().cuda()

    batch_size = min(args.batch_size, len(train_wrapped))

    # Create data loaders
    data_loader_tng = DataLoader(
        train_wrapped,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )
    data_loader_val = DataLoader(
        val_wrapped,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.num_workers,
    )

    # Create output directory and tensorboard writer
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(args.output_dir)

    global_step = 0

    # Initial validation run
    print("Running initial validation...")
    model.eval()
    with torch.no_grad():
        val_loss = 0
        num_batches = 0
        matches = 0
        for obs, rew in tqdm(data_loader_val, desc="Initial validation"):
            obs_orig = obs
            obs = obs.cuda()
            rew = rew.cuda()
            _ = model({"obs": obs})
            rew_pred = model.value_function()

            # Calculate accuracy
            preds = torch.zeros_like(rew)
            preds[rew_pred >= 0.5] = 1.0
            preds[rew_pred < 0.5] = 0.0
            match = (preds == rew).float().mean().item()
            matches += match

            loss = loss_fn(rew_pred, rew)
            val_loss += loss.item()
            num_batches += 1

        avg_val_loss = val_loss / num_batches
        avg_val_acc = matches / num_batches
        writer.add_scalar("val_loss", avg_val_loss, global_step)
        writer.add_scalar("val_accuracy", avg_val_acc, global_step)
        print(f"Initial validation - Loss: {avg_val_loss:.4f}, Acc: {avg_val_acc:.4f}")

        # Save bbox visualizations if requested
        if args.bbox:
            bbox_preds = model._encoder._stages[0].gdino_outputs["bboxes"]
            images = process_outputs(obs_orig[:5], bbox_preds[:5])

            frames_dir = osp.join(args.output_dir, "bboxes")
            os.makedirs(frames_dir, exist_ok=True)
            for j, img in enumerate(images):
                img_path = osp.join(frames_dir, f"f-{global_step:03d}-{j:06d}.png")
                skio.imsave(img_path, img[:, :, :3], check_contrast=False)

            writer.add_images(
                "bboxes",
                torch.stack([torch.from_numpy(img) for img in images]).permute(
                    0, 3, 1, 2
                )[:, :3],
                global_step=global_step,
            )

        # Save original observations
        frames_dir = osp.join(args.output_dir, "obs_orig")
        os.makedirs(frames_dir, exist_ok=True)
        for j, img in enumerate(obs_orig[:5]):
            img_path = osp.join(frames_dir, f"f-{global_step:03d}-{j:06d}.png")
            skio.imsave(
                img_path,
                img[:, :, :3].numpy(),
                check_contrast=False,
            )

    # Training loop
    model.train()
    print("Starting training...")

    for epoch in trange(args.epochs, desc="Training epochs"):
        for obs, rew in tqdm(data_loader_tng, desc=f"Epoch {epoch+1}"):
            obs = obs.cuda()
            rew = rew.cuda()

            _ = model({"obs": obs})
            rew_pred = model.value_function()
            loss = loss_fn(rew_pred, rew)

            scaler.scale(loss).backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(opt)
            scaler.update()

            # Log training loss
            global_step += 1
            writer.add_scalar("tng_loss", loss.item(), global_step)

            # Log text embedding gradients if bbox visualization is enabled
            if args.bbox:
                text_embed = model._encoder._stages[0]._model.text_embed.weight
                if text_embed.grad is not None:
                    writer.add_histogram(
                        "text_embed_grad",
                        text_embed.grad.data.cpu().numpy(),
                        global_step,
                    )

            opt.zero_grad(set_to_none=True)

        # Validation after each epoch
        model.eval()
        with torch.no_grad():
            val_loss = 0
            num_batches = 0
            matches = 0
            for obs, rew in tqdm(
                data_loader_val, desc=f"Epoch {epoch+1} VAL", leave=False
            ):
                obs_orig = obs
                obs = obs.cuda()
                rew = rew.cuda()
                _ = model({"obs": obs})
                rew_pred = model.value_function()

                # Calculate accuracy
                preds = torch.zeros_like(rew)
                preds[rew_pred >= 0.5] = 1.0
                preds[rew_pred < 0.5] = 0.0
                match = (preds == rew).float().mean().item()
                matches += match

                loss = loss_fn(rew_pred, rew)
                val_loss += loss.item()
                num_batches += 1

            avg_val_loss = val_loss / num_batches
            avg_val_acc = matches / num_batches
            writer.add_scalar("val_loss", avg_val_loss, global_step)
            writer.add_scalar("val_accuracy", avg_val_acc, global_step)
            print(
                f"Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}"
            )

            # Save bbox visualizations
            if args.bbox:
                bbox_preds = model._encoder._stages[0].gdino_outputs["bboxes"]
                images = process_outputs(obs_orig[:5], bbox_preds[:5])

                frames_dir = osp.join(args.output_dir, "bboxes")
                for j, img in enumerate(images):
                    img_path = osp.join(frames_dir, f"f-{global_step:03d}-{j:06d}.png")
                    skio.imsave(img_path, img[:, :, :3], check_contrast=False)

                writer.add_images(
                    "bboxes",
                    torch.stack([torch.from_numpy(img) for img in images]).permute(
                        0, 3, 1, 2
                    )[:, :3],
                    global_step=global_step,
                )

            # Save original observations
            frames_dir = osp.join(args.output_dir, "obs_orig")
            for j, img in enumerate(obs_orig[:5]):
                img_path = osp.join(frames_dir, f"f-{global_step:03d}-{j:06d}.png")
                skio.imsave(
                    img_path,
                    img[:, :, :3].numpy(),
                    check_contrast=False,
                )

        model.train()

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

    # Model arguments
    parser.add_argument("--cfg", required=True, help="Config file path")

    # Training arguments
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

    # Dataset arguments
    parser.add_argument("--agent-radius", type=float, default=15, help="Agent radius")
    parser.add_argument(
        "--image-size", nargs=2, type=int, default=[100, 100], help="Image size"
    )
    parser.add_argument(
        "--num-workers", type=int, default=2, help="Number of workers for data loading"
    )

    # Visualization arguments
    parser.add_argument("--bbox", action="store_true", help="Enable bbox visualization")

    main(parser.parse_args())
