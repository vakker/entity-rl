import argparse
import os
import pickle
from os import path as osp

import gymnasium as gym
import numpy as np
import torch
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy
from skimage import io as skio
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm, trange

from entity_rl import utils
from entity_rl.models.enros import ENROSPolicy


class CustomDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        self.label_map = {0: 0, 1: 1, -1: 2}

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        x = torch.tensor(data["obs"], dtype=torch.uint8)
        y = self.label_map[np.sign(data["reward"])]
        return x, y


def add_bbox(frame, bbox_pred):
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
    images = []
    for obs, bbox_preds in zip(obs_batch, bbox_preds_batch):
        frames = []
        # obs = obs.reshape(100, 100, 3, 2)
        obs = [obs[:, :, 3 * i : 3 * (i + 1)] for i in range(2)]
        for frame, bbox_pred in zip(obs, bbox_preds):
            frame = add_bbox(frame, bbox_pred).numpy()
            frames.append(frame)

        img = np.concatenate(frames, 2)
        images.append(img)

    return images


def main(args):
    # Open the file in binary read mode
    with open(args.data, "rb") as file:
        # Use pickle.load() to load the object from the file
        dataset = pickle.load(file)

    if len(dataset) > 2000:
        dataset = np.random.choice(dataset, 2000)

    rewards = [d["reward"] for d in dataset]
    print("Data stats:")
    print(f"Total: {len(dataset)}")
    values, counts = np.unique(rewards, return_counts=True)
    for v, n in zip(values, counts):
        print(f"Reward {v}: {n}")

    height = dataset[0]["obs"].shape[0]
    width = dataset[0]["obs"].shape[1]
    stack = dataset[0]["obs"].shape[2] // 3
    obs_space = gym.spaces.Box(low=0, high=255, shape=(height, width, 3 * stack))
    action_space = gym.spaces.MultiDiscrete([3, 3])

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

    # epochs = 10
    # loss_fn = torch.nn.MSELoss().cuda()
    loss_fn = torch.nn.CrossEntropyLoss().cuda()

    batch_size = 64 * 2
    batch_size = 64
    dataset_tng, dataset_val = train_test_split(dataset, test_size=0.25)
    dataset_tng = CustomDataset(dataset_tng)
    dataset_val = CustomDataset(dataset_val)
    data_loader_tng = DataLoader(
        dataset_tng,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )

    # create output dir
    os.makedirs(args.output_dir, exist_ok=True)

    writer = SummaryWriter(args.output_dir)

    global_step = 0

    model.eval()
    with torch.no_grad():
        val_loss = 0
        total_samples = 0
        for obs, rew in tqdm(data_loader_val):
            obs_orig = obs
            obs = obs.cuda()
            rew = rew.cuda()
            _ = model({"obs": obs})
            rew_pred = model.value_function()
            # compare = list(zip(rew, rew_pred))
            # print(compare)
            val_loss += loss_fn(rew_pred, rew).item()
            # total_samples += obs.shape[0]
            total_samples += 1

        val_loss /= total_samples
        writer.add_scalar("val_loss", val_loss, global_step)

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
                np.stack(images).transpose(0, 3, 1, 2)[:, :3],
                global_step=global_step,
            )

        frames_dir = osp.join(args.output_dir, "obs_orig")
        os.makedirs(frames_dir, exist_ok=True)
        for j, img in enumerate(obs_orig[:5]):
            img_path = osp.join(frames_dir, f"f-{global_step:03d}-{j:06d}.png")
            skio.imsave(
                img_path,
                img[:, :, :3].numpy(),
                check_contrast=False,
            )

    model.train()

    for epoch in trange(args.epochs):
        for obs, rew in tqdm(data_loader_tng):
            obs = obs.cuda()
            rew = rew.cuda()
            _ = model({"obs": obs})
            rew_pred = model.value_function()
            loss = loss_fn(rew_pred, rew)
            # tqdm.write(str(loss.item()))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            # Log metrics to TensorBoard
            global_step += 1
            writer.add_scalar("tng_loss", loss.item(), global_step)

            if args.bbox:
                text_embed = model._encoder._stages[0]._model.text_embed.weight
                name = "text_embed"
                param = text_embed
                writer.add_histogram(
                    name + "_grad",
                    param.grad.data.cpu().numpy(),
                    global_step,
                )
            # for name, param in model.named_parameters():
            #     if not param.requires_grad or param.grad is None:
            #         continue

            #     writer.add_histogram(
            #         name + "_grad",
            #         param.grad.data.cpu().numpy(),
            #         global_step,
            #     )

            opt.zero_grad(set_to_none=True)

        model.eval()
        with torch.no_grad():
            val_loss = 0
            total_samples = 0
            for obs, rew in tqdm(data_loader_val):
                obs_orig = obs
                obs = obs.cuda()
                rew = rew.cuda()
                _ = model({"obs": obs})
                rew_pred = model.value_function()
                # compare = list(zip(rew, rew_pred))
                # print(compare)
                val_loss += loss_fn(rew_pred, rew).item()
                total_samples += 1
                # total_samples += obs.shape[0]

            val_loss /= total_samples
            writer.add_scalar("val_loss", val_loss, global_step)

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
                    np.stack(images).transpose(0, 3, 1, 2)[:, :3],
                    global_step=global_step,
                )

            frames_dir = osp.join(args.output_dir, "obs_orig")
            os.makedirs(frames_dir, exist_ok=True)
            for j, img in enumerate(obs_orig[:5]):
                img_path = osp.join(frames_dir, f"f-{global_step:03d}-{j:06d}.png")
                skio.imsave(
                    img_path,
                    img[:, :, :3].numpy(),
                    check_contrast=False,
                )

        model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg")
    parser.add_argument("--data")
    parser.add_argument("--output-dir")
    parser.add_argument("--bbox", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, required=True)
    args = parser.parse_args()
    main(args)
