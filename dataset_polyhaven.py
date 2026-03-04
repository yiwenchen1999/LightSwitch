import os
import json
import imageio.v3 as imageio

import torch
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F

from dataset_colmap import (
    generate_directional_embeddings,
    generate_plucker_rays,
    hlg_oetf,
    perspective,
    rotate_x,
    pad_to_multiple,
    focal2fov,
)


def reconstruct_hdr_from_pngs(envmap_dir, frame_idx=0):
    """Reconstruct an HDR envmap from the _hdr.png + _ldr.png pair saved by preprocess_objaverse.

    preprocess_objaverse.py stores:
        _ldr.png = uint8( raw.clip(0,1) ** (1/2.2) * 255 )
        _hdr.png = uint8( log1p(10*raw) / max(log1p(10*raw)) * 255 )

    We invert both and use non-saturated LDR pixels to recover the unknown max_log scale.
    """
    hdr_path = os.path.join(envmap_dir, f"{frame_idx:05d}_hdr.png")
    ldr_path = os.path.join(envmap_dir, f"{frame_idx:05d}_ldr.png")

    hdr_png = imageio.imread(hdr_path)[..., :3].astype(np.float64) / 255.0
    ldr_png = imageio.imread(ldr_path)[..., :3].astype(np.float64) / 255.0

    ldr_linear = ldr_png ** 2.2

    hdr_norm = hdr_png  # = log1p(10*raw) / max_log

    non_sat = (ldr_linear > 0.01) & (ldr_linear < 0.95) & (hdr_norm > 0.01)
    if non_sat.any():
        max_log = np.median(np.log1p(10.0 * ldr_linear[non_sat]) / hdr_norm[non_sat])
    else:
        max_log = np.log1p(10.0)

    raw_hdr = np.expm1(hdr_norm * max_log) / 10.0
    return raw_hdr.astype(np.float32)


class DatasetPolyhaven(Dataset):
    """Dataset for polyhaven_lvsm format with JSON metadata (OpenCV w2c + fxfycxcy)."""

    def __init__(self, data_root, scene_name, args, transform,
                 env_transform=None, envmap_path=None, view_indices=None):
        self.args = args
        self.data_root = data_root
        self.scene_name = scene_name

        metadata_path = os.path.join(data_root, "metadata", f"{scene_name}.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        all_frames = metadata["frames"]

        if view_indices is not None:
            self.frames = [all_frames[i] for i in view_indices]
            self.view_indices = view_indices
        else:
            self.frames = all_frames
            self.view_indices = list(range(len(all_frames)))

        self.n_images = len(self.frames)
        self.downsample = args.downsample

        if os.path.isdir(envmap_path):
            self.envmap = reconstruct_hdr_from_pngs(envmap_path)
            print(f"Reconstructed HDR envmap from {envmap_path} "
                  f"(range [{self.envmap.min():.3f}, {self.envmap.max():.3f}])")
        else:
            self.envmap = imageio.imread(envmap_path)[..., :3]

        #todo: check if this is to the right or to theleft, Align with polyhaven convention: rotate envmap 90° left (rightmost 1/4 → leftmost)
        w = self.envmap.shape[1]
        self.envmap = np.roll(self.envmap, -w // 4, axis=1)

        if env_transform is None:
            self.env_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((256, 512), antialias=True),
                transforms.Normalize([0.5], [0.5]),
            ])
        else:
            self.env_transform = env_transform

        self.env_darker = (np.log10(self.envmap + 1) / np.log10(self.envmap.max())).clip(0, 1)
        self.env_brighter = hlg_oetf(self.env_darker).clip(0, 1)
        self.env_darker = self.env_transform(self.env_darker)
        self.env_brighter = self.env_transform(self.env_brighter)

        first_img = imageio.imread(self.frames[0]["image_path"])
        self.resolution = first_img.shape[:2]
        self.aspect = self.resolution[1] / self.resolution[0]
        self.h = self.resolution[0] // self.downsample
        self.w = self.resolution[1] // self.downsample

        self.all_img = [os.path.basename(f["image_path"]) for f in self.frames]

        self.transform = transform
        self.resize = transforms.Resize((self.h, self.w), antialias=True)
        self.dir_embeds = torch.tensor(
            generate_directional_embeddings(), dtype=torch.float32
        ).permute(2, 0, 1)

        print(f"DatasetPolyhaven: {self.n_images} images from '{scene_name}', "
              f"shape [{self.resolution[0]}, {self.resolution[1]}], downsample {self.downsample}")

    def _w2c_to_frame_transform(self, w2c):
        """Convert OpenCV w2c (4x4) to the frame_transform used by the pipeline.

        COLMAP stores R_c2w and t_w2c separately, then flips Y/Z.
        OpenCV w2c gives R_w2c and t_w2c directly, so we just transpose R.
        """
        w2c = np.array(w2c)
        R_c2w = w2c[:3, :3].T
        t_w2c = w2c[:3, 3]

        Rt = np.zeros((4, 4))
        Rt[:3, :3] = R_c2w
        Rt[:3, 3] = t_w2c
        Rt[:3, 1:3] *= -1
        Rt[3, 3] = 1.0
        return torch.tensor(Rt, dtype=torch.float32)

    def _parse_frame(self, cam_near_far=[0.1, 1000.0]):
        imgs, masks, mvps = [], [], []
        envs_darker, envs_brighter, dir_embeds, pluckers = [], [], [], []

        for i in range(self.n_images):
            frame = self.frames[i]
            raw_img = imageio.imread(frame["image_path"])
            raw_img = torch.tensor(raw_img / 255.0, dtype=torch.float32)

            raw_img = self.resize(raw_img.permute(2, 0, 1))

            if raw_img.shape[0] == 4:
                alpha = raw_img[3:4]
                img = raw_img[:3]
                mask = alpha.expand(3, -1, -1)
            else:
                img = raw_img
                mask = torch.ones_like(img)

            img = pad_to_multiple(img, multiple=8)
            mask = pad_to_multiple(mask, multiple=8)
            img = img * mask
            img = 2 * (img - 0.5)

            fx, fy, cx, cy = frame["fxfycxcy"]
            h_orig, w_orig = self.resolution
            FovX = focal2fov(fx, w_orig)
            FovY = focal2fov(fy, h_orig)

            frame_transform = self._w2c_to_frame_transform(frame["w2c"])
            mv = torch.linalg.inv(frame_transform)

            frame_pluckers = torch.tensor(
                generate_plucker_rays(
                    frame_transform, img.shape[1:3], [FovX, FovY]
                )
            )

            imgs.append(img)
            masks.append(mask)
            dir_embeds.append(self.dir_embeds)
            pluckers.append(frame_pluckers)
            envs_darker.append(self.env_darker)
            envs_brighter.append(self.env_brighter)

            proj = perspective(FovY, self.aspect, cam_near_far[0], cam_near_far[1])
            mv = mv @ rotate_x(-np.pi / 2)
            mvp = proj @ mv
            t = mvp[:3, 3]
            r = torch.linalg.norm(t)
            theta = torch.arccos(t[2] / r)
            phi = torch.arctan2(t[1], t[0])
            mvp = torch.tensor([theta, torch.sin(phi), torch.cos(phi), r])
            mvps.append(mvp)

        return (
            torch.stack(imgs), torch.stack(masks), torch.stack(mvps),
            torch.stack(envs_darker), torch.stack(envs_brighter),
            torch.stack(dir_embeds), torch.stack(pluckers),
        )

    def __len__(self):
        return 1

    def __getitem__(self, itr):
        img, mask, mvp, envs_darker, envs_brighter, dir_embeds, pluckers = self._parse_frame()
        return {
            'T': mvp,
            'img': img,
            'mask': mask,
            'dir_embeds': dir_embeds,
            'pluckers': pluckers,
            'envs_darker': envs_darker,
            'envs_brighter': envs_brighter,
        }

    def collate(self, batch):
        return {
            'image': torch.cat([b['img'] for b in batch]),
            'mask': torch.cat([b['mask'] for b in batch]),
            'T': torch.cat([b['T'] for b in batch]),
            'envs_darker': torch.cat([b['envs_darker'] for b in batch]),
            'envs_brighter': torch.cat([b['envs_brighter'] for b in batch]),
            'dir_embeds': torch.cat([b['dir_embeds'] for b in batch]),
            'pluckers': torch.cat([b['pluckers'] for b in batch]),
        }
