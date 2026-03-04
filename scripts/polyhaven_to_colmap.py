#!/usr/bin/env python3
"""Convert polyhaven/objaverse JSON metadata to COLMAP binary format + RGB images.

Creates a directory structure compatible with gaussian-splatting:
    output_dir/
    ├── images/          # RGB PNGs (alpha stripped)
    └── sparse/0/
        ├── cameras.bin
        ├── images.bin
        └── points3d.bin
"""

import os
import sys
import json
import struct
import argparse

import numpy as np
from PIL import Image


# COLMAP camera model IDs
CAMERA_MODEL_PINHOLE = 1  # params: fx, fy, cx, cy


def rotmat2qvec(R):
    """Convert 3x3 rotation matrix to COLMAP quaternion (w, x, y, z)."""
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
    ]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def write_cameras_bin(cameras, path):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for cam_id, cam in cameras.items():
            f.write(struct.pack("<I", cam_id))
            f.write(struct.pack("<i", cam["model_id"]))
            f.write(struct.pack("<Q", cam["width"]))
            f.write(struct.pack("<Q", cam["height"]))
            for p in cam["params"]:
                f.write(struct.pack("<d", p))


def write_images_bin(images, path):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for img_id, img in images.items():
            f.write(struct.pack("<I", img_id))
            for q in img["qvec"]:
                f.write(struct.pack("<d", q))
            for t in img["tvec"]:
                f.write(struct.pack("<d", t))
            f.write(struct.pack("<I", img["camera_id"]))
            f.write(img["name"].encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", 0))  # num_points2D


def write_points3d_bin(path, num_points=500, bbox_min=-0.5, bbox_max=0.5, seed=42):
    """Write points3d.bin with random points for GS initialization.

    Polyhaven/objaverse objects are centered near the world origin,
    so we scatter points in a cube around it.
    """
    rng = np.random.RandomState(seed)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", num_points))
        for pid in range(1, num_points + 1):
            xyz = rng.uniform(bbox_min, bbox_max, size=3)
            rgb = rng.randint(128, 256, size=3, dtype=np.uint8)
            f.write(struct.pack("<Q", pid))           # point3D_id
            for v in xyz:
                f.write(struct.pack("<d", v))         # x, y, z
            for c in rgb:
                f.write(struct.pack("<B", int(c)))    # r, g, b
            f.write(struct.pack("<d", 0.0))           # error
            f.write(struct.pack("<Q", 0))             # track_length = 0


def convert_scene(metadata_path, output_dir, downsample=1, image_source_dir=None, skip_images=False):
    """Convert a single scene from polyhaven metadata to COLMAP format.

    Args:
        metadata_path: Path to the scene's JSON metadata file.
        output_dir: Where to write the COLMAP-format output.
        downsample: Downsample factor for images and intrinsics.
        image_source_dir: If set, copy images from here instead of metadata paths.
        skip_images: If True, only write sparse data (useful when images already exist).
    """
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    frames = metadata["frames"]
    if len(frames) == 0:
        print(f"No frames found in {metadata_path}")
        return

    first_img = Image.open(frames[0]["image_path"])
    orig_w, orig_h = first_img.size
    w = orig_w // downsample
    h = orig_h // downsample

    sparse_dir = os.path.join(output_dir, "sparse", "0")
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(sparse_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    cameras = {}
    colmap_images = {}

    fx0, fy0, cx0, cy0 = frames[0]["fxfycxcy"]
    all_same_intrinsics = all(
        frames[i]["fxfycxcy"] == frames[0]["fxfycxcy"] for i in range(len(frames))
    )

    if all_same_intrinsics:
        cameras[1] = {
            "model_id": CAMERA_MODEL_PINHOLE,
            "width": w,
            "height": h,
            "params": [fx0 / downsample, fy0 / downsample,
                        cx0 / downsample, cy0 / downsample],
        }

    for i, frame in enumerate(frames):
        img_id = i + 1  # COLMAP IDs are 1-based
        fname = os.path.basename(frame["image_path"])

        if not all_same_intrinsics:
            fx, fy, cx, cy = frame["fxfycxcy"]
            cameras[img_id] = {
                "model_id": CAMERA_MODEL_PINHOLE,
                "width": w,
                "height": h,
                "params": [fx / downsample, fy / downsample,
                            cx / downsample, cy / downsample],
            }

        w2c = np.array(frame["w2c"])
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        qvec = rotmat2qvec(R)

        colmap_images[img_id] = {
            "qvec": qvec.tolist(),
            "tvec": t.tolist(),
            "camera_id": 1 if all_same_intrinsics else img_id,
            "name": fname,
        }

        if not skip_images:
            src_path = frame["image_path"]
            if image_source_dir is not None:
                src_path = os.path.join(image_source_dir, fname)

            img = Image.open(src_path)
            if downsample > 1:
                img = img.resize((w, h), Image.LANCZOS)
            if img.mode == "RGBA":
                bg = Image.new("RGB", img.size, (0, 0, 0))
                bg.paste(img, mask=img.split()[3])
                img = bg
            elif img.mode != "RGB":
                img = img.convert("RGB")
            img.save(os.path.join(images_dir, fname))

    write_cameras_bin(cameras, os.path.join(sparse_dir, "cameras.bin"))
    write_images_bin(colmap_images, os.path.join(sparse_dir, "images.bin"))
    write_points3d_bin(os.path.join(sparse_dir, "points3d.bin"))

    print(f"Wrote COLMAP data: {len(colmap_images)} images, {len(cameras)} camera(s) "
          f"[{w}x{h}] -> {sparse_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert polyhaven metadata to COLMAP format")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory (contains metadata/, images/)")
    parser.add_argument("--scene_name", type=str, required=True,
                        help="Scene name (matches metadata/{scene_name}.json)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for COLMAP-format data")
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--image_source_dir", type=str, default=None,
                        help="Override image source directory (e.g. relit images)")
    parser.add_argument("--skip_images", action="store_true",
                        help="Only write sparse data, skip image copying")
    args = parser.parse_args()

    metadata_path = os.path.join(args.data_root, "metadata", f"{args.scene_name}.json")
    if not os.path.exists(metadata_path):
        print(f"Metadata not found: {metadata_path}", file=sys.stderr)
        sys.exit(1)

    convert_scene(
        metadata_path=metadata_path,
        output_dir=args.output_dir,
        downsample=args.downsample,
        image_source_dir=args.image_source_dir,
        skip_images=args.skip_images,
    )


if __name__ == "__main__":
    main()
