"""
Render a real 3DGS model using calibrated cameras from cameras.json.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import imageio.v2 as imageio
import numpy as np
import torch
import torchvision.utils as tvu
from tqdm import tqdm
import tyro

from src.camera import Camera
from src.renderer import GSRasterizer
from src.scene import Scene
from utils.render import compute_proj_mat, load_ply


@dataclass
class Args:
    model_root: Path = PROJECT_ROOT / "external_datasets" / "camenduru_gaussian_splatting" / "train"
    """Directory containing cameras.json and point_cloud/iteration_*/point_cloud.ply."""
    ply_path: Path | None = None
    """Optional direct path to a 3DGS point_cloud.ply."""
    cameras_path: Path | None = None
    """Optional direct path to cameras.json."""
    out_dir: Path = PROJECT_ROOT / "outputs" / "real_renders" / "camenduru_train"
    """Directory for rendered frames and animation."""
    device_type: str = "cuda"
    """Torch device, usually cuda or cpu."""
    max_width: int = 500
    """Render width cap. Output dimensions are rounded to the renderer tile size."""
    num_views: int = 12
    """Number of calibrated cameras to render."""
    start: int = 0
    """First camera index in cameras.json."""
    stride: int = 25
    """Stride through cameras.json."""
    background: str = "black"
    """Background color: black or white."""
    frustum_guard: float = 1.3
    """Keep projected splat centers within this NDC guard band."""
    max_splat_radius: float = 150.0
    """Cull unstable screen-space splats larger than this pixel radius."""
    near: float = 0.01
    """Near clipping plane."""
    far: float = 100.0
    """Far clipping plane."""
    fps: int = 6
    """Animation frame rate."""


def main(args: Args) -> None:
    ply_path = args.ply_path or args.model_root / "point_cloud" / "iteration_7000" / "point_cloud.ply"
    cameras_path = args.cameras_path or args.model_root / "cameras.json"
    assert ply_path.exists(), f"Missing PLY: {ply_path}"
    assert cameras_path.exists(), f"Missing cameras.json: {cameras_path}"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device_type)

    mean_3d, shs, opacities, scales, rotations = load_ply(ply_path)
    scene = Scene(
        mean_3d=mean_3d.to(device),
        shs=shs.to(device),
        opacities=opacities.to(device),
        scales=scales.to(device),
        rotations=rotations.to(device),
    )

    with open(cameras_path, "r", encoding="utf-8") as f:
        cameras = json.load(f)
    selected = cameras[args.start::args.stride][:args.num_views]
    assert selected, "No cameras selected."

    renderer = GSRasterizer()
    renderer.white_bkgd = args.background == "white"
    renderer.frustum_guard = args.frustum_guard
    renderer.max_splat_radius = args.max_splat_radius

    frames = []
    for frame_idx, cam_entry in tqdm(list(enumerate(selected)), total=len(selected)):
        camera = camera_from_entry(cam_entry, args, device, renderer.tile_size)
        img = renderer.render_scene(scene, camera).reshape(camera.image_height, camera.image_width, 3)
        img = torch.clamp(img, 0.0, 1.0)

        frame = (img.detach().cpu().numpy() * 255).astype(np.uint8)
        frames.append(frame)

        image_name = cam_entry.get("img_name", f"{cam_entry['id']:05d}")
        out_path = args.out_dir / f"cam_{cam_entry['id']:03d}_{image_name}.png"
        tvu.save_image(img.permute(2, 0, 1), out_path)

    imageio.mimsave(args.out_dir / "real_train_calibrated.gif", frames, fps=args.fps)
    writer = imageio.get_writer(args.out_dir / "real_train_calibrated.mp4", fps=args.fps, macro_block_size=1)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()

    print(f"Rendered {len(frames)} calibrated real-scene views to {args.out_dir}")


def camera_from_entry(cam_entry: dict, args: Args, device: torch.device, tile_size: int) -> Camera:
    width, height, scale_x, scale_y = scaled_dimensions(
        cam_entry["width"],
        cam_entry["height"],
        args.max_width,
        tile_size,
    )
    fx = cam_entry["fx"] * scale_x
    fy = cam_entry["fy"] * scale_y
    fov_x = 2.0 * math.atan(width / (2.0 * fx))
    fov_y = 2.0 * math.atan(height / (2.0 * fy))

    c2w_colmap = np.eye(4, dtype=np.float32)
    c2w_colmap[:3, :3] = np.asarray(cam_entry["rotation"], dtype=np.float32)
    c2w_colmap[:3, 3] = np.asarray(cam_entry["position"], dtype=np.float32)

    # GSRasterizer.render_scene applies diag(1, -1, -1) internally for Blender
    # transforms. cameras.json is already in the trained model's camera frame, so
    # pre-apply the same flip here to cancel that internal conversion.
    axis_flip = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    c2w_for_renderer = c2w_colmap.copy()
    c2w_for_renderer[:3, :3] = c2w_colmap[:3, :3] @ axis_flip

    c2w = torch.from_numpy(c2w_for_renderer).to(device)
    fov_x_t = torch.tensor(fov_x, device=device, dtype=torch.float32)
    fov_y_t = torch.tensor(fov_y, device=device, dtype=torch.float32)
    fx_t = torch.tensor(fx, device=device, dtype=torch.float32)
    fy_t = torch.tensor(fy, device=device, dtype=torch.float32)
    proj_mat = compute_proj_mat(args.near, args.far, fov_x_t, fov_y_t)

    return Camera(
        camera_to_world=c2w,
        proj_mat=proj_mat,
        cam_center=c2w[:3, 3],
        fov_x=fov_x_t,
        fov_y=fov_y_t,
        near=args.near,
        far=args.far,
        image_width=width,
        image_height=height,
        f_x=fx_t,
        f_y=fy_t,
        c_x=width / 2,
        c_y=height / 2,
    )


def scaled_dimensions(original_width: int, original_height: int, max_width: int, tile_size: int):
    target_width = min(original_width, max_width)
    target_height = original_height * target_width / original_width
    quantum = math.lcm(tile_size, 2)
    width = max(quantum, int(round(target_width / quantum)) * quantum)
    height = max(quantum, int(round(target_height / quantum)) * quantum)
    return width, height, width / original_width, height / original_height


if __name__ == "__main__":
    main(tyro.cli(Args))
