"""
evaluate.py

A script for evaluating the quality of rendered images.
"""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm

from src.rgb_metrics import (
    compute_lpips_between_directories,
    compute_psnr_between_directories,
    compute_ssim_between_directories,
)

DEFAULT_SCENES = ["chair", "lego", "materials", "drums"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate rendered 3DGS images.")
    parser.add_argument(
        "--ref-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "nerf_synthetic",
        help="Directory containing reference NeRF synthetic scene folders.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="Directory containing rendered scene folders.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=PROJECT_ROOT / "evaluation.txt",
        help="Path where evaluation metrics will be written.",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=DEFAULT_SCENES,
        help="Scene names to evaluate.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()

    # Directory containing rendered images
    ref_root = args.ref_root
    out_root = args.out_root
    scene_types = args.scenes
    
    # Evaluate
    metrics_list = []

    lpips_avg = 0.0
    psnr_avg = 0.0

    for scene_type in tqdm(scene_types):
        metrics = {}

        ref_dir = ref_root / scene_type / "test"
        out_dir = out_root / scene_type
        assert ref_dir.exists(), f"Scene {scene_type} not found."
        assert out_dir.exists(), f"Scene {scene_type} not found."
        print(f"Evaluating scene: {scene_type}")

        metrics["scene"] = scene_type
        metrics["lpips"] = compute_lpips_between_directories(out_dir, ref_dir)
        metrics["psnr"] = compute_psnr_between_directories(out_dir, ref_dir)
        #metrics["ssim"] = compute_ssim_between_directories(out_dir, ref_dir)

        lpips_avg += metrics["lpips"]
        psnr_avg += metrics["psnr"]
        #ssim_avg += metrics["ssim"]

        metrics_list.append(metrics)

    # Compute average
    lpips_avg /= len(scene_types)
    psnr_avg /= len(scene_types)
    #ssim_avg /= len(scene_types)
    metrics_list.append({
        "scene": "average",
        "lpips": lpips_avg,
        "psnr": psnr_avg,
    })

    # Save metrics to CSV
    print(f"LPIPS: {lpips_avg:.4f}")
    print(f"PSNR: {psnr_avg:.4f}")
    # print(f"SSIM: {ssim_avg:.4f}")

    # Save per-scene metrics and averages to evaluation.txt
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w") as f:
        for metrics in metrics_list:
            f.write(
                f"scene: {metrics['scene']}, "
                f"lpips: {metrics['lpips']:.4f}, "
                f"psnr: {metrics['psnr']:.4f}\n"
            )
    print("Done.")



if __name__ == "__main__":
    main()
