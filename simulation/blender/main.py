"""Blender CLI entrypoint for ITMS-VirtualTrack simulation tasks.

Usage inside Blender:
    blender -b --factory-startup --python simulation/blender/main.py -- --mode preview
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BLENDER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BLENDER_DIR.parent.parent
if str(BLENDER_DIR) not in sys.path:
    sys.path.insert(0, str(BLENDER_DIR))

from scene_manager import (  # noqa: E402
    run_milestone2_pipeline,
    run_milestone3_pipeline,
    run_milestone4_speed_run,
    run_milestone5_masks,
    run_milestone5_variant,
)


def _parse_blender_args(argv: list[str]) -> argparse.Namespace:
    if "--" in argv:
        script_args = argv[argv.index("--") + 1 :]
    else:
        script_args = []

    parser = argparse.ArgumentParser(
        description="ITMS-VirtualTrack Blender headless simulation entrypoint."
    )
    parser.add_argument(
        "--milestone",
        choices=("m2", "m3", "m4", "m5_masks", "m5_variant"),
        default="m2",
        help="Pipeline milestone to run.",
    )
    parser.add_argument(
        "--mode",
        choices=("preview", "production"),
        default="preview",
        help="Resolution/FPS/sample preset from camera_config and rendering_config.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=60.0,
        help="Train speed in km/h (used for m4 / m5_masks).",
    )
    parser.add_argument(
        "--seg-dir",
        default=None,
        help="Segmentation output root directory (used for m5_masks).",
    )
    parser.add_argument(
        "--stem-prefix",
        default=None,
        help="Frame filename prefix (used for m5_masks).",
    )
    parser.add_argument(
        "--variant-config",
        default=None,
        help="Path to variant JSON config (used for m5_variant).",
    )
    parser.add_argument(
        "--motion-blur",
        action="store_true",
        help="Enable Blender motion blur for this run.",
    )
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="Run trajectory & visibility validation without rendering video (m4).",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip preview MP4 rendering.",
    )
    parser.add_argument(
        "--no-keyframes",
        action="store_true",
        help="Skip PNG keyframe rendering.",
    )
    parser.add_argument(
        "--no-blend",
        action="store_true",
        help="Skip saving .blend file.",
    )
    return parser.parse_args(script_args)


def main() -> int:
    args = _parse_blender_args(sys.argv)
    if args.milestone == "m5_masks":
        if not args.seg_dir or not args.stem_prefix:
            raise ValueError("--seg-dir and --stem-prefix are required for m5_masks")
        res = run_milestone5_masks(
            project_root=PROJECT_ROOT,
            speed_kmh=args.speed,
            mode=args.mode,
            seg_out_dir=Path(args.seg_dir),
            stem_prefix=args.stem_prefix,
        )
        print(
            f"[ITMS-VirtualTrack] M5 masks ({args.speed} km/h): "
            f"A3={res['a3_mask_count']}, A4={res['a4_mask_count']}"
        )
        return 0

    if args.milestone == "m5_variant":
        if not args.variant_config:
            raise ValueError("--variant-config is required for m5_variant")
        res = run_milestone5_variant(
            project_root=PROJECT_ROOT,
            variant_cfg_path=Path(args.variant_config),
            mode=args.mode,
        )
        passed = res["validation"]["all_passed"]
        print(
            f"[ITMS-VirtualTrack] M5 variant {res['variant_id']} passed={passed}"
        )
        return 0 if passed else 2

    if args.milestone == "m4":
        result = run_milestone4_speed_run(
            project_root=PROJECT_ROOT,
            speed_kmh=args.speed,
            mode=args.mode,
            motion_blur=True if args.motion_blur else None,
            skip_render=args.skip_render,
        )
        passed = result["validation"]["all_passed"]
        print(
            f"[ITMS-VirtualTrack] Milestone 4 ({args.speed} km/h, {args.mode}) passed: {passed}"
        )
        return 0 if passed else 2

    if args.milestone == "m3":
        result = run_milestone3_pipeline(project_root=PROJECT_ROOT, mode=args.mode)
        passed = result["validation"]["blender_all_passed"]
        print(f"[ITMS-VirtualTrack] Milestone 3 Blender validation passed: {passed}")
        return 0 if passed else 2

    result = run_milestone2_pipeline(
        project_root=PROJECT_ROOT,
        mode=args.mode,
        render_video=not args.no_video,
        render_keyframes=not args.no_keyframes,
        save_blend=not args.no_blend,
    )
    passed = result["validation"]["blender_all_passed"]
    print(f"[ITMS-VirtualTrack] Milestone 2 Blender validation passed: {passed}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

