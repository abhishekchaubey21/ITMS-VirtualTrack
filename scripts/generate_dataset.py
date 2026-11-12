"""Milestone 5 dataset generation, frame extraction, annotation, and validation CLI.

Combines:
  Group A: Controlled Evaluation Dataset (all 7 existing M4 speed videos)
  Group B: Independently Randomized Scene Variants (70% train / 15% val / 15% test)

Preserves all M2, M3, and M4 outputs untouched on D:.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from check_environment import find_blender_executable, load_json  # noqa: E402
from simulation.python.annotation_converter import (  # noqa: E402
    convert_video_annotations,
    write_dataset_configs,
)
from simulation.python.dataset_manager import (  # noqa: E402
    build_manifest_and_statistics,
    generate_visual_samples,
    validate_dataset_quality,
)
from simulation.python.video_processor import extract_frames_from_video  # noqa: E402


PRESERVED_FILES = [
    "results/scenes/railway_250m_base.blend",
    "results/scenes/railway_250m_baseline.blend",
    "results/scenes/railway_250m_A1_only.blend",
    "results/scenes/railway_250m_A2_only.blend",
    "results/scenes/railway_250m_A3_only.blend",
    "results/scenes/railway_250m_A4_only.blend",
    "results/scenes/railway_250m_anomalies.blend",
    "results/videos/m4_speed_runs/railway_30kmh_preview.mp4",
    "results/videos/m4_speed_runs/railway_60kmh_preview.mp4",
    "results/videos/m4_speed_runs/railway_90kmh_preview.mp4",
    "results/videos/m4_speed_runs/railway_120kmh_preview.mp4",
    "results/videos/m4_speed_runs/railway_150kmh_preview.mp4",
    "results/videos/m4_speed_runs/railway_180kmh_preview.mp4",
    "results/videos/m4_speed_runs/railway_200kmh_preview.mp4",
]


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _create_dataset_structure(dataset_root: Path) -> None:
    subdirs = [
        "videos",
        "images/train",
        "images/val",
        "images/test",
        "labels/train",
        "labels/val",
        "labels/test",
        "segmentation/train/a3_masks",
        "segmentation/train/a4_masks",
        "segmentation/val/a3_masks",
        "segmentation/val/a4_masks",
        "segmentation/test/a3_masks",
        "segmentation/test/a4_masks",
        "ground_truth/frame_annotations",
        "ground_truth/anomaly_metadata",
        "ground_truth/geometry_profiles",
        "metadata",
        "configs",
    ]
    for sub in subdirs:
        (dataset_root / sub).mkdir(parents=True, exist_ok=True)


def _compute_track_datum(sim_cfg: dict[str, Any]) -> dict[str, float]:
    track = sim_cfg["track"]
    gauge_m = float(track["gauge_m"])
    head_w = float(track["rail"]["head_width_m"])
    left_cx = -gauge_m * 0.5 - head_w * 0.5
    right_cx = gauge_m * 0.5 + head_w * 0.5
    return {
        "left_rail_center_x": left_cx,
        "right_rail_center_x": right_cx,
    }


def _generate_variant_specs(
    train_count: int = 14,
    val_count: int = 3,
    test_count: int = 3,
    base_seed: int = 20261111,
) -> list[dict[str, Any]]:
    """Build 70% / 15% / 15% independent randomized scene variant configurations."""
    rng = random.Random(base_seed)
    speeds_cycle = [120, 150, 180]
    total = train_count + val_count + test_count
    specs: list[dict[str, Any]] = []

    for idx in range(total):
        if idx < train_count:
            split = "train"
        elif idx < train_count + val_count:
            split = "val"
        else:
            split = "test"

        variant_id = f"var_{idx:02d}"
        speed_kmh = speeds_cycle[idx % len(speeds_cycle)]
        # Distinct sun, weather turbidity, exposure, surface tints, and anomaly geometry
        weather_roll = rng.random()
        if weather_roll < 0.30:
            weather_name = "hazy_overcast"
            turbidity = round(rng.uniform(4.8, 6.8), 2)
            sun_energy = round(rng.uniform(1.8, 2.8), 2)
            sky_strength = round(rng.uniform(0.13, 0.17), 3)
        elif weather_roll < 0.65:
            weather_name = "clear_daylight"
            turbidity = round(rng.uniform(1.8, 2.8), 2)
            sun_energy = round(rng.uniform(4.0, 5.2), 2)
            sky_strength = round(rng.uniform(0.10, 0.14), 3)
        else:
            weather_name = "low_sun_warm"
            turbidity = round(rng.uniform(2.6, 4.4), 2)
            sun_energy = round(rng.uniform(3.0, 4.4), 2)
            sky_strength = round(rng.uniform(0.09, 0.13), 3)

        variant_params = {
            "weather": weather_name,
            "sun_elevation_deg": round(rng.uniform(16.0, 66.0), 2),
            "sun_azimuth_deg": round(rng.uniform(75.0, 285.0), 2),
            "sun_energy": sun_energy,
            "sky_strength": sky_strength,
            "turbidity": turbidity,
            "exposure_ev": round(rng.uniform(-0.35, 0.20), 3),
            "ballast_tint_scale": round(rng.uniform(0.82, 1.18), 3),
            "ground_tint_scale": round(rng.uniform(0.84, 1.16), 3),
            "concrete_tint_scale": round(rng.uniform(0.86, 1.14), 3),
            "anomaly_scale_factor": 1.0,
            "a1_scale_factor": round(rng.uniform(0.86, 1.16), 3),
            "a2_scale_factor": round(rng.uniform(0.85, 1.18), 3),
            "a3_scale_factor": round(rng.uniform(0.88, 1.20), 3),
            "a4_deviation_scale": round(rng.uniform(0.80, 1.25), 3),
            "anomaly_yaw_delta_deg": 0.0,
            "a1_yaw_delta_deg": round(rng.uniform(-55.0, 55.0), 2),
            "a2_yaw_delta_deg": round(rng.uniform(-45.0, 45.0), 2),
            "anomaly_lateral_jitter_m": round(rng.uniform(-0.22, 0.22), 4),
        }

        specs.append(
            {
                "variant_id": variant_id,
                "split": split,
                "dataset_group": "randomized_variants",
                "seed_offset": 2000 + idx * 17,
                "speed_kmh": speed_kmh,
                "stem_prefix": variant_id,
                "variant_params": variant_params,
            }
        )

    return specs


def _run_blender(
    blender_exe: str,
    env: dict[str, str],
    blender_args: list[str],
    log_handle: Any,
) -> int:
    blender_main = PROJECT_ROOT / "simulation" / "blender" / "main.py"
    cmd = [
        blender_exe,
        "-b",
        "--factory-startup",
        "--python-exit-code",
        "1",
        "--python",
        str(blender_main),
        "--",
        *blender_args,
    ]
    proc = subprocess.run(
        cmd,
        env=env,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log_handle.write(proc.stdout or "")
    log_handle.write(f"\n[Exit code: {proc.returncode}]\n")
    log_handle.flush()
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Milestone 5 synthetic dataset, annotations, masks, and validation."
    )
    parser.add_argument(
        "--mode",
        choices=("preview", "production"),
        default="preview",
        help="Input M4 mode & variant render preset (default: preview 960x540 @ 30 FPS).",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Frame extraction stride (default: 1 = retain all frames).",
    )
    parser.add_argument(
        "--eval-speeds",
        type=float,
        nargs="+",
        default=None,
        help="Optional subset of M4 speeds (defaults to all 7 speeds).",
    )
    parser.add_argument(
        "--train-variants",
        type=int,
        default=14,
        help="Number of independent training scene variants (default: 14 = 70%%).",
    )
    parser.add_argument(
        "--val-variants",
        type=int,
        default=3,
        help="Number of independent validation scene variants (default: 3 = 15%%).",
    )
    parser.add_argument(
        "--test-variants",
        type=int,
        default=3,
        help="Number of independent test scene variants (default: 3 = 15%%).",
    )
    parser.add_argument(
        "--dataset-dir",
        default="dataset/synthetic",
        help="Output dataset directory relative to project root.",
    )
    parser.add_argument(
        "--samples-dir",
        default="results/videos/m5_gt_samples",
        help="Output visual verification overlays directory.",
    )
    parser.add_argument(
        "--blender",
        default=None,
        help="Optional explicit path to blender.exe.",
    )
    args = parser.parse_args()

    t_start = time.perf_counter()
    sim_cfg = load_json(PROJECT_ROOT / "config" / "simulation_config.json")
    anom_cfg = load_json(PROJECT_ROOT / "config" / "anomaly_config.json")
    cam_cfg = load_json(PROJECT_ROOT / "config" / "camera_config.json")
    m3_meta = load_json(PROJECT_ROOT / "results" / "metrics" / "m3_anomaly_metadata.json")
    track_datum = _compute_track_datum(sim_cfg)

    blender_exe = find_blender_executable(args.blender, sim_cfg)
    if not blender_exe:
        print("ERROR: Blender executable not found.", file=sys.stderr)
        return 1

    dataset_root = PROJECT_ROOT / args.dataset_dir
    _create_dataset_structure(dataset_root)

    temp_dir = PROJECT_ROOT / sim_cfg["paths"].get("temp_dir", "results/_tmp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    variant_tmp_dir = temp_dir / "m5_variants"
    variant_tmp_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["TEMP"] = str(temp_dir)
    env["TMP"] = str(temp_dir)
    env["TMPDIR"] = str(temp_dir)

    preserved_before = {rel: _sha256_file(PROJECT_ROOT / rel) for rel in PRESERVED_FILES}
    log_path = PROJECT_ROOT / "results" / "metrics" / "m5_dataset_generation.log"

    eval_speeds = args.eval_speeds or list(sim_cfg["simulation"]["speeds_kmh"])
    suffix = "_preview" if args.mode == "preview" else ""

    video_entries: list[dict[str, Any]] = []
    all_frame_records: list[dict[str, Any]] = []

    with log_path.open("w", encoding="utf-8") as log_handle:
        # ------------------------------------------------------------------
        # GROUP A: Controlled Evaluation Dataset (7 existing M4 speed runs)
        # ------------------------------------------------------------------
        for spd in eval_speeds:
            spd_tag = int(spd) if float(spd).is_integer() else spd
            video_id = f"eval_{spd_tag}kmh"
            src_video = (
                PROJECT_ROOT
                / "results"
                / "videos"
                / "m4_speed_runs"
                / f"railway_{spd_tag}kmh{suffix}.mp4"
            )
            src_meta_path = (
                PROJECT_ROOT
                / "results"
                / "metrics"
                / "m4_speed_runs"
                / f"railway_{spd_tag}kmh{suffix}_metadata.json"
            )
            if not src_video.is_file() or not src_meta_path.is_file():
                raise FileNotFoundError(
                    f"Missing M4 artifact for {spd_tag} km/h: {src_video} / {src_meta_path}"
                )

            dst_video = dataset_root / "videos" / src_video.name
            if not dst_video.exists() or _sha256_file(dst_video) != _sha256_file(src_video):
                shutil.copy2(src_video, dst_video)

            raw_meta = load_json(src_meta_path)
            print(f"[M5] Group A ({video_id}): rendering A3/A4 masks & extracting frames...")

            rc = _run_blender(
                blender_exe,
                env,
                [
                    "--milestone",
                    "m5_masks",
                    "--mode",
                    args.mode,
                    "--speed",
                    str(spd_tag),
                    "--seg-dir",
                    str(dataset_root / "segmentation" / "test"),
                    "--stem-prefix",
                    video_id,
                ],
                log_handle,
            )
            if rc != 0:
                raise RuntimeError(f"Blender mask pass failed for {video_id} (rc={rc})")

            ext_res = extract_frames_from_video(
                video_path=dst_video,
                output_dir=dataset_root / "images" / "test",
                stem_prefix=video_id,
                expected_frame_count=int(raw_meta["frame_count"]),
                frame_stride=args.frame_stride,
                image_ext="png",
            )

            conv_res = convert_video_annotations(
                video_id=video_id,
                source_video_path=dst_video,
                source_scene="base_experiment_m4",
                split="test",
                dataset_group="controlled_evaluation",
                stem_prefix=video_id,
                raw_meta=raw_meta,
                extracted_frames=ext_res["frames"],
                anomalies_world_meta=m3_meta["anomalies"],
                a4_cfg=anom_cfg["anomalies"]["A4"],
                track_datum=track_datum,
                dataset_root=dataset_root,
            )

            video_entries.append(
                {
                    "video_id": video_id,
                    "source_scene": "base_experiment_m4",
                    "split": "test",
                    "dataset_group": "controlled_evaluation",
                    "speed_kmh": raw_meta["speed_kmh"],
                    "fps": raw_meta["fps"],
                    "expected_frames": int(raw_meta["frame_count"]),
                    "extracted_frames": ext_res["extracted_count"],
                    "video_path": str(dst_video),
                    "frame_gt_path": conv_res["frame_gt_path"],
                    "yolo_boxes": conv_res["yolo_boxes_written"],
                    "a3_masks": conv_res["a3_masks_linked"],
                    "a4_masks": conv_res["a4_masks_linked"],
                }
            )
            all_frame_records.extend(conv_res["frame_records"])

        # ------------------------------------------------------------------
        # GROUP B: Independent Randomized Scene Variants (70 / 15 / 15)
        # ------------------------------------------------------------------
        variant_specs = _generate_variant_specs(
            train_count=args.train_variants,
            val_count=args.val_variants,
            test_count=args.test_variants,
            base_seed=20261111,
        )

        for spec in variant_specs:
            vid = spec["variant_id"]
            split = spec["split"]
            v_video = dataset_root / "videos" / f"{vid}_{spec['speed_kmh']}kmh.mp4"
            v_meta_path = variant_tmp_dir / f"{vid}_raw_meta.json"
            v_cfg_path = variant_tmp_dir / f"{vid}_cfg.json"

            full_spec = {
                **spec,
                "video_out_path": str(v_video),
                "seg_out_dir": str(dataset_root / "segmentation" / split),
                "meta_out_path": str(v_meta_path),
            }
            with v_cfg_path.open("w", encoding="utf-8") as handle:
                json.dump(full_spec, handle, indent=2)

            print(
                f"[M5] Group B ({vid} -> {split}, {spec['speed_kmh']} km/h, "
                f"{spec['variant_params']['weather']})..."
            )
            rc = _run_blender(
                blender_exe,
                env,
                [
                    "--milestone",
                    "m5_variant",
                    "--mode",
                    args.mode,
                    "--variant-config",
                    str(v_cfg_path),
                ],
                log_handle,
            )
            if rc != 0:
                raise RuntimeError(f"Blender variant generation failed for {vid} (rc={rc})")

            raw_meta = load_json(v_meta_path)
            ext_res = extract_frames_from_video(
                video_path=v_video,
                output_dir=dataset_root / "images" / split,
                stem_prefix=vid,
                expected_frame_count=int(raw_meta["frame_count"]),
                frame_stride=args.frame_stride,
                image_ext="png",
            )

            # Build scaled A4 config for accurate geometry profile export
            a4_scaled_cfg = dict(anom_cfg["anomalies"]["A4"])
            a4_scaled_cfg["max_lateral_deviation_m"] = round(
                float(a4_scaled_cfg["max_lateral_deviation_m"])
                * float(spec["variant_params"]["a4_deviation_scale"]),
                5,
            )

            conv_res = convert_video_annotations(
                video_id=vid,
                source_video_path=v_video,
                source_scene=vid,
                split=split,
                dataset_group="randomized_variants",
                stem_prefix=vid,
                raw_meta=raw_meta,
                extracted_frames=ext_res["frames"],
                anomalies_world_meta=raw_meta["anomalies_metadata"],
                a4_cfg=a4_scaled_cfg,
                track_datum=track_datum,
                dataset_root=dataset_root,
            )

            video_entries.append(
                {
                    "video_id": vid,
                    "source_scene": vid,
                    "split": split,
                    "dataset_group": "randomized_variants",
                    "random_seed": raw_meta["random_seed"],
                    "variant_params": spec["variant_params"],
                    "speed_kmh": raw_meta["speed_kmh"],
                    "fps": raw_meta["fps"],
                    "expected_frames": int(raw_meta["frame_count"]),
                    "extracted_frames": ext_res["extracted_count"],
                    "video_path": str(v_video),
                    "frame_gt_path": conv_res["frame_gt_path"],
                    "yolo_boxes": conv_res["yolo_boxes_written"],
                    "a3_masks": conv_res["a3_masks_linked"],
                    "a4_masks": conv_res["a4_masks_linked"],
                }
            )
            all_frame_records.extend(conv_res["frame_records"])

    # Configs, manifest, statistics, quality validation, visual samples
    cfg_paths = write_dataset_configs(dataset_root, anom_cfg)

    total_var = max(1, args.train_variants + args.val_variants + args.test_variants)
    split_plan = {
        "base_seed": 20261111,
        "controlled_evaluation_videos": len(eval_speeds),
        "randomized_variant_scenes": {
            "train": args.train_variants,
            "val": args.val_variants,
            "test": args.test_variants,
            "train_ratio": round(args.train_variants / total_var, 4),
            "val_ratio": round(args.val_variants / total_var, 4),
            "test_ratio": round(args.test_variants / total_var, 4),
        },
        "leakage_prevention": "Split strictly by independent scene/variant ID; evaluation M4 runs are excluded from train and val.",
    }

    man_res = build_manifest_and_statistics(
        dataset_root=dataset_root,
        video_entries=video_entries,
        all_frame_records=all_frame_records,
        split_plan=split_plan,
    )

    res_cfg = cam_cfg["resolution"]["preview" if args.mode == "preview" else "production"]
    val_res = validate_dataset_quality(
        dataset_root=dataset_root,
        video_entries=video_entries,
        all_frame_records=all_frame_records,
        expected_width=int(res_cfg["width"]),
        expected_height=int(res_cfg["height"]),
    )

    samples_res = generate_visual_samples(
        dataset_root=dataset_root,
        all_frame_records=all_frame_records,
        output_dir=PROJECT_ROOT / args.samples_dir,
        seed=42,
    )

    preserved_after = {rel: _sha256_file(PROJECT_ROOT / rel) for rel in PRESERVED_FILES}
    preserved_ok = all(
        preserved_before[k] is not None and preserved_before[k] == preserved_after[k]
        for k in PRESERVED_FILES
    )

    total_elapsed_s = round(time.perf_counter() - t_start, 2)
    report_path = PROJECT_ROOT / "results" / "metrics" / "m5_validation_report.json"
    full_report = {
        "dataset_type": "SYNTHETIC",
        "milestone": "M5_DATASET_GENERATION",
        "elapsed_seconds": total_elapsed_s,
        "preserved_m2_m3_m4_unchanged": preserved_ok,
        "configs": cfg_paths,
        "manifest_path": man_res["manifest_path"],
        "statistics_path": man_res["statistics_path"],
        "statistics": man_res["statistics"],
        "validation": val_res,
        "visual_samples": samples_res,
        "overall_passed": bool(val_res["passed"] and preserved_ok),
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(full_report, handle, indent=2)

    print("\n" + "=" * 68)
    print("ITMS-VirtualTrack — Milestone 5 Dataset Report (SYNTHETIC)")
    print("=" * 68)
    print(f"Total videos processed     : {man_res['statistics']['total_video_count']}")
    print(f"Total frames extracted     : {man_res['statistics']['total_extracted_frame_count']}")
    print(f"YOLO bounding boxes (A1/A2): {man_res['statistics']['total_yolo_bounding_box_annotations']}")
    print(f"Segmentation masks (A3/A4) : {man_res['statistics']['total_segmentation_masks']['total']}")
    print(f"Leakage-free split OK      : {val_res['checks']['leakage_free']}")
    print(f"Preserved M2/M3/M4 OK      : {preserved_ok}")
    print(f"Storage total              : {man_res['statistics']['storage_consumption_mb']['total_dataset_mb']} MB")
    print(f"Overall Milestone 5 Passed : {full_report['overall_passed']}")
    print(f"Report saved to            : {report_path}")
    print("=" * 68)

    return 0 if full_report["overall_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

