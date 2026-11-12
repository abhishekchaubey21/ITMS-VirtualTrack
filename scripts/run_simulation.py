"""Sequential 7-speed simulation runner, OpenCV validator, and contact-sheet builder.

Runs each speed condition sequentially in Blender (preserving M2/M3 .blend files),
verifies every rendered video with OpenCV, builds 7-frame inspection contact sheets,
and generates visibility, render-performance, and validation reports on D:.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from check_environment import find_blender_executable, load_json  # noqa: E402


PRESERVED_BLEND_FILES = [
    "results/scenes/railway_250m_base.blend",
    "results/scenes/railway_250m_baseline.blend",
    "results/scenes/railway_250m_A1_only.blend",
    "results/scenes/railway_250m_A2_only.blend",
    "results/scenes/railway_250m_A3_only.blend",
    "results/scenes/railway_250m_A4_only.blend",
    "results/scenes/railway_250m_anomalies.blend",
]

CONTACT_SHEET_TARGETS = [
    ("Start (0m)", 0.0),
    ("Before A1 (35m)", 35.0),
    ("A1 @ 50m (cam 42.5m)", 42.5),
    ("A2 @ 100m (cam 91.5m)", 91.5),
    ("A3 @ 150m (cam 143.8m)", 143.8),
    ("A4 @ 200m (cam 186.0m)", 186.0),
    ("End (250m)", 250.0),
]


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _snapshot_preserved_blends() -> dict[str, str | None]:
    return {
        rel: _sha256_file(PROJECT_ROOT / rel) for rel in PRESERVED_BLEND_FILES
    }


def _verify_video_with_opencv(
    video_path: Path, meta: dict[str, Any]
) -> dict[str, Any]:
    if not video_path.is_file():
        return {"passed": False, "exists": False, "path": str(video_path)}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"passed": False, "exists": True, "opened": False, "path": str(video_path)}

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    expected_fc = int(meta["frame_count"])
    expected_fps = float(meta["fps"])
    expected_w = int(meta["blender_render_configuration"]["width"])
    expected_h = int(meta["blender_render_configuration"]["height"])

    sample_indices = sorted({0, max(0, frame_count // 2), max(0, frame_count - 1)})
    sample_stds: list[float] = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            sample_stds.append(round(float(np.std(frame)), 2))
        else:
            sample_stds.append(0.0)
    cap.release()

    passed = (
        frame_count == expected_fc
        and abs(fps - expected_fps) <= 0.25
        and width == expected_w
        and height == expected_h
        and all(s > 5.0 for s in sample_stds)
    )
    return {
        "passed": passed,
        "exists": True,
        "opened": True,
        "path": str(video_path),
        "size_bytes": int(video_path.stat().st_size),
        "expected_frame_count": expected_fc,
        "measured_frame_count": frame_count,
        "expected_fps": expected_fps,
        "measured_fps": round(fps, 3),
        "resolution": [width, height],
        "sample_frame_stds": sample_stds,
    }


def _build_contact_sheet(
    video_path: Path, meta: dict[str, Any], output_path: Path
) -> str | None:
    if not video_path.is_file():
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    frames_meta = meta["frames"]
    tile_w, tile_h = 480, 270
    cols, rows = 4, 2
    canvas = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    canvas[:] = (22, 24, 28)

    for slot_idx, (label, target_y) in enumerate(CONTACT_SHEET_TARGETS):
        best = min(frames_meta, key=lambda f: abs(float(f["camera_position_m"]) - target_y))
        f_idx = int(best["frame_index"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ok, img = cap.read()
        r = slot_idx // cols
        c = slot_idx % cols
        x0, y0 = c * tile_w, r * tile_h

        if ok and img is not None:
            tile = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        else:
            tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)

        # Header banner on tile
        cv2.rectangle(tile, (0, 0), (tile_w, 30), (15, 18, 22), -1)
        caption = (
            f"{label} | F{best['frame']} t={best['timestamp_s']:.2f}s "
            f"y={best['camera_position_m']:.2f}m"
        )
        cv2.putText(
            tile,
            caption,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (235, 240, 245),
            1,
            cv2.LINE_AA,
        )
        canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile

    cap.release()

    # 8th tile: Run Summary Card
    sx0, sy0 = 3 * tile_w, 1 * tile_h
    summary_lines = [
        f"ITMS-VirtualTrack (SYNTHETIC)",
        f"Speed: {meta['speed_kmh']} km/h ({meta['speed_mps']:.2f} m/s)",
        f"Mode: {meta['blender_render_configuration']['mode']} ({meta['fps']} FPS)",
        f"Res: {meta['blender_render_configuration']['width']}x{meta['blender_render_configuration']['height']}",
        f"Frames: {meta['frame_count']} | Travel: {meta['actual_duration_s']:.2f}s",
        f"Step/frame: {meta['distance_per_frame_m']:.3f} m",
        f"Range: {meta['camera_start_position_m']:.2f}m -> {meta['camera_end_position_m']:.2f}m",
        f"Render time: {meta['blender_render_configuration']['render_time_seconds']:.2f}s",
    ]
    for li, line in enumerate(summary_lines):
        color = (100, 220, 255) if li == 0 else (225, 230, 235)
        cv2.putText(
            canvas,
            line,
            (sx0 + 18, sy0 + 38 + li * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    return str(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run sequential 7-speed synthetic railway simulations (Milestone 4)."
    )
    parser.add_argument(
        "--mode",
        choices=("preview", "production"),
        default="preview",
        help="preview=960x540 @ 30 FPS (default); production=1920x1080 @ 60 FPS.",
    )
    parser.add_argument(
        "--speeds",
        type=float,
        nargs="+",
        default=None,
        help="Optional subset of speeds in km/h (defaults to all 7 speeds in simulation_config.json).",
    )
    parser.add_argument(
        "--blender",
        default=None,
        help="Optional explicit path to blender.exe.",
    )
    parser.add_argument(
        "--motion-blur",
        action="store_true",
        help="Enable Blender shutter motion blur.",
    )
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="Validate trajectories & visibility without encoding MP4 videos.",
    )
    args = parser.parse_args()

    sim_cfg = load_json(PROJECT_ROOT / "config" / "simulation_config.json")
    blender_exe = find_blender_executable(args.blender, sim_cfg)
    if not blender_exe:
        print("ERROR: Blender executable not found.", file=sys.stderr)
        return 1

    speeds = args.speeds if args.speeds else list(sim_cfg["simulation"]["speeds_kmh"])
    temp_dir = PROJECT_ROOT / sim_cfg["paths"].get("temp_dir", "results/_tmp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["TEMP"] = str(temp_dir)
    env["TMP"] = str(temp_dir)
    env["TMPDIR"] = str(temp_dir)

    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    run_meta_dir = metrics_dir / "m4_speed_runs"
    run_meta_dir.mkdir(parents=True, exist_ok=True)
    contact_dir = PROJECT_ROOT / "results" / "videos" / "m4_speed_runs" / "contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_meta_dir / "run_simulation.log"

    preserved_before = _snapshot_preserved_blends()
    blender_main = PROJECT_ROOT / "simulation" / "blender" / "main.py"
    suffix = "_preview" if args.mode == "preview" else ""

    run_summaries: list[dict[str, Any]] = []
    visibility_report: dict[str, Any] = {}
    perf_entries: list[dict[str, Any]] = []
    all_ok = True
    failed_speeds: list[Any] = []

    t_batch_start = time.perf_counter()

    with log_path.open("w", encoding="utf-8") as log_handle:
        for spd in speeds:
            spd_tag = int(spd) if float(spd).is_integer() else spd
            cmd = [
                blender_exe,
                "-b",
                "--factory-startup",
                "--python-exit-code",
                "1",
                "--python",
                str(blender_main),
                "--",
                "--milestone",
                "m4",
                "--mode",
                args.mode,
                "--speed",
                str(spd_tag),
            ]
            if args.motion_blur:
                cmd.append("--motion-blur")
            if args.skip_render:
                cmd.append("--skip-render")

            banner = f"[M4] Starting {spd_tag} km/h ({args.mode})..."
            print(banner)
            log_handle.write(banner + "\n")
            log_handle.flush()

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
            log_handle.write(f"\n[M4] Exit code for {spd_tag} km/h: {proc.returncode}\n\n")
            log_handle.flush()

            meta_path = run_meta_dir / f"railway_{spd_tag}kmh{suffix}_metadata.json"
            if proc.returncode != 0 or not meta_path.is_file():
                all_ok = False
                failed_speeds.append({"speed_kmh": spd_tag, "returncode": proc.returncode})
                print(f"[M4] FAILED {spd_tag} km/h (rc={proc.returncode})")
                continue

            meta = load_json(meta_path)
            video_path = Path(meta["video_path"]) if meta.get("video_path") else None
            cv_check = (
                _verify_video_with_opencv(video_path, meta)
                if (not args.skip_render and video_path is not None)
                else {"passed": args.skip_render, "skipped": args.skip_render}
            )
            contact_path = (
                _build_contact_sheet(
                    video_path,
                    meta,
                    contact_dir / f"contact_sheet_{spd_tag}kmh.png",
                )
                if (not args.skip_render and video_path is not None)
                else None
            )

            speed_passed = bool(meta["validation"]["all_passed"] and cv_check["passed"])
            if not speed_passed:
                all_ok = False
                failed_speeds.append({"speed_kmh": spd_tag, "validation_failed": True})

            visibility_report[f"{spd_tag}kmh"] = {
                "speed_kmh": meta["speed_kmh"],
                "speed_mps": meta["speed_mps"],
                "fps": meta["fps"],
                "distance_per_frame_m": meta["distance_per_frame_m"],
                "frame_count": meta["frame_count"],
                "anomalies": meta["visibility_summary"],
            }

            ren_cfg = meta["blender_render_configuration"]
            perf_entries.append(
                {
                    "speed_kmh": meta["speed_kmh"],
                    "frame_count": meta["frame_count"],
                    "render_time_seconds": ren_cfg["render_time_seconds"],
                    "total_elapsed_seconds": ren_cfg["total_elapsed_seconds"],
                    "ms_per_frame": ren_cfg["ms_per_frame"],
                    "video_size_bytes": meta["video_size_bytes"],
                    "video_size_mb": round(meta["video_size_bytes"] / (1024 * 1024), 3),
                }
            )

            run_summaries.append(
                {
                    "speed_kmh": meta["speed_kmh"],
                    "passed": speed_passed,
                    "metadata_path": str(meta_path),
                    "contact_sheet_path": contact_path,
                    "blender_validation": meta["validation"],
                    "opencv_validation": cv_check,
                }
            )
            print(
                f"[M4] Completed {spd_tag} km/h: passed={speed_passed}, "
                f"frames={meta['frame_count']}, render={ren_cfg['render_time_seconds']}s, "
                f"size={meta['video_size_bytes']} B"
            )

    preserved_after = _snapshot_preserved_blends()
    preserved_checks = {
        rel: {
            "passed": (preserved_before[rel] is not None)
            and (preserved_before[rel] == preserved_after[rel]),
            "sha256_before": preserved_before[rel],
            "sha256_after": preserved_after[rel],
        }
        for rel in PRESERVED_BLEND_FILES
    }
    all_preserved_ok = all(v["passed"] for v in preserved_checks.values())
    if not all_preserved_ok:
        all_ok = False

    total_batch_s = round(time.perf_counter() - t_batch_start, 2)
    total_frames_rendered = sum(p["frame_count"] for p in perf_entries)
    total_render_s = round(sum(p["render_time_seconds"] for p in perf_entries), 2)
    total_size_mb = round(sum(p["video_size_mb"] for p in perf_entries), 3)
    avg_ms_per_frame = round(
        (total_render_s * 1000.0) / max(1, total_frames_rendered), 2
    )

    # Save visibility analysis
    vis_out_path = metrics_dir / "m4_visibility_analysis.json"
    with vis_out_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "data_label": "SYNTHETIC",
                "milestone": "M4_SPEED_SIMULATION",
                "mode": args.mode,
                "near_ground_blind_zone_note": (
                    "At 1.85m height, -7.5 deg pitch, and 35mm lens (36x20.25mm sensor), "
                    "the lower frustum edge intersects the rail head at ~4.23m ahead of the camera. "
                    "At higher speeds (larger m/frame step), narrow rail-surface defects like A3 "
                    "experience fewer sufficiently visible frames before exiting the near-ground blind zone."
                ),
                "speeds": visibility_report,
            },
            handle,
            indent=2,
        )

    # Save render performance report
    perf_out_path = metrics_dir / "m4_render_performance.json"
    with perf_out_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "data_label": "SYNTHETIC",
                "milestone": "M4_SPEED_SIMULATION",
                "mode": args.mode,
                "sequential_execution": True,
                "total_batch_wall_seconds": total_batch_s,
                "total_blender_render_seconds": total_render_s,
                "total_frames_across_runs": total_frames_rendered,
                "mean_ms_per_frame": avg_ms_per_frame,
                "total_video_size_mb": total_size_mb,
                "per_speed": perf_entries,
            },
            handle,
            indent=2,
        )

    # Save master M4 validation report
    val_out_path = metrics_dir / "m4_validation_report.json"
    master_validation = {
        "data_label": "SYNTHETIC",
        "milestone": "M4_SPEED_SIMULATION",
        "mode": args.mode,
        "speeds_requested": speeds,
        "speeds_completed": len(run_summaries),
        "failed_speeds": failed_speeds,
        "all_seven_speeds_present": len(run_summaries) == 7 and len(speeds) == 7,
        "preserved_scenes_unchanged": all_preserved_ok,
        "preserved_scenes_checks": preserved_checks,
        "overall_passed": bool(all_ok and len(failed_speeds) == 0),
        "runs": run_summaries,
        "visibility_analysis_path": str(vis_out_path),
        "render_performance_path": str(perf_out_path),
        "log_path": str(log_path),
    }
    with val_out_path.open("w", encoding="utf-8") as handle:
        json.dump(master_validation, handle, indent=2)

    print("\n" + "=" * 68)
    print(f"ITMS-VirtualTrack — Milestone 4 Validation ({args.mode}, SYNTHETIC)")
    print("=" * 68)
    print(f"Speeds completed           : {len(run_summaries)}/{len(speeds)}")
    print(f"Preserved M2/M3 scenes OK  : {all_preserved_ok}")
    print(f"Total frames               : {total_frames_rendered}")
    print(f"Total batch wall time      : {total_batch_s} s")
    print(f"Total video disk size      : {total_size_mb} MB")
    print(f"Overall Milestone 4 Passed : {master_validation['overall_passed']}")
    print(f"Validation report          : {val_out_path}")
    print("=" * 68)

    return 0 if master_validation["overall_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
