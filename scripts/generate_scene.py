"""Host Python CLI to generate, render, and validate the Milestone 2 railway scene.

Invokes Blender headlessly, keeps temp/cache directories on the workspace drive (D:),
and runs host-side OpenCV validation on the rendered preview outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from check_environment import find_blender_executable, load_json  # noqa: E402


def verify_rendered_outputs(
    project_root: Path,
    metadata: dict[str, Any],
    expect_video: bool,
    expect_keyframes: bool,
    expect_blend: bool,
) -> dict[str, Any]:
    """Verify .blend, preview MP4, and PNG keyframes exist and contain valid imagery."""
    outputs = metadata["outputs"]
    ren_cfg = metadata["camera_configuration"]["render_settings"]
    expected_w = int(ren_cfg["width"])
    expected_h = int(ren_cfg["height"])
    expected_fps = float(ren_cfg["fps"])

    checks: dict[str, Any] = {}
    all_ok = True

    def record(name: str, passed: bool, details: dict[str, Any]) -> None:
        nonlocal all_ok
        checks[name] = {"passed": passed, **details}
        if not passed:
            all_ok = False

    if expect_blend:
        bpath = Path(outputs["blend_file"]) if outputs.get("blend_file") else None
        ok = bool(bpath and bpath.is_file() and bpath.stat().st_size > 10_000)
        record(
            "blend_file_exists",
            ok,
            {
                "path": str(bpath) if bpath else None,
                "size_bytes": bpath.stat().st_size if bpath and bpath.is_file() else 0,
            },
        )

    if expect_keyframes:
        kf_list = outputs.get("preview_keyframes", [])
        kf_stats: list[dict[str, Any]] = []
        kf_ok = len(kf_list) > 0
        for p_str in kf_list:
            p = Path(p_str)
            if not p.is_file():
                kf_ok = False
                kf_stats.append({"path": p_str, "exists": False})
                continue
            img = cv2.imread(str(p))
            if img is None:
                kf_ok = False
                kf_stats.append({"path": p_str, "exists": True, "readable": False})
                continue
            h, w = img.shape[:2]
            std_val = float(np.std(img))
            valid_img = (w == expected_w) and (h == expected_h) and (std_val > 5.0)
            if not valid_img:
                kf_ok = False
            kf_stats.append(
                {
                    "path": p_str,
                    "size_bytes": p.stat().st_size,
                    "resolution": [w, h],
                    "pixel_std": round(std_val, 2),
                    "valid": valid_img,
                }
            )
        record("preview_keyframes_valid", kf_ok, {"keyframes": kf_stats})

    if expect_video:
        v_str = outputs.get("preview_video")
        v_path = Path(v_str) if v_str else None
        if not v_path or not v_path.is_file():
            record("preview_video_valid", False, {"path": v_str, "exists": False})
        else:
            cap = cv2.VideoCapture(str(v_path))
            opened = cap.isOpened()
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
            fps = float(cap.get(cv2.CAP_PROP_FPS)) if opened else 0.0
            vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
            vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0
            ret, frame0 = cap.read() if opened else (False, None)
            cap.release()
            frame_std = float(np.std(frame0)) if (ret and frame0 is not None) else 0.0
            vid_ok = (
                opened
                and frame_count >= 10
                and vw == expected_w
                and vh == expected_h
                and abs(fps - expected_fps) < 0.5
                and frame_std > 5.0
            )
            record(
                "preview_video_valid",
                vid_ok,
                {
                    "path": str(v_path),
                    "size_bytes": v_path.stat().st_size,
                    "resolution": [vw, vh],
                    "fps": round(fps, 3),
                    "frame_count": frame_count,
                    "first_frame_std": round(frame_std, 2),
                },
            )

    return {"all_passed": all_ok, "checks": checks}


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_m3_outputs(
    metadata: dict[str, Any],
    base_blend_path: Path,
    base_hash_before: str | None,
) -> dict[str, Any]:
    """Verify all M3 .blend files, preview images, segmentation masks, and generate GT overlays."""
    checks: dict[str, Any] = {}
    all_ok = True

    def record(name: str, passed: bool, details: dict[str, Any]) -> None:
        nonlocal all_ok
        checks[name] = {"passed": passed, **details}
        if not passed:
            all_ok = False

    base_hash_after = _sha256_file(base_blend_path)
    preserved_ok = (base_hash_before is not None) and (base_hash_before == base_hash_after)
    record(
        "base_blend_preserved_unchanged",
        preserved_ok,
        {
            "path": str(base_blend_path),
            "sha256_before": base_hash_before,
            "sha256_after": base_hash_after,
        },
    )

    overlay_paths: list[str] = []
    for variant_name, vinfo in metadata["variants"].items():
        bpath = Path(vinfo["blend_file"])
        b_ok = bpath.is_file() and bpath.stat().st_size > 10_000
        record(
            f"{variant_name}_blend_exists",
            b_ok,
            {"path": str(bpath), "size_bytes": bpath.stat().st_size if bpath.is_file() else 0},
        )

        for view in vinfo["inspection_views"]:
            img_path = Path(view["image_path"])
            img = cv2.imread(str(img_path)) if img_path.is_file() else None
            img_ok = img is not None and float(np.std(img)) > 5.0
            record(
                f"{variant_name}_{img_path.stem}_render",
                img_ok,
                {
                    "path": str(img_path),
                    "resolution": [img.shape[1], img.shape[0]] if img is not None else None,
                    "pixel_std": round(float(np.std(img)), 2) if img is not None else 0.0,
                },
            )

            mask_str = view.get("mask_path")
            mask_img = None
            if mask_str:
                mpath = Path(mask_str)
                mask_img = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE) if mpath.is_file() else None
                fg_pixels = int(np.count_nonzero(mask_img > 127)) if mask_img is not None else 0
                mask_ok = mask_img is not None and fg_pixels >= 25
                record(
                    f"{variant_name}_{mpath.stem}_valid",
                    mask_ok,
                    {
                        "path": str(mpath),
                        "foreground_pixels": fg_pixels,
                    },
                )

            # Save annotated overlay image for visual ground-truth verification
            if img is not None and view["projections"]:
                overlay = img.copy()
                if mask_img is not None:
                    mask_bool = mask_img > 127
                    overlay[mask_bool] = (
                        overlay[mask_bool] * 0.45 + np.array([0, 220, 255]) * 0.55
                    ).astype(np.uint8)
                for proj in view["projections"]:
                    if not proj.get("visible") or not proj.get("bbox"):
                        continue
                    x1, y1, x2, y2 = [int(round(v)) for v in proj["bbox"]]
                    color = (0, 230, 80) if proj["yolo_enabled"] else (0, 140, 255)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                    label = f"{proj['id']} {proj['class']} ({proj['camera_distance_m']:.1f}m)"
                    cv2.putText(
                        overlay,
                        label,
                        (max(4, x1), max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                        cv2.LINE_AA,
                    )
                ov_path = img_path.with_name(img_path.stem + "_gt_overlay.png")
                cv2.imwrite(str(ov_path), overlay)
                overlay_paths.append(str(ov_path))

    return {"all_passed": all_ok, "checks": checks, "overlay_images": overlay_paths}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build, validate, and render ITMS-VirtualTrack railway scenes."
    )
    parser.add_argument(
        "--milestone",
        choices=("m2", "m3"),
        default="m2",
        help="m2 = base 250m track preview; m3 = baseline + individual + combined anomaly scenes.",
    )
    parser.add_argument(
        "--mode",
        choices=("preview", "production"),
        default="preview",
        help="Render preset (default: preview 960x540).",
    )
    parser.add_argument(
        "--blender",
        default=None,
        help="Optional explicit path to blender.exe.",
    )
    parser.add_argument("--no-video", action="store_true", help="Skip MP4 preview video (m2).")
    parser.add_argument("--no-keyframes", action="store_true", help="Skip PNG keyframes (m2).")
    parser.add_argument("--no-blend", action="store_true", help="Skip saving .blend file (m2).")
    args = parser.parse_args()

    sim_cfg = load_json(PROJECT_ROOT / "config" / "simulation_config.json")
    ren_cfg = load_json(PROJECT_ROOT / "config" / "rendering_config.json")
    blender_exe = find_blender_executable(args.blender, sim_cfg)
    if not blender_exe:
        print("ERROR: Blender executable not found. Run scripts/check_environment.py.", file=sys.stderr)
        return 1

    temp_dir = PROJECT_ROOT / sim_cfg["paths"].get("temp_dir", "results/_tmp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["TEMP"] = str(temp_dir)
    env["TMP"] = str(temp_dir)
    env["TMPDIR"] = str(temp_dir)

    base_blend_path = PROJECT_ROOT / ren_cfg["output"].get(
        "blend_filename", "results/scenes/railway_250m_base.blend"
    )
    base_hash_before = _sha256_file(base_blend_path) if args.milestone == "m3" else None

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
        "--milestone",
        args.milestone,
        "--mode",
        args.mode,
    ]
    if args.milestone == "m2":
        if args.no_video:
            cmd.append("--no-video")
        if args.no_keyframes:
            cmd.append("--no-keyframes")
        if args.no_blend:
            cmd.append("--no-blend")

    print(f"[ITMS-VirtualTrack] Launching Blender ({args.milestone}): {' '.join(cmd)}")
    proc = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT), check=False)
    if proc.returncode != 0:
        print(f"ERROR: Blender exited with code {proc.returncode}", file=sys.stderr)
        return proc.returncode

    metrics_dir = PROJECT_ROOT / sim_cfg["paths"].get("results_dir", "results") / "metrics"

    if args.milestone == "m3":
        metadata = load_json(metrics_dir / "m3_anomaly_metadata.json")
        blender_val = load_json(metrics_dir / "m3_blender_validation.json")
        output_val = verify_m3_outputs(metadata, base_blend_path, base_hash_before)
        combined_report = {
            "data_label": "SYNTHETIC",
            "milestone": "M3_ANOMALY_GENERATION",
            "overall_passed": bool(
                blender_val["blender_all_passed"] and output_val["all_passed"]
            ),
            "blender_variants_validation": blender_val["variants"],
            "output_validation": output_val,
        }
        report_path = metrics_dir / "m3_validation_report.json"
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(combined_report, handle, indent=2)

        print("\n" + "=" * 68)
        print("ITMS-VirtualTrack — Milestone 3 Validation Report (SYNTHETIC)")
        print("=" * 68)
        for vname, vval in blender_val["variants"].items():
            print(f"[variant:{vname}] passed={vval['all_passed']}")
            for section in ("geometry_validation", "camera_validation", "anomaly_validation"):
                for cname, cdet in vval[section]["checks"].items():
                    print(f"  - {cname:<32} : {'PASS' if cdet.get('passed') else 'FAIL'}")
        print(f"[output_validation] passed={output_val['all_passed']}")
        for cname, cdet in output_val["checks"].items():
            print(f"  - {cname:<36} : {'PASS' if cdet.get('passed') else 'FAIL'}")
        print(f"Overall Milestone 3 Validation : {'PASS' if combined_report['overall_passed'] else 'FAIL'}")
        print(f"Report saved to: {report_path}")
        print("=" * 68)
        return 0 if combined_report["overall_passed"] else 2

    metadata = load_json(metrics_dir / "m2_scene_metadata.json")
    blender_val = load_json(metrics_dir / "m2_blender_validation.json")

    output_val = verify_rendered_outputs(
        PROJECT_ROOT,
        metadata=metadata,
        expect_video=not args.no_video,
        expect_keyframes=not args.no_keyframes,
        expect_blend=not args.no_blend,
    )

    combined_report = {
        "data_label": "SYNTHETIC",
        "milestone": "M2_BASIC_RAILWAY_SIMULATION",
        "overall_passed": bool(
            blender_val["blender_all_passed"] and output_val["all_passed"]
        ),
        "geometry_validation": blender_val["geometry_validation"],
        "camera_validation": blender_val["camera_validation"],
        "output_validation": output_val,
    }

    report_path = metrics_dir / "m2_validation_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(combined_report, handle, indent=2)

    print("\n" + "=" * 68)
    print("ITMS-VirtualTrack — Milestone 2 Validation Report (SYNTHETIC)")
    print("=" * 68)
    for section in ("geometry_validation", "camera_validation", "output_validation"):
        print(f"[{section}] passed={combined_report[section]['all_passed']}")
        for check_name, details in combined_report[section]["checks"].items():
            status = "PASS" if details.get("passed") else "FAIL"
            print(f"  - {check_name:<30} : {status}")
    print(f"Overall Milestone 2 Validation : {'PASS' if combined_report['overall_passed'] else 'FAIL'}")
    print(f"Report saved to: {report_path}")
    print("=" * 68)

    return 0 if combined_report["overall_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

