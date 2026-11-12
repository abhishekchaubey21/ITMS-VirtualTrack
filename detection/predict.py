"""Run frozen hybrid detection models across all 7 controlled evaluation speeds & test variants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detection.geometry_analyzer import reconstruct_and_evaluate_track_profile  # noqa: E402
from detection.inference_pipeline import HybridRailwayDetector, annotate_frame  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_video_inference(
    detector: HybridRailwayDetector,
    video_id: str,
    frame_gt_path: Path,
    geom_gt_path: Path,
    dataset_root: Path,
    out_video_path: Path | None,
    out_json_path: Path,
    save_crack_overlays_dir: Path | None = None,
) -> dict[str, Any]:
    """Run frozen detector frame-by-frame on a video's extracted frames and save predictions."""
    frame_gt = _load_json(frame_gt_path)
    geom_gt = _load_json(geom_gt_path)
    frames_list = frame_gt["frames"]
    speed_kmh = frame_gt["train_speed_kmh"]
    fps = float(frame_gt.get("fps", 30))

    writer = None
    if out_video_path is not None and frames_list:
        out_video_path.parent.mkdir(parents=True, exist_ok=True)
        w = int(frames_list[0]["image_width"])
        h = int(frames_list[0]["image_height"])
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (w, h))

    frame_predictions: list[dict[str, Any]] = []
    geom_frame_results: list[dict[str, Any]] = []
    crack_eval_records: list[dict[str, Any]] = []

    for f_rec in frames_list:
        img_path = dataset_root / f_rec["image_path"]
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Unreadable frame image: {img_path}")

        f_num = int(f_rec["frame_number"])
        cam_y = float(f_rec["camera_position_m"])
        pred = detector.predict_frame(img, camera_position_m=cam_y)

        if writer is not None:
            vis = annotate_frame(img, pred, f_num, speed_kmh, cam_y)
            writer.write(vis)

        crack = pred["crack_prediction"]
        pred_mask = crack.get("mask")

        # Check GT crack mask alignment/metrics for this frame (evaluation record)
        a3_gt = next((a for a in f_rec["anomalies"] if a["anomaly_id"] == "A3"), None)
        gt_mask_rel = a3_gt.get("segmentation_mask") if a3_gt else None
        gt_mask = None
        if gt_mask_rel:
            gt_mask = cv2.imread(str(dataset_root / gt_mask_rel), cv2.IMREAD_GRAYSCALE)

        if pred_mask is not None or gt_mask is not None:
            p_bin = (pred_mask > 127) if pred_mask is not None else np.zeros((img.shape[0], img.shape[1]), dtype=bool)
            g_bin = (gt_mask > 127) if gt_mask is not None else np.zeros((img.shape[0], img.shape[1]), dtype=bool)
            tp_px = int(np.logical_and(p_bin, g_bin).sum())
            fp_px = int(np.logical_and(p_bin, np.logical_not(g_bin)).sum())
            fn_px = int(np.logical_and(np.logical_not(p_bin), g_bin).sum())
            if tp_px + fp_px + fn_px > 0:
                crack_eval_records.append(
                    {
                        "frame_number": f_num,
                        "camera_position_m": cam_y,
                        "camera_distance_m": a3_gt["camera_to_anomaly_distance_m"] if a3_gt else None,
                        "gt_visible": bool(a3_gt["visible"]) if a3_gt else False,
                        "pred_detected": bool(crack["detected"]),
                        "tp_pixels": tp_px,
                        "fp_pixels": fp_px,
                        "fn_pixels": fn_px,
                    }
                )

            if save_crack_overlays_dir is not None and (gt_mask is not None or crack["detected"]):
                save_crack_overlays_dir.mkdir(parents=True, exist_ok=True)
                ov = img.copy()
                if gt_mask is not None:
                    ov[g_bin] = (ov[g_bin] * 0.3 + np.array([0, 255, 0]) * 0.7).astype(np.uint8)
                if pred_mask is not None and crack["detected"]:
                    ov[p_bin] = (ov[p_bin] * 0.3 + np.array([0, 0, 255]) * 0.7).astype(np.uint8)
                cv2.putText(
                    ov,
                    f"{video_id} F{f_num:04d} | GT(green)={int(g_bin.sum())}px Pred(red)={int(p_bin.sum())}px",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imwrite(str(save_crack_overlays_dir / f"{video_id}_f{f_num:04d}_crack_cmp.png"), ov)

        geom = pred["geometry_prediction"]
        geom_frame_results.append(geom)

        frame_predictions.append(
            {
                "frame_number": f_num,
                "timestamp_s": float(f_rec["timestamp_s"]),
                "camera_position_m": cam_y,
                "image_path": f_rec["image_path"],
                "yolo_detections": pred["yolo_detections"],
                "crack_prediction": {
                    "detected": crack["detected"],
                    "confidence": crack["confidence"],
                    "pixel_area": crack["pixel_area"],
                    "bbox": crack["bbox"],
                },
                "geometry_prediction": {
                    "detected": geom["detected"],
                    "confidence": geom["confidence"],
                    "max_estimated_deviation_m": geom["max_estimated_deviation_m"],
                    "affected_interval_m": geom["affected_interval_m"],
                    "bbox": geom["bbox"],
                    "valid_scanlines": geom["valid_scanlines"],
                },
                "latencies_ms": pred["latencies_ms"],
            }
        )

    if writer is not None:
        writer.release()

    # Track-level A4 geometry reconstruction evaluated against 0.25m GT profile
    geom_profile_eval = reconstruct_and_evaluate_track_profile(
        geom_frame_results, geom_gt
    )

    out_payload = {
        "dataset_type": "SYNTHETIC",
        "milestone": "M6_PREDICTIONS",
        "video_id": video_id,
        "dataset_group": frame_gt["dataset_group"],
        "split": frame_gt["split"],
        "train_speed_kmh": speed_kmh,
        "fps": fps,
        "frame_count": len(frame_predictions),
        "annotated_video_path": str(out_video_path) if out_video_path else None,
        "crack_pixel_eval_frames": crack_eval_records,
        "geometry_track_evaluation": geom_profile_eval,
        "frames": frame_predictions,
    }
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    return out_payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run frozen Milestone 6 models on controlled speeds and test variants."
    )
    parser.add_argument(
        "--yolo-weights",
        default="results/models/yolo/baseline_v1/weights/best.pt",
    )
    parser.add_argument(
        "--crack-weights",
        default="results/models/crack_unet/best_crack_unet.pt",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    sim_cfg = _load_json(PROJECT_ROOT / "config" / "simulation_config.json")
    cam_cfg = _load_json(PROJECT_ROOT / "config" / "camera_config.json")
    dataset_root = PROJECT_ROOT / "dataset" / "synthetic"
    manifest = _load_json(dataset_root / "metadata" / "dataset_manifest.json")

    detector = HybridRailwayDetector(
        yolo_weights_path=PROJECT_ROOT / args.yolo_weights,
        crack_unet_path=PROJECT_ROOT / args.crack_weights,
        cam_cfg=cam_cfg,
        sim_cfg=sim_cfg,
        yolo_conf=args.conf,
        imgsz=args.imgsz,
    )

    pred_vid_dir = PROJECT_ROOT / "results" / "videos" / "m6_predictions"
    pred_json_dir = PROJECT_ROOT / "results" / "metrics" / "m6_predictions"
    crack_ov_dir = pred_vid_dir / "crack_overlays"

    for v_entry in manifest["videos"]:
        vid = v_entry["video_id"]
        split = v_entry["split"]
        grp = v_entry["dataset_group"]
        if split != "test":
            continue

        frame_gt_path = dataset_root / "ground_truth" / "frame_annotations" / f"{vid}_frames.json"
        geom_gt_path = dataset_root / "ground_truth" / "geometry_profiles" / f"{vid}_a4_geometry.json"

        if grp == "controlled_evaluation":
            spd = v_entry["speed_kmh"]
            out_vid = pred_vid_dir / f"pred_railway_{spd}kmh.mp4"
            out_json = pred_json_dir / f"pred_{vid}.json"
            ov_dir = crack_ov_dir
        else:
            out_vid = pred_vid_dir / f"pred_{vid}.mp4"
            out_json = pred_json_dir / f"pred_{vid}.json"
            ov_dir = None

        print(f"[M6] Running frozen inference on {vid} ({grp})...")
        run_video_inference(
            detector=detector,
            video_id=vid,
            frame_gt_path=frame_gt_path,
            geom_gt_path=geom_gt_path,
            dataset_root=dataset_root,
            out_video_path=out_vid,
            out_json_path=out_json,
            save_crack_overlays_dir=ov_dir,
        )

    print("[M6] Inference complete across all 7 controlled speeds and 3 test variants.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

