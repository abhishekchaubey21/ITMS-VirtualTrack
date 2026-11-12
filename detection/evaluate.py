"""Milestone 6 evaluation suite: frame-level, event-level, A3 seg, and A4 geometry metrics.

Strictly evaluates the 7 controlled evaluation speeds separately (30..200 km/h)
and reports the 3 independent test variants separately (no metric mixing).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    return inter / max(1e-9, area_a + area_b - inter)


def _compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate_yolo_video(
    pred_data: dict[str, Any],
    gt_data: dict[str, Any],
    iou_thresh: float = 0.50,
) -> dict[str, Any]:
    """Compute frame-level and anomaly-event-level YOLO metrics for A1 & A2."""
    gt_frames = {int(f["frame_number"]): f for f in gt_data["frames"]}
    pred_frames = pred_data["frames"]

    # Collect per-class predictions and ground truths across frames for AP calculation
    preds_by_cls: dict[int, list[tuple[int, float, list[float]]]] = {0: [], 1: []}
    gts_by_cls_frame: dict[int, dict[int, list[list[float]]]] = {0: {}, 1: {}}

    tp_50 = 0
    fp_50 = 0
    fn_50 = 0
    latencies_yolo: list[float] = []
    latencies_total: list[float] = []

    # Track anomaly-event discovery per ID (A1 -> class 0, A2 -> class 1)
    events: dict[str, dict[str, Any]] = {
        "A1": {
            "class_id": 0,
            "class_name": "animal_carcass",
            "position_m": 50.0,
            "first_visible_frame": None,
            "first_visible_timestamp_s": None,
            "first_detect_frame": None,
            "first_detect_timestamp_s": None,
            "first_detect_distance_m": None,
            "matched_frames": 0,
            "visible_gt_frames": 0,
        },
        "A2": {
            "class_id": 1,
            "class_name": "grain_spillage",
            "position_m": 100.0,
            "first_visible_frame": None,
            "first_visible_timestamp_s": None,
            "first_detect_frame": None,
            "first_detect_timestamp_s": None,
            "first_detect_distance_m": None,
            "matched_frames": 0,
            "visible_gt_frames": 0,
        },
    }

    fp_outside_event_frames = 0

    for pf in pred_frames:
        f_num = int(pf["frame_number"])
        ts = float(pf["timestamp_s"])
        cam_y = float(pf["camera_position_m"])
        latencies_yolo.append(float(pf["latencies_ms"]["yolo_ms"]))
        latencies_total.append(float(pf["latencies_ms"]["total_ms"]))

        gf = gt_frames[f_num]
        gt_boxes_frame: list[tuple[int, str, list[float]]] = []
        for a in gf["anomalies"]:
            if a["anomaly_id"] in ("A1", "A2") and a["visible"] and a.get("bbox"):
                aid = a["anomaly_id"]
                cid = int(a["class_id"])
                gt_boxes_frame.append((cid, aid, [float(v) for v in a["bbox"]]))
                gts_by_cls_frame[cid].setdefault(f_num, []).append([float(v) for v in a["bbox"]])

                ev = events[aid]
                ev["visible_gt_frames"] += 1
                if ev["first_visible_frame"] is None:
                    ev["first_visible_frame"] = f_num
                    ev["first_visible_timestamp_s"] = ts

        p_boxes_frame = [
            (int(d["class_id"]), float(d["confidence"]), [float(v) for v in d["bbox"]])
            for d in pf["yolo_detections"]
        ]
        for cid, conf, box in p_boxes_frame:
            if cid in preds_by_cls:
                preds_by_cls[cid].append((f_num, conf, box))

        # Greedy IoU matching per class at iou_thresh
        matched_gt: set[int] = set()
        for cid, _conf, pbox in sorted(p_boxes_frame, key=lambda x: -x[1]):
            best_iou = 0.0
            best_gi = -1
            for gi, (gcid, _aid, gbox) in enumerate(gt_boxes_frame):
                if gi in matched_gt or gcid != cid:
                    continue
                iou = _bbox_iou(pbox, gbox)
                if iou >= iou_thresh and iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            if best_gi >= 0:
                matched_gt.add(best_gi)
                tp_50 += 1
                aid = gt_boxes_frame[best_gi][1]
                ev = events[aid]
                ev["matched_frames"] += 1
                if ev["first_detect_frame"] is None:
                    ev["first_detect_frame"] = f_num
                    ev["first_detect_timestamp_s"] = ts
                    ev["first_detect_distance_m"] = round(ev["position_m"] - cam_y, 4)
            else:
                fp_50 += 1
                if not gt_boxes_frame:
                    fp_outside_event_frames += 1

        fn_50 += len(gt_boxes_frame) - len(matched_gt)

    prec = tp_50 / max(1, tp_50 + fp_50)
    rec = tp_50 / max(1, tp_50 + fn_50)
    f1 = (2.0 * prec * rec) / max(1e-9, prec + rec) if (prec + rec) > 0 else 0.0

    # Compute mAP50 and mAP50-95 directly over frame GT
    iou_thresholds = [round(0.50 + 0.05 * k, 2) for k in range(10)]
    ap_per_thresh: dict[float, list[float]] = {t: [] for t in iou_thresholds}
    for cid in (0, 1):
        preds_sorted = sorted(preds_by_cls[cid], key=lambda x: -x[1])
        total_gts = sum(len(v) for v in gts_by_cls_frame[cid].values())
        if total_gts == 0:
            continue
        for t in iou_thresholds:
            used: dict[int, set[int]] = {}
            tp_arr = np.zeros(len(preds_sorted), dtype=np.float64)
            fp_arr = np.zeros(len(preds_sorted), dtype=np.float64)
            for idx, (f_num, _conf, pbox) in enumerate(preds_sorted):
                gt_list = gts_by_cls_frame[cid].get(f_num, [])
                best_iou = 0.0
                best_gi = -1
                used_f = used.setdefault(f_num, set())
                for gi, gbox in enumerate(gt_list):
                    if gi in used_f:
                        continue
                    iou = _bbox_iou(pbox, gbox)
                    if iou >= t and iou > best_iou:
                        best_iou = iou
                        best_gi = gi
                if best_gi >= 0:
                    used_f.add(best_gi)
                    tp_arr[idx] = 1.0
                else:
                    fp_arr[idx] = 1.0
            tp_cum = np.cumsum(tp_arr)
            fp_cum = np.cumsum(fp_arr)
            recalls = tp_cum / total_gts
            precisions = tp_cum / np.maximum(1e-9, tp_cum + fp_cum)
            ap_per_thresh[t].append(_compute_ap(recalls, precisions))

    map50 = float(np.mean(ap_per_thresh[0.50])) if ap_per_thresh[0.50] else 0.0
    map50_95 = float(
        np.mean([np.mean(ap_per_thresh[t]) for t in iou_thresholds if ap_per_thresh[t]])
    )

    # Event-level summary (no duplicate frame discovery counting)
    detected_events = 0
    missed_events = 0
    det_distances: list[float] = []
    det_latencies_s: list[float] = []
    for aid, ev in events.items():
        found = ev["first_detect_frame"] is not None
        ev["detected"] = found
        if found:
            detected_events += 1
            det_distances.append(float(ev["first_detect_distance_m"]))
            if ev["first_visible_timestamp_s"] is not None:
                lat_s = max(
                    0.0, float(ev["first_detect_timestamp_s"]) - float(ev["first_visible_timestamp_s"])
                )
                ev["detection_latency_s"] = round(lat_s, 4)
                det_latencies_s.append(lat_s)
        else:
            missed_events += 1
            ev["detection_latency_s"] = None

    return {
        "frame_level": {
            "tp": tp_50,
            "fp": fp_50,
            "fn": fn_50,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "mAP50": round(map50, 4),
            "mAP50_95": round(map50_95, 4),
            "fp_outside_positive_frames": fp_outside_event_frames,
            "mean_yolo_inference_ms": round(float(np.mean(latencies_yolo)), 3),
            "mean_pipeline_inference_ms": round(float(np.mean(latencies_total)), 3),
        },
        "event_level": {
            "total_gt_events": 2,
            "detected_events": detected_events,
            "missed_events": missed_events,
            "event_recall": round(detected_events / 2.0, 4),
            "mean_first_detection_distance_m": round(float(np.mean(det_distances)), 3)
            if det_distances
            else None,
            "mean_detection_latency_s": round(float(np.mean(det_latencies_s)), 4)
            if det_latencies_s
            else None,
            "per_anomaly": events,
        },
    }


def evaluate_crack_video(pred_data: dict[str, Any]) -> dict[str, Any]:
    """Compute A3 pixel Dice, IoU, precision, recall, and detection distance."""
    recs = pred_data.get("crack_pixel_eval_frames", [])
    tp = sum(r["tp_pixels"] for r in recs)
    fp = sum(r["fp_pixels"] for r in recs)
    fn = sum(r["fn_pixels"] for r in recs)

    dice = (2.0 * tp) / max(1, 2 * tp + fp + fn)
    iou = tp / max(1, tp + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)

    gt_pos_frames = [r for r in recs if r["gt_visible"]]
    hit_frames = [r for r in gt_pos_frames if r["tp_pixels"] > 0]
    first_hit_dist = (
        max(float(r["camera_distance_m"]) for r in hit_frames if r["camera_distance_m"] is not None)
        if hit_frames
        else None
    )

    return {
        "gt_visible_crack_frames": len(gt_pos_frames),
        "frames_with_tp_crack_pixels": len(hit_frames),
        "event_detected": len(hit_frames) > 0,
        "first_detection_distance_m": round(first_hit_dist, 3) if first_hit_dist is not None else None,
        "tp_pixels": tp,
        "fp_pixels": fp,
        "fn_pixels": fn,
        "dice": round(float(dice), 4),
        "iou": round(float(iou), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
    }


def _generate_m6_plots(
    speed_evals: list[dict[str, Any]],
    pred_dir: Path,
    graphs_dir: Path,
) -> dict[str, str]:
    graphs_dir.mkdir(parents=True, exist_ok=True)
    speeds = [e["speed_kmh"] for e in speed_evals]

    # 1. Speed vs YOLO Precision/Recall/F1/mAP50
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(speeds, [e["yolo"]["frame_level"]["precision"] for e in speed_evals], "o-", label="Precision")
    ax.plot(speeds, [e["yolo"]["frame_level"]["recall"] for e in speed_evals], "s-", label="Recall")
    ax.plot(speeds, [e["yolo"]["frame_level"]["f1_score"] for e in speed_evals], "^-", label="F1-score")
    ax.plot(speeds, [e["yolo"]["frame_level"]["mAP50"] for e in speed_evals], "d--", label="mAP50")
    ax.set_title("Controlled Speed Evaluation: YOLO Object Detection (A1 & A2) [SYNTHETIC]")
    ax.set_xlabel("Train Speed (km/h)")
    ax.set_ylabel("Score")
    ax.set_xticks(speeds)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    p1 = graphs_dir / "m6_speed_vs_yolo_metrics.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    # 2. Speed vs First Detection Distance & Latency
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    a1_dists = [e["yolo"]["event_level"]["per_anomaly"]["A1"]["first_detect_distance_m"] for e in speed_evals]
    a2_dists = [e["yolo"]["event_level"]["per_anomaly"]["A2"]["first_detect_distance_m"] for e in speed_evals]
    a3_dists = [e["a3_segmentation"]["first_detection_distance_m"] for e in speed_evals]

    axes[0].plot(speeds, [d if d is not None else np.nan for d in a1_dists], "o-", label="A1 animal_carcass")
    axes[0].plot(speeds, [d if d is not None else np.nan for d in a2_dists], "s-", label="A2 grain_spillage")
    axes[0].plot(speeds, [d if d is not None else np.nan for d in a3_dists], "^-", label="A3 rail_crack")
    axes[0].set_title("Earliest Correct Detection Distance vs Speed [SYNTHETIC]")
    axes[0].set_xlabel("Train Speed (km/h)")
    axes[0].set_ylabel("Camera-to-Anomaly Distance (m)")
    axes[0].set_xticks(speeds)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    a1_lats = [e["yolo"]["event_level"]["per_anomaly"]["A1"]["detection_latency_s"] for e in speed_evals]
    a2_lats = [e["yolo"]["event_level"]["per_anomaly"]["A2"]["detection_latency_s"] for e in speed_evals]
    axes[1].plot(speeds, [x if x is not None else np.nan for x in a1_lats], "o-", label="A1 latency (s)")
    axes[1].plot(speeds, [x if x is not None else np.nan for x in a2_lats], "s-", label="A2 latency (s)")
    axes[1].set_title("Event Detection Latency vs Speed [SYNTHETIC]")
    axes[1].set_xlabel("Train Speed (km/h)")
    axes[1].set_ylabel("Latency from First GT Visibility (s)")
    axes[1].set_xticks(speeds)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    p2 = graphs_dir / "m6_speed_vs_detection_distance_latency.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=150)
    plt.close(fig)

    # 3. A4 Geometry Profile Comparison (30 km/h vs 120 km/h vs 200 km/h vs Ground Truth)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    for spd_plot in (30, 120, 200):
        pj = _load_json(pred_dir / f"pred_eval_{spd_plot}kmh.json")
        comp = pj["geometry_track_evaluation"]["compared_profile"]
        ys = [c["y_m"] for c in comp]
        est = [c["estimated_shift_m"] if c["estimated_shift_m"] is not None else np.nan for c in comp]
        axes[0].plot(ys, [v * 100.0 if not np.isnan(v) else np.nan for v in est], "o--", markersize=3, label=f"Est @ {spd_plot} km/h")
        if spd_plot == 30:
            gt_cm = [c["gt_shift_m"] * 100.0 for c in comp]
            axes[0].plot(ys, gt_cm, "k-", linewidth=2.2, label="Ground Truth (A4)")

    axes[0].set_title("A4 Lateral Displacement Profile vs Ground Truth [SYNTHETIC]")
    axes[0].set_xlabel("Track Position Y (m)")
    axes[0].set_ylabel("Lateral Displacement (cm)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    maes = [e["a4_geometry"]["displacement_mae_m"] * 100.0 for e in speed_evals]
    rmses = [e["a4_geometry"]["displacement_rmse_m"] * 100.0 for e in speed_evals]
    axes[1].plot(speeds, maes, "o-", label="MAE (cm)")
    axes[1].plot(speeds, rmses, "s-", label="RMSE (cm)")
    axes[1].set_title("A4 Geometry Estimation Error vs Speed [SYNTHETIC]")
    axes[1].set_xlabel("Train Speed (km/h)")
    axes[1].set_ylabel("Error (cm)")
    axes[1].set_xticks(speeds)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    p3 = graphs_dir / "m6_a4_geometry_comparison.png"
    fig.tight_layout()
    fig.savefig(p3, dpi=150)
    plt.close(fig)

    return {
        "speed_vs_yolo_metrics": str(p1),
        "speed_vs_distance_latency": str(p2),
        "a4_geometry_comparison": str(p3),
    }


def main() -> int:
    dataset_root = PROJECT_ROOT / "dataset" / "synthetic"
    manifest = _load_json(dataset_root / "metadata" / "dataset_manifest.json")
    pred_dir = PROJECT_ROOT / "results" / "metrics" / "m6_predictions"
    graphs_dir = PROJECT_ROOT / "results" / "graphs"

    controlled_evals: list[dict[str, Any]] = []
    variant_test_evals: list[dict[str, Any]] = []

    for v_entry in manifest["videos"]:
        if v_entry["split"] != "test":
            continue
        vid = v_entry["video_id"]
        pred_path = pred_dir / f"pred_{vid}.json"
        gt_path = dataset_root / "ground_truth" / "frame_annotations" / f"{vid}_frames.json"
        pred_data = _load_json(pred_path)
        gt_data = _load_json(gt_path)

        entry = {
            "video_id": vid,
            "speed_kmh": v_entry["speed_kmh"],
            "frame_count": v_entry["extracted_frames"],
            "yolo": evaluate_yolo_video(pred_data, gt_data),
            "a3_segmentation": evaluate_crack_video(pred_data),
            "a4_geometry": pred_data["geometry_track_evaluation"],
        }
        # Drop verbose profile list from summary report (already stored per video)
        entry["a4_geometry"] = {
            k: val for k, val in entry["a4_geometry"].items() if k != "compared_profile"
        }

        if v_entry["dataset_group"] == "controlled_evaluation":
            controlled_evals.append(entry)
        else:
            variant_test_evals.append(entry)

    controlled_evals.sort(key=lambda x: x["speed_kmh"])
    plot_paths = _generate_m6_plots(controlled_evals, pred_dir, graphs_dir)

    report = {
        "dataset_type": "SYNTHETIC",
        "milestone": "M6_CONTROLLED_EVALUATION",
        "separation_note": (
            "Controlled 7-speed evaluation (30-200 km/h) and independent variant test scenes "
            "(var_17-var_19) are reported separately and never mixed into a single score."
        ),
        "controlled_speed_benchmark": controlled_evals,
        "independent_test_variants": variant_test_evals,
        "graphs": plot_paths,
    }
    out_path = PROJECT_ROOT / "results" / "metrics" / "m6_evaluation_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("ITMS-VirtualTrack — Milestone 6 Controlled Speed Benchmark (SYNTHETIC)")
    print("=" * 78)
    print(f"{'Speed':>7} | {'P':>6} {'R':>6} {'F1':>6} {'mAP50':>6} {'Ev':>4} | {'A3 Dice':>7} {'IoU':>6} | {'A4 MAE(cm)':>10}")
    print("-" * 78)
    for row in controlled_evals:
        yf = row["yolo"]["frame_level"]
        ye = row["yolo"]["event_level"]
        a3 = row["a3_segmentation"]
        a4 = row["a4_geometry"]
        mae_cm = a4["displacement_mae_m"] * 100.0 if a4["displacement_mae_m"] is not None else -1.0
        print(
            f"{row['speed_kmh']:>5}kmh | {yf['precision']:>6.3f} {yf['recall']:>6.3f} "
            f"{yf['f1_score']:>6.3f} {yf['mAP50']:>6.3f} {ye['detected_events']:>2}/2 | "
            f"{a3['dice']:>7.3f} {a3['iou']:>6.3f} | {mae_cm:>10.2f}"
        )
    print("=" * 78)
    print(f"Full report saved to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

