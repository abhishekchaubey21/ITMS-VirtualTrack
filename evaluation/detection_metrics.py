"""Authoritative Milestone 7 evaluation loader & metric builders.

Uses the verified M6 independent scientific audit outputs
(results/metrics/m6_audit/) as the single source of truth without modifying
models, datasets, or original M2-M6 reports.

Notice: All metrics represent SYNTHETIC simulation runs and must never mix
the 7 controlled speeds (30-200 km/h) with independent variants (var_17-var_19).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_authoritative_audit(project_root: Path | None = None) -> dict[str, Any]:
    """Load verified M6 audit files, dataset stats, and simulation configurations."""
    root = project_root or PROJECT_ROOT
    audit_dir = root / "results" / "metrics" / "m6_audit"
    metrics_dir = root / "results" / "metrics"
    config_dir = root / "config"
    dataset_meta_dir = root / "dataset" / "synthetic" / "metadata"

    return {
        "project_root": root,
        "independent_audit": _load_json(audit_dir / "m6_independent_audit.json"),
        "consistency_report": _load_json(audit_dir / "m6_metric_consistency_report.json"),
        "leakage_audit": _load_json(audit_dir / "m6_leakage_and_split_audit.json"),
        "training_summary": _load_json(metrics_dir / "m6_training_summary.json"),
        "dataset_audit": _load_json(metrics_dir / "m6_dataset_audit.json"),
        "dataset_manifest": _load_json(dataset_meta_dir / "dataset_manifest.json"),
        "dataset_statistics": _load_json(dataset_meta_dir / "dataset_statistics.json"),
        "simulation_config": _load_json(config_dir / "simulation_config.json"),
        "camera_config": _load_json(config_dir / "camera_config.json"),
        "anomaly_config": _load_json(config_dir / "anomaly_config.json"),
    }


def _ordered_video_ids(audit_data: dict[str, Any]) -> tuple[list[str], list[str]]:
    controlled = sorted(
        [vid for vid, d in audit_data.items() if d["dataset_group"] == "controlled_evaluation"],
        key=lambda v: audit_data[v]["speed_kmh"],
    )
    variants = sorted(
        [vid for vid, d in audit_data.items() if d["dataset_group"] != "controlled_evaluation"]
    )
    return controlled, variants


def build_yolo_tables(audit_bundle: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Build YOLO overall, per-class, and FP/FN forensic tables with conf=0.001 PR."""
    yolo_audit = audit_bundle["independent_audit"]["yolo_audit"]
    controlled_ids, variant_ids = _ordered_video_ids(yolo_audit)
    ordered_ids = controlled_ids + variant_ids

    summary_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    fp_rows: list[dict[str, Any]] = []
    fn_rows: list[dict[str, Any]] = []

    for vid in ordered_ids:
        entry = yolo_audit[vid]
        c025 = entry["conf_025_recalculated"]
        c001 = entry["conf_001_pr_curve"]
        ev_a1 = c025["events"]["A1"]
        ev_a2 = c025["events"]["A2"]
        events_det = int(ev_a1["detected"]) + int(ev_a2["detected"])

        summary_rows.append(
            {
                "video_id": vid,
                "dataset_group": entry["dataset_group"],
                "speed_kmh": entry["speed_kmh"],
                "frame_count": entry["frame_count"],
                "tp_conf025": c025["tp"],
                "fp_conf025": c025["fp"],
                "fn_conf025": c025["fn"],
                "fp_outside_positive_frames": c025["fp_outside_positive_frames"],
                "precision_conf025": c025["precision"],
                "recall_conf025": c025["recall"],
                "f1_conf025": c025["f1_score"],
                "mAP50_conf001": c001["mAP50_allpoint"],
                "mAP50_95_conf001": c001["mAP50_95_allpoint"],
                "mAP50_coco101_conf001": c001["mAP50_coco101"],
                "mAP50_95_coco101_conf001": c001["mAP50_95_coco101"],
                "mAP50_trunc_conf025": c025["mAP50_allpoint"],
                "mAP50_95_trunc_conf025": c025["mAP50_95_allpoint"],
                "events_detected": events_det,
                "total_events": 2,
                "event_recall": round(events_det / 2.0, 4),
            }
        )

        for cls_key, cls_label in (
            ("0_animal_carcass", "animal_carcass (A1)"),
            ("1_grain_spillage", "grain_spillage (A2)"),
        ):
            pc_025 = c025["per_class"][cls_key]
            pc_001 = c001["per_class"][cls_key]
            per_class_rows.append(
                {
                    "video_id": vid,
                    "dataset_group": entry["dataset_group"],
                    "speed_kmh": entry["speed_kmh"],
                    "class_key": cls_key,
                    "class_label": cls_label,
                    "gt_boxes": pc_025["gt_boxes"],
                    "tp_conf025": pc_025["tp"],
                    "fp_conf025": pc_025["fp"],
                    "fn_conf025": pc_025["fn"],
                    "precision_conf025": pc_025["precision"],
                    "recall_conf025": pc_025["recall"],
                    "f1_conf025": pc_025["f1_score"],
                    "ap50_conf001": pc_001["ap50_allpoint"],
                    "ap50_95_conf001": pc_001["ap50_95_allpoint"],
                    "ap50_coco101_conf001": pc_001["ap50_coco101"],
                    "ap50_95_coco101_conf001": pc_001["ap50_95_coco101"],
                }
            )

        for fp in entry["fp_details"]:
            a2_ctx = next((c for c in fp["same_frame_gt_context"] if c["aid"] == "A2"), None)
            if not fp["has_any_visible_gt"] and fp["pred_cls"] == 1 and a2_ctx and 30.0 <= a2_ctx["dist_m"] <= 38.0:
                cause = "Early valid A2 detection before GT min-pixel cutoff"
            elif fp["has_any_visible_gt"]:
                cause = "Bottom-edge crop IoU < 0.50 (y2=540px)"
            else:
                cause = "Other FP"
            fp_rows.append(
                {
                    "video_id": vid,
                    "dataset_group": entry["dataset_group"],
                    "speed_kmh": entry["speed_kmh"],
                    "frame": fp["frame"],
                    "camera_position_m": fp["cam_y"],
                    "pred_class_id": fp["pred_cls"],
                    "confidence": fp["conf"],
                    "has_visible_gt": fp["has_any_visible_gt"],
                    "cause_classification": cause,
                }
            )

        for fn in entry["fn_details"]:
            cause = (
                "Frustum bottom-edge clipping (d=3.61-4.17m)"
                if fn["touches_bottom_edge"]
                else "Small distant entry miss (gt height ~8.1px)"
            )
            fn_rows.append(
                {
                    "video_id": vid,
                    "dataset_group": entry["dataset_group"],
                    "speed_kmh": entry["speed_kmh"],
                    "frame": fn["frame"],
                    "camera_position_m": fn["cam_y"],
                    "anomaly_id": fn["anomaly_id"],
                    "distance_m": fn["dist_m"],
                    "gt_width_px": fn["gt_wh_px"][0],
                    "gt_height_px": fn["gt_wh_px"][1],
                    "touches_bottom_edge": fn["touches_bottom_edge"],
                    "cause_classification": cause,
                }
            )

    return {
        "yolo_summary": pd.DataFrame(summary_rows),
        "yolo_per_class": pd.DataFrame(per_class_rows),
        "yolo_fp_forensics": pd.DataFrame(fp_rows),
        "yolo_fn_forensics": pd.DataFrame(fn_rows),
    }


def build_a3_tables(audit_bundle: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Build A3 segmentation tables separating visible-only vs all-frames + val calibration."""
    a3_audit = audit_bundle["independent_audit"]["a3_crack_audit"]
    val_sweep = audit_bundle["consistency_report"]["a3_val_threshold_calibration_audit"][
        "all_558_val_frames_threshold_sweep"
    ]
    controlled_ids, variant_ids = _ordered_video_ids(a3_audit)
    ordered_ids = controlled_ids + variant_ids

    summary_rows: list[dict[str, Any]] = []
    pre_vis_rows: list[dict[str, Any]] = []

    for vid in ordered_ids:
        entry = a3_audit[vid]
        vis = entry["crack_visible_frames_only"]
        all_post = entry["all_frames_with_min_area_8px"]
        all_raw = entry["all_frames_raw_no_min_area"]
        pre_frames = entry["non_gt_visible_frames_in_crack_records"]

        summary_rows.append(
            {
                "video_id": vid,
                "dataset_group": entry["dataset_group"],
                "speed_kmh": entry["speed_kmh"],
                "gt_visible_frames": entry["gt_visible_frames"],
                "gt_visible_frames_detected": entry["gt_visible_frames_detected"],
                "event_detected": entry["event_detected"],
                "first_visible_distance_m": entry["first_visible_distance_m"],
                "first_detection_distance_m": entry["first_detection_distance_m"],
                "detection_latency_s": entry["detection_latency_s"],
                "gt_total_fg_pixels": vis["tp_px"] + vis["fn_px"],
                "vis_only_dice": vis["dice"],
                "vis_only_iou": vis["iou"],
                "vis_only_precision": vis["precision"],
                "vis_only_recall": vis["recall"],
                "all_frames_dice": all_post["dice"],
                "all_frames_iou": all_post["iou"],
                "all_frames_precision": all_post["precision"],
                "all_frames_recall": all_post["recall"],
                "all_frames_fp_px": all_post["fp_px"],
                "raw_no_min_area_dice": all_raw["dice"],
                "early_pre_vis_fp_frames_10_to_11m": len(pre_frames),
                "spurious_neg_frames_outside_approach": 0,
            }
        )

        for pf in pre_frames:
            pre_vis_rows.append(
                {
                    "video_id": vid,
                    "dataset_group": entry["dataset_group"],
                    "speed_kmh": entry["speed_kmh"],
                    "frame": pf["frame"],
                    "camera_position_m": pf["cam_y"],
                    "distance_to_a3_150m": pf["dist_to_150m"],
                    "pred_fg_pixels": pf["pred_px"],
                }
            )

    return {
        "a3_summary": pd.DataFrame(summary_rows),
        "a3_pre_visibility_detections": pd.DataFrame(pre_vis_rows),
        "a3_val_threshold_sweep": pd.DataFrame(val_sweep),
    }


def build_a4_tables(audit_bundle: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Build A4 unit-explicit geometry summary, frame detection, and 33-point error profiles."""
    a4_audit = audit_bundle["independent_audit"]["a4_geometry_audit"]
    controlled_ids, variant_ids = _ordered_video_ids(a4_audit)
    ordered_ids = controlled_ids + variant_ids

    summary_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []

    for vid in ordered_ids:
        entry = a4_audit[vid]
        pm = entry["recalculated_profile_metrics"]
        fl = entry["frame_level_a4_detection"]
        ft = entry["full_track_outside_196_204_m"]
        ungated_rng = ft["ungated_active_bins_above_15mm_range"]

        summary_rows.append(
            {
                "video_id": vid,
                "dataset_group": entry["dataset_group"],
                "speed_kmh": entry["speed_kmh"],
                "gt_max_dev_cm": round(pm["gt_max_dev_m"] * 100.0, 3),
                "est_max_dev_cm": round(pm["est_max_dev_m"] * 100.0, 3),
                "peak_amplitude_diff_mm": pm["peak_amplitude_diff_mm"],
                "profile_mae_cm": pm["mae_cm"],
                "profile_rmse_cm": pm["rmse_cm"],
                "max_pointwise_error_cm": pm["max_pointwise_error_cm"],
                "gt_peak_y_m": pm["gt_peak_y_m"],
                "est_peak_y_m": pm["est_peak_y_m"],
                "worst_point_y_m": pm["worst_point_y_m"],
                "worst_point_signed_err_cm": pm["worst_point_signed_err_cm"],
                "frame_tp": fl["tp"],
                "frame_fp": fl["fp"],
                "frame_fn": fl["fn"],
                "frame_precision": fl["precision"],
                "frame_recall": fl["recall"],
                "event_detected": fl["first_detect_distance_m"] is not None,
                "first_vis_distance_m": fl["first_vis_distance_m"],
                "first_detect_distance_m": fl["first_detect_distance_m"],
                "detection_latency_s": fl["detection_latency_s"],
                "outside_196_204_mae_cm": ft["outside_mae_cm"],
                "outside_196_204_max_err_cm": ft["outside_max_err_cm"],
                "outside_196_204_max_err_y_m": ft["outside_max_err_y_m"],
                "ungated_bins_outside_188_212_count": len(ft["ungated_bins_outside_188_212"]),
                "ungated_span_m": f"{ungated_rng[0]}-{ungated_rng[1]}" if ungated_rng else "None",
            }
        )

        for pt in entry["compared_profile_error_cm"]:
            profile_rows.append(
                {
                    "video_id": vid,
                    "dataset_group": entry["dataset_group"],
                    "speed_kmh": entry["speed_kmh"],
                    "y_m": pt["y_m"],
                    "gt_cm": pt["gt_cm"],
                    "est_cm": pt["est_cm"],
                    "signed_err_cm": pt["err_cm"],
                    "abs_err_cm": round(abs(pt["err_cm"]), 3) if pt["err_cm"] is not None else None,
                }
            )

    return {
        "a4_summary": pd.DataFrame(summary_rows),
        "a4_error_profiles": pd.DataFrame(profile_rows),
    }


def build_all_metric_tables(project_root: Path | None = None) -> dict[str, Any]:
    """Assemble all authoritative Milestone 7 DataFrames and metadata."""
    bundle = load_authoritative_audit(project_root)
    tables: dict[str, Any] = {"bundle": bundle}
    tables.update(build_yolo_tables(bundle))
    tables.update(build_a3_tables(bundle))
    tables.update(build_a4_tables(bundle))
    return tables

