"""Cross-speed performance aggregation and latency/real-time budget analysis.

Strictly separates compute-only latency from disk I/O + annotation + video
encoding overhead, and compares against 30 FPS (33.33 ms) and 60 FPS (16.67 ms)
real-time frame budgets using authoritative M6 independent audit values.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_latency_table(tables: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build per-video compute latency breakdown and measured I/O wall-clock summary."""
    lat_audit = tables["bundle"]["independent_audit"]["latency_and_realtime_audit"]
    per_vid = lat_audit["per_video_compute_latencies"]
    wall = lat_audit["wall_clock_io_breakdown_eval_60kmh"]
    yolo_audit = tables["bundle"]["independent_audit"]["yolo_audit"]

    ordered_ids = sorted(
        [v for v in per_vid if yolo_audit[v]["dataset_group"] == "controlled_evaluation"],
        key=lambda v: per_vid[v]["speed_kmh"],
    ) + sorted(
        [v for v in per_vid if yolo_audit[v]["dataset_group"] != "controlled_evaluation"]
    )

    rows: list[dict[str, Any]] = []
    for vid in ordered_ids:
        r = per_vid[vid]
        tot = float(r["total_pipeline_ms_mean"])
        geom_share_pct = round(100.0 * float(r["geometry_ms_mean"]) / max(1e-9, tot), 1)
        rows.append(
            {
                "video_id": vid,
                "dataset_group": yolo_audit[vid]["dataset_group"],
                "speed_kmh": int(r["speed_kmh"]),
                "frame_count": int(r["frames"]),
                "yolo_ms_mean": r["yolo_ms_mean"],
                "yolo_ms_p95": r["yolo_ms_p95"],
                "crack_unet_ms_mean": r["crack_unet_ms_mean"],
                "crack_unet_ms_p95": r["crack_unet_ms_p95"],
                "geometry_ms_mean": r["geometry_ms_mean"],
                "geometry_ms_p95": r["geometry_ms_p95"],
                "geometry_share_pct": geom_share_pct,
                "compute_only_ms_mean": tot,
                "compute_only_ms_p95": r["total_pipeline_ms_p95"],
                "compute_only_fps": r["effective_fps_compute_only"],
                "budget_30fps_ms": 33.333,
                "budget_60fps_ms": 16.667,
                "meets_30fps": bool(tot <= 33.333),
                "meets_60fps": bool(tot <= 16.667),
                "slowdown_vs_30fps": round(tot / 33.333, 2),
                "slowdown_vs_60fps": round(tot / 16.667, 2),
            }
        )

    io_summary = {
        "benchmark_sample": wall["benchmark_sample"],
        "png_disk_read_ms_mean": wall["png_disk_read_ms_mean"],
        "mp4_video_decode_ms_mean": wall["mp4_video_decode_ms_mean"],
        "compute_only_predict_frame_ms_mean": wall["hybrid_compute_predict_frame_ms_mean"],
        "compute_only_fps_eval60_bench": round(1000.0 / float(wall["hybrid_compute_predict_frame_ms_mean"]), 2),
        "annotate_frame_ms_mean": wall["annotate_frame_ms_mean"],
        "cv2_videowriter_encode_ms_mean": wall["cv2_videowriter_encode_ms_mean"],
        "end_to_end_png_compute_annotate_encode_ms_mean": wall["end_to_end_with_png_and_video_write_ms_mean"],
        "end_to_end_fps_measured": round(
            1000.0 / float(wall["end_to_end_with_png_and_video_write_ms_mean"]), 2
        ),
        "end_to_end_mp4_decode_estimated_ms_mean": round(
            float(wall["mp4_video_decode_ms_mean"])
            + float(wall["hybrid_compute_predict_frame_ms_mean"])
            + float(wall["annotate_frame_ms_mean"])
            + float(wall["cv2_videowriter_encode_ms_mean"]),
            3,
        ),
        "meets_30fps_compute_only": False,
        "meets_60fps_compute_only": False,
        "meets_30fps_end_to_end": False,
        "meets_60fps_end_to_end": False,
        "scope_note": wall["note"],
    }
    return pd.DataFrame(rows), io_summary


def build_speed_synthesis(
    tables: dict[str, Any],
    distance_df: pd.DataFrame,
    latency_df: pd.DataFrame,
    io_summary: dict[str, Any],
) -> dict[str, Any]:
    """Synthesize controlled-speed trends vs independent variant benchmarks cleanly."""
    y_df = tables["yolo_summary"]
    y_ctrl = y_df[y_df["dataset_group"] == "controlled_evaluation"].sort_values("speed_kmh")
    y_var = y_df[y_df["dataset_group"] != "controlled_evaluation"]

    a3_df = tables["a3_summary"]
    a3_ctrl = a3_df[a3_df["dataset_group"] == "controlled_evaluation"].sort_values("speed_kmh")
    a3_var = a3_df[a3_df["dataset_group"] != "controlled_evaluation"]

    a4_df = tables["a4_summary"]
    a4_ctrl = a4_df[a4_df["dataset_group"] == "controlled_evaluation"].sort_values("speed_kmh")
    a4_var = a4_df[a4_df["dataset_group"] != "controlled_evaluation"]

    return {
        "dataset_type": "SYNTHETIC",
        "scientific_disclaimers": [
            "Data is entirely synthetic and Blender-generated (Blender 5.3.0 EEVEE, 960x540, 30 FPS).",
            "Railway geometry is a simplified straight 250 m Indian Broad Gauge (1.676 m) track without curves, switches, or catenary.",
            "Motion blur (camera_config.motion_blur.enabled=false), lens distortion, rolling shutter, suspension dynamics, and CMOS noise are disabled.",
            "Only 20 randomized environment variants were created (14 train / 3 val / 3 held-out test).",
            "The 7 controlled speeds (30-200 km/h) reuse one benchmark environment and do not represent 7 independent real-world environments.",
            "No real railway footage was used for training or validation; results do not establish real-world generalization or railway safety certification.",
        ],
        "controlled_7_speed_findings": {
            "speeds_kmh": [int(x) for x in y_ctrl["speed_kmh"].tolist()],
            "yolo_conf025_precision_range": [float(y_ctrl["precision_conf025"].min()), float(y_ctrl["precision_conf025"].max())],
            "yolo_conf025_recall_range": [float(y_ctrl["recall_conf025"].min()), float(y_ctrl["recall_conf025"].max())],
            "yolo_conf001_mAP50_range": [float(y_ctrl["mAP50_conf001"].min()), float(y_ctrl["mAP50_conf001"].max())],
            "yolo_conf001_mAP50_95_range": [float(y_ctrl["mAP50_95_conf001"].min()), float(y_ctrl["mAP50_95_conf001"].max())],
            "yolo_200kmh_recall_note": (
                "At 200 km/h (stride 1.852 m/f), recall drops to 0.9535 (41/43, mAP50=0.9514) because Frame 26 (A1) "
                "and Frame 53 (A2) both fall at d=3.704 m where the object clips the bottom image edge (y2=540 px), "
                "magnifying 2 misses across only 43 GT boxes (4.65% drop vs 0.72% for 2 exit misses at 30 km/h)."
            ),
            "a3_vis_only_dice_range": [float(a3_ctrl["vis_only_dice"].min()), float(a3_ctrl["vis_only_dice"].max())],
            "a3_all_frames_dice_range": [float(a3_ctrl["all_frames_dice"].min()), float(a3_ctrl["all_frames_dice"].max())],
            "a3_visible_frames_30_vs_200": [
                int(a3_ctrl.iloc[0]["gt_visible_frames"]),
                int(a3_ctrl.iloc[-1]["gt_visible_frames"]),
            ],
            "a3_gt_pixels_at_200kmh": int(a3_ctrl.iloc[-1]["gt_total_fg_pixels"]),
            "a4_scalar_peak_amplitude_diff_mm_range": [
                float(a4_ctrl["peak_amplitude_diff_mm"].min()),
                float(a4_ctrl["peak_amplitude_diff_mm"].max()),
            ],
            "a4_profile_mae_cm_range": [
                float(a4_ctrl["profile_mae_cm"].min()),
                float(a4_ctrl["profile_mae_cm"].max()),
            ],
            "a4_profile_rmse_cm_range": [
                float(a4_ctrl["profile_rmse_cm"].min()),
                float(a4_ctrl["profile_rmse_cm"].max()),
            ],
            "a4_pointwise_max_error_cm_range": [
                float(a4_ctrl["max_pointwise_error_cm"].min()),
                float(a4_ctrl["max_pointwise_error_cm"].max()),
            ],
            "a4_outside_196_204_mae_cm_range": [
                float(a4_ctrl["outside_196_204_mae_cm"].min()),
                float(a4_ctrl["outside_196_204_mae_cm"].max()),
            ],
        },
        "independent_test_variants_summary": {
            "videos": y_var["video_id"].tolist(),
            "yolo_conf025_f1_range": [float(y_var["f1_conf025"].min()), float(y_var["f1_conf025"].max())],
            "yolo_conf001_mAP50_range": [float(y_var["mAP50_conf001"].min()), float(y_var["mAP50_conf001"].max())],
            "a3_all_frames_dice_range": [float(a3_var["all_frames_dice"].min()), float(a3_var["all_frames_dice"].max())],
            "a4_profile_mae_cm_range": [float(a4_var["profile_mae_cm"].min()), float(a4_var["profile_mae_cm"].max())],
            "a4_frame_fp_note": "var_19 experienced 2 genuine early A4 false-positive frames ahead of anomalies (f16 @ 20.83 m, f19 @ 25.00 m; frame precision=0.875).",
        },
        "throughput_and_realtime": io_summary,
        "compute_fps_range_across_10_runs": [
            float(latency_df["compute_only_fps"].min()),
            float(latency_df["compute_only_fps"].max()),
        ],
    }

