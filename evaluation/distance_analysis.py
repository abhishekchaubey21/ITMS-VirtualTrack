"""Anomaly-level earliest detection distance, sampling stride, and timeline analysis.

Synthesizes deduplicated event-level detection behavior across A1-A4 from the
verified M6 independent audit without mixing controlled speeds with test variants.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_anomaly_distance_timeline(tables: dict[str, Any]) -> pd.DataFrame:
    """Build unified per-anomaly (A1-A4) distance, delay, and window duration table."""
    bundle = tables["bundle"]
    yolo_audit = bundle["independent_audit"]["yolo_audit"]
    a3_audit = bundle["independent_audit"]["a3_crack_audit"]
    a4_audit = bundle["independent_audit"]["a4_geometry_audit"]

    ordered_ids = sorted(
        [v for v, d in yolo_audit.items() if d["dataset_group"] == "controlled_evaluation"],
        key=lambda v: yolo_audit[v]["speed_kmh"],
    ) + sorted(
        [v for v, d in yolo_audit.items() if d["dataset_group"] != "controlled_evaluation"]
    )

    rows: list[dict[str, Any]] = []
    for vid in ordered_ids:
        y_entry = yolo_audit[vid]
        spd = float(y_entry["speed_kmh"])
        v_mps = spd / 3.6
        fps = 30.0
        stride_m = v_mps / fps
        grp = y_entry["dataset_group"]

        # A1 & A2 from YOLO event deduplication
        for aid, pos_m, cname, mod_name in (
            ("A1", 50.0, "animal_carcass", "YOLOv8n Object Detection"),
            ("A2", 100.0, "grain_spillage", "YOLOv8n Object Detection"),
        ):
            ev = y_entry["conf_025_recalculated"]["events"][aid]
            first_vis_dist = round(pos_m - float(ev["first_vis_cam"]), 4)
            censor_note = (
                "LEFT-CENSORED: visible on Frame 1 at cam=0.0 m (50 m is track-start limit, not max detector range)"
                if aid == "A1"
                else "Uncensored GT visibility (~33-36 m); pre-visibility conf>=0.25 boxes start up to 36.7 m"
            )
            rows.append(
                {
                    "video_id": vid,
                    "dataset_group": grp,
                    "speed_kmh": int(spd),
                    "speed_mps": round(v_mps, 3),
                    "stride_m_per_frame": round(stride_m, 4),
                    "anomaly_id": aid,
                    "anomaly_name": cname,
                    "track_position_m": pos_m,
                    "detector_module": mod_name,
                    "event_detected": bool(ev["detected"]),
                    "first_gt_visible_distance_m": first_vis_dist,
                    "first_detect_distance_m": ev["first_det_dist"],
                    "detection_delay_s": ev["detection_latency_s"],
                    "gt_visible_frames": int(ev["vis"]),
                    "matched_frames": int(ev["matched"]),
                    "visible_window_duration_s": round(int(ev["vis"]) / fps, 4),
                    "range_censoring_note": censor_note,
                }
            )

        # A3 from RailCrackUNet
        a3_e = a3_audit[vid]
        rows.append(
            {
                "video_id": vid,
                "dataset_group": grp,
                "speed_kmh": int(spd),
                "speed_mps": round(v_mps, 3),
                "stride_m_per_frame": round(stride_m, 4),
                "anomaly_id": "A3",
                "anomaly_name": "rail_crack",
                "track_position_m": 150.0,
                "detector_module": "RailCrackUNet ROI (512x256)",
                "event_detected": bool(a3_e["event_detected"]),
                "first_gt_visible_distance_m": a3_e["first_visible_distance_m"],
                "first_detect_distance_m": a3_e["first_detection_distance_m"],
                "detection_delay_s": a3_e["detection_latency_s"],
                "gt_visible_frames": int(a3_e["gt_visible_frames"]),
                "matched_frames": int(a3_e["gt_visible_frames_detected"]),
                "visible_window_duration_s": round(int(a3_e["gt_visible_frames"]) / fps, 4),
                "range_censoring_note": (
                    "GT visibility <=10.0 m; track-corridor ROI captures 4.2-13 m ahead "
                    f"(only {a3_e['gt_visible_frames']} GT frames / {a3_e['crack_visible_frames_only']['tp_px'] + a3_e['crack_visible_frames_only']['fn_px']} px)"
                ),
            }
        )

        # A4 from RailGeometryAnalyzer frame-level alert
        a4_fl = a4_audit[vid]["frame_level_a4_detection"]
        a4_vis_cnt = int(a4_fl["tp"]) + int(a4_fl["fn"])
        rows.append(
            {
                "video_id": vid,
                "dataset_group": grp,
                "speed_kmh": int(spd),
                "speed_mps": round(v_mps, 3),
                "stride_m_per_frame": round(stride_m, 4),
                "anomaly_id": "A4",
                "anomaly_name": "rail_misalignment",
                "track_position_m": 200.0,
                "detector_module": "RailGeometryAnalyzer IPM",
                "event_detected": a4_fl["first_detect_distance_m"] is not None,
                "first_gt_visible_distance_m": a4_fl["first_vis_distance_m"],
                "first_detect_distance_m": a4_fl["first_detect_distance_m"],
                "detection_delay_s": a4_fl["detection_latency_s"],
                "gt_visible_frames": a4_vis_cnt,
                "matched_frames": int(a4_fl["tp"]),
                "visible_window_duration_s": round(a4_vis_cnt / fps, 4),
                "range_censoring_note": (
                    "Frame alert requires >=3.0 cm smoothed deviation over >=1.2 m span; "
                    f"profile peak at {a4_audit[vid]['recalculated_profile_metrics']['est_peak_y_m']} m (GT 200.0 m)"
                ),
            }
        )

    return pd.DataFrame(rows)

