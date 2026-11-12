"""Physically consistent speed controller, frame validator, and visibility analyzer."""

from __future__ import annotations

from typing import Any

import bpy

from ground_truth_generator import project_anomaly_to_camera
from train_camera import _set_action_linear_interpolation


def compute_speed_schedule(
    speed_kmh: float,
    start_m: float,
    end_m: float,
    fps: int,
) -> dict[str, Any]:
    """Compute exact deterministic kinematic frame schedule for a given speed and FPS."""
    speed_kmh_f = float(speed_kmh)
    if speed_kmh_f <= 0.0:
        raise ValueError(f"speed_kmh must be > 0, got {speed_kmh}")
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")

    track_distance_m = float(end_m) - float(start_m)
    if track_distance_m <= 0.0:
        raise ValueError(f"Invalid track span [{start_m}, {end_m}]")

    speed_mps = speed_kmh_f / 3.6
    duration_s = track_distance_m / speed_mps
    frame_interval_s = 1.0 / float(fps)
    distance_per_frame_m = speed_mps * frame_interval_s

    # Exact intervals cover [0, 250] m; +1 inclusive endpoint frame ensures both 0.0m and 250.0m are rendered
    num_intervals = max(1, int(round(duration_s * float(fps))))
    frame_count = num_intervals + 1
    actual_travel_duration_s = num_intervals * frame_interval_s
    video_container_duration_s = frame_count * frame_interval_s
    actual_end_m = float(start_m) + speed_mps * actual_travel_duration_s

    frames: list[dict[str, Any]] = []
    for idx in range(frame_count):
        t_s = idx * frame_interval_s
        pos_m = float(start_m) + speed_mps * t_s
        frames.append(
            {
                "frame": idx + 1,
                "frame_index": idx,
                "timestamp_s": round(t_s, 6),
                "expected_position_m": round(pos_m, 6),
            }
        )

    return {
        "speed_kmh": int(speed_kmh_f) if speed_kmh_f.is_integer() else speed_kmh_f,
        "speed_mps": round(speed_mps, 6),
        "track_length_m": round(track_distance_m, 6),
        "fps": int(fps),
        "frame_interval_s": round(frame_interval_s, 8),
        "distance_per_frame_m": round(distance_per_frame_m, 6),
        "start_position_m": float(start_m),
        "target_end_position_m": float(end_m),
        "actual_end_position_m": round(actual_end_m, 6),
        "expected_duration_s": round(duration_s, 6),
        "actual_travel_duration_s": round(actual_travel_duration_s, 6),
        "video_container_duration_s": round(video_container_duration_s, 6),
        "num_intervals": num_intervals,
        "frame_count": frame_count,
        "frame_start": 1,
        "frame_end": frame_count,
        "frames": frames,
    }


def apply_speed_trajectory(
    scene: bpy.types.Scene,
    cam_obj: bpy.types.Object,
    schedule: dict[str, Any],
) -> None:
    """Configure deterministic linear camera translation keyframes matching schedule."""
    scene.render.fps = int(schedule["fps"])
    scene.render.fps_base = 1.0
    scene.frame_start = int(schedule["frame_start"])
    scene.frame_end = int(schedule["frame_end"])
    scene.frame_step = 1

    if cam_obj.animation_data is not None:
        cam_obj.animation_data_clear()

    x = cam_obj.location.x
    z = cam_obj.location.z

    cam_obj.location = (x, float(schedule["start_position_m"]), z)
    cam_obj.keyframe_insert(data_path="location", frame=int(schedule["frame_start"]))

    cam_obj.location = (x, float(schedule["actual_end_position_m"]), z)
    cam_obj.keyframe_insert(data_path="location", frame=int(schedule["frame_end"]))

    if cam_obj.animation_data is not None:
        _set_action_linear_interpolation(cam_obj.animation_data.action)

    scene.frame_set(int(schedule["frame_start"]))


def evaluate_trajectory_and_visibility(
    scene: bpy.types.Scene,
    cam_obj: bpy.types.Object,
    schedule: dict[str, Any],
    anomaly_objects: dict[str, tuple[bpy.types.Object, dict[str, Any]]],
    gt_rules: dict[str, Any],
    tol_m: float = 1e-4,
) -> dict[str, Any]:
    """Validate camera positions at every frame and analyze A1-A4 frustum/visibility."""
    base_x = cam_obj.location.x
    base_z = cam_obj.location.z

    max_err_y = 0.0
    sum_err_y = 0.0
    max_drift_x = 0.0
    max_drift_z = 0.0
    worst_frame = None
    first_y = None
    last_y = None
    prev_y = None
    monotonic_ok = True

    per_frame_records: list[dict[str, Any]] = []
    vis_accum: dict[str, dict[str, Any]] = {
        aid: {
            "id": aid,
            "class": meta["class"],
            "detection_method": meta["detection_method"],
            "anomaly_position_m": float(meta["configured_position_m"]),
            "first_frustum_frame": None,
            "first_frustum_camera_y_m": None,
            "first_frustum_distance_m": None,
            "first_sufficiently_visible_frame": None,
            "first_sufficiently_visible_camera_y_m": None,
            "first_sufficiently_visible_distance_m": None,
            "last_frustum_frame": None,
            "last_frustum_camera_y_m": None,
            "last_frustum_distance_m": None,
            "last_sufficiently_visible_frame": None,
            "last_sufficiently_visible_camera_y_m": None,
            "last_sufficiently_visible_distance_m": None,
            "total_frustum_frames": 0,
            "total_sufficiently_visible_frames": 0,
            "partially_occluded_frame_count": 0,
            "fully_occluded_frame_count": 0,
            "min_visible_fraction": 1.0,
            "first_sufficiently_visible_bbox": None,
            "peak_area_frame": None,
            "peak_area_distance_m": None,
            "peak_area_bbox": None,
            "_peak_area_px": -1.0,
        }
        for aid, (_obj, meta) in anomaly_objects.items()
    }

    depsgraph = bpy.context.evaluated_depsgraph_get()

    for entry in schedule["frames"]:
        f_num = int(entry["frame"])
        scene.frame_set(f_num)
        depsgraph.update()
        cam_eval = cam_obj.evaluated_get(depsgraph)
        loc = cam_eval.matrix_world.translation

        y_actual = float(loc.y)
        err_y = abs(y_actual - float(entry["expected_position_m"]))
        drift_x = abs(float(loc.x) - float(base_x))
        drift_z = abs(float(loc.z) - float(base_z))

        if first_y is None:
            first_y = y_actual
        if prev_y is not None and y_actual <= prev_y:
            monotonic_ok = False
        prev_y = y_actual
        last_y = y_actual

        sum_err_y += err_y
        if err_y > max_err_y:
            max_err_y = err_y
            worst_frame = f_num
        max_drift_x = max(max_drift_x, drift_x)
        max_drift_z = max(max_drift_z, drift_z)

        frame_anomalies: list[dict[str, Any]] = []
        for aid in sorted(anomaly_objects.keys()):
            a_obj, a_meta = anomaly_objects[aid]
            exempt = {"Rail_Right", "Rail_Left"} if aid in ("A3", "A4") else None
            proj = project_anomaly_to_camera(
                scene, cam_eval, a_obj, a_meta, gt_rules, occluder_exempt_names=exempt
            )
            frame_anomalies.append(proj)

            acc = vis_accum[aid]
            cam_dist = float(proj["camera_distance_m"])
            if proj.get("in_frustum"):
                acc["total_frustum_frames"] += 1
                if acc["first_frustum_frame"] is None:
                    acc["first_frustum_frame"] = f_num
                    acc["first_frustum_camera_y_m"] = round(y_actual, 4)
                    acc["first_frustum_distance_m"] = round(cam_dist, 4)
                acc["last_frustum_frame"] = f_num
                acc["last_frustum_camera_y_m"] = round(y_actual, 4)
                acc["last_frustum_distance_m"] = round(cam_dist, 4)

            if proj.get("occluded"):
                acc["partially_occluded_frame_count"] += 1
            if proj.get("in_frustum") and float(proj.get("visible_fraction", 1.0)) < float(
                gt_rules.get("min_visible_fraction", 0.15)
            ):
                acc["fully_occluded_frame_count"] += 1
            if proj.get("in_frustum") and proj.get("min_depth_m", 999.0) <= float(
                gt_rules.get("max_annotation_distance_m", 120.0)
            ):
                acc["min_visible_fraction"] = round(
                    min(acc["min_visible_fraction"], float(proj.get("visible_fraction", 1.0))),
                    4,
                )

            if proj.get("sufficiently_visible") and proj.get("bbox"):
                acc["total_sufficiently_visible_frames"] += 1
                if acc["first_sufficiently_visible_frame"] is None:
                    acc["first_sufficiently_visible_frame"] = f_num
                    acc["first_sufficiently_visible_camera_y_m"] = round(y_actual, 4)
                    acc["first_sufficiently_visible_distance_m"] = round(cam_dist, 4)
                    acc["first_sufficiently_visible_bbox"] = proj["bbox"]
                acc["last_sufficiently_visible_frame"] = f_num
                acc["last_sufficiently_visible_camera_y_m"] = round(y_actual, 4)
                acc["last_sufficiently_visible_distance_m"] = round(cam_dist, 4)

                bx1, by1, bx2, by2 = proj["bbox"]
                area = max(0.0, (bx2 - bx1) * (by2 - by1))
                if area > acc["_peak_area_px"]:
                    acc["_peak_area_px"] = area
                    acc["peak_area_frame"] = f_num
                    acc["peak_area_distance_m"] = round(cam_dist, 4)
                    acc["peak_area_bbox"] = proj["bbox"]

        per_frame_records.append(
            {
                "frame": f_num,
                "frame_index": entry["frame_index"],
                "timestamp_s": entry["timestamp_s"],
                "camera_position_m": round(y_actual, 6),
                "anomalies": frame_anomalies,
            }
        )

    scene.frame_set(int(schedule["frame_start"]))

    # Finalize visibility summaries (drop private helper key)
    visibility_summary: dict[str, Any] = {}
    for aid, acc in vis_accum.items():
        clean = {k: v for k, v in acc.items() if not k.startswith("_")}
        clean["any_partially_occluded"] = clean["partially_occluded_frame_count"] > 0
        clean["any_fully_occluded"] = clean["fully_occluded_frame_count"] > 0
        visibility_summary[aid] = clean

    n_frames = len(schedule["frames"])
    mean_err_y = sum_err_y / max(1, n_frames)
    duration_err = abs(
        float(schedule["actual_travel_duration_s"]) - float(schedule["expected_duration_s"])
    )
    end_err = abs(float(last_y or 0.0) - float(schedule["target_end_position_m"]))
    start_err = abs(float(first_y or 0.0) - float(schedule["start_position_m"]))

    trajectory_passed = (
        monotonic_ok
        and max_err_y <= tol_m
        and max_drift_x <= tol_m
        and max_drift_z <= tol_m
        and duration_err <= 1e-4
        and start_err <= tol_m
        and end_err <= tol_m
    )

    trajectory_validation = {
        "passed": trajectory_passed,
        "frames_checked": n_frames,
        "monotonic_forward": monotonic_ok,
        "first_frame_y_m": round(first_y or 0.0, 6),
        "last_frame_y_m": round(last_y or 0.0, 6),
        "start_position_error_m": round(start_err, 8),
        "end_position_error_m": round(end_err, 8),
        "max_position_error_m": round(max_err_y, 8),
        "mean_position_error_m": round(mean_err_y, 8),
        "worst_error_frame": worst_frame,
        "max_lateral_drift_m": round(max_drift_x, 8),
        "max_vertical_drift_m": round(max_drift_z, 8),
        "duration_error_s": round(duration_err, 8),
    }

    return {
        "trajectory_validation": trajectory_validation,
        "visibility_summary": visibility_summary,
        "frames": per_frame_records,
    }
