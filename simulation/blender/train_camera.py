"""Front-facing train-mounted camera builder, motion setup, and validator."""

from __future__ import annotations

import math
from typing import Any

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Euler, Vector


def _set_action_linear_interpolation(action: bpy.types.Action | None) -> None:
    """Set all keyframes on an action to LINEAR interpolation across Blender 4.x/5.x."""
    if action is None:
        return
    # Blender 4.4+ / 5.x Layered Actions
    if hasattr(action, "layers"):
        for layer in action.layers:
            for strip in getattr(layer, "strips", []):
                for cb in getattr(strip, "channelbags", []):
                    for fc in cb.fcurves:
                        for kp in fc.keyframe_points:
                            kp.interpolation = "LINEAR"
        return
    # Legacy fallback
    if hasattr(action, "fcurves"):
        for fc in action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"


def compute_camera_pose(
    camera_cfg: dict[str, Any], track_datum: dict[str, float], track_pos_m: float = 0.0
) -> dict[str, Any]:
    """Compute world-space position and Euler rotation for the train camera."""
    mount = camera_cfg["mounting"]
    orient = camera_cfg["orientation_deg"]

    cam_x = float(mount["lateral_offset_m"])
    cam_y = float(track_pos_m) + float(mount["longitudinal_offset_m"])
    cam_z = float(track_datum["rail_head_top_z"]) + float(mount["height_above_rail_head_m"])

    rx_deg = float(orient.get("blender_base_rx_deg", 90.0)) + float(orient["pitch_deg"])
    ry_deg = float(orient["roll_deg"])
    rz_deg = float(orient["yaw_deg"])

    return {
        "location": (cam_x, cam_y, cam_z),
        "rotation_deg": (rx_deg, ry_deg, rz_deg),
        "rotation_rad": (math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)),
        "height_above_rail_head_m": float(mount["height_above_rail_head_m"]),
        "pitch_deg": float(orient["pitch_deg"]),
    }


def create_train_camera(
    camera_cfg: dict[str, Any], track_datum: dict[str, float], start_pos_m: float = 0.0
) -> bpy.types.Object:
    """Create and configure the forward-facing train camera in the active scene."""
    optics = camera_cfg["optics"]
    pose = compute_camera_pose(camera_cfg, track_datum, start_pos_m)

    cam_data = bpy.data.cameras.new(name="Train_Camera_Data")
    cam_data.type = "PERSP"
    cam_data.lens = float(optics["focal_length_mm"])
    cam_data.sensor_width = float(optics["sensor_width_mm"])
    cam_data.sensor_height = float(optics["sensor_height_mm"])
    cam_data.sensor_fit = str(optics.get("sensor_fit", "HORIZONTAL"))
    cam_data.clip_start = float(optics.get("clip_start_m", 0.2))
    cam_data.clip_end = float(optics.get("clip_end_m", 500.0))

    cam_obj = bpy.data.objects.new(name="Train_Camera", object_data=cam_data)
    cam_obj.rotation_mode = str(camera_cfg["orientation_deg"].get("blender_euler_order", "XYZ"))
    cam_obj.location = pose["location"]
    cam_obj.rotation_euler = Euler(pose["rotation_rad"], cam_obj.rotation_mode)

    col = bpy.data.collections.get("Camera_Rig")
    if col is None:
        col = bpy.data.collections.new("Camera_Rig")
        bpy.context.scene.collection.children.link(col)
    col.objects.link(cam_obj)

    bpy.context.scene.camera = cam_obj
    return cam_obj


def configure_linear_camera_motion(
    cam_obj: bpy.types.Object,
    start_y_m: float,
    speed_kmh: float,
    fps: float,
    frame_start: int,
    frame_end: int,
) -> dict[str, float]:
    """Animate camera linearly along +Y at exact physical speed."""
    speed_mps = float(speed_kmh) / 3.6
    duration_s = (frame_end - frame_start) / float(fps)
    end_y_m = start_y_m + speed_mps * duration_s
    z = cam_obj.location.z
    x = cam_obj.location.x

    if cam_obj.animation_data is not None:
        cam_obj.animation_data_clear()

    cam_obj.location = (x, start_y_m, z)
    cam_obj.keyframe_insert(data_path="location", frame=frame_start)

    cam_obj.location = (x, end_y_m, z)
    cam_obj.keyframe_insert(data_path="location", frame=frame_end)

    if cam_obj.animation_data is not None:
        _set_action_linear_interpolation(cam_obj.animation_data.action)

    # Reset scene frame to start frame so initial state is deterministic
    bpy.context.scene.frame_set(frame_start)
    return {
        "speed_kmh": float(speed_kmh),
        "speed_mps": round(speed_mps, 6),
        "fps": float(fps),
        "frame_start": frame_start,
        "frame_end": frame_end,
        "duration_s": round(duration_s, 6),
        "start_y_m": round(start_y_m, 6),
        "end_y_m": round(end_y_m, 6),
    }


def validate_camera(
    scene: bpy.types.Scene,
    camera_cfg: dict[str, Any],
    track_datum: dict[str, float],
    probe_distance_m: float = 15.0,
) -> dict[str, Any]:
    """Validate camera existence, height, orientation, optics, and rail/sleeper visibility."""
    cam_obj = scene.camera
    checks: dict[str, Any] = {}
    all_passed = True

    def record(name: str, passed: bool, details: dict[str, Any]) -> None:
        nonlocal all_passed
        checks[name] = {"passed": passed, **details}
        if not passed:
            all_passed = False

    if cam_obj is None or cam_obj.type != "CAMERA":
        record("camera_exists", False, {"error": "No active scene camera"})
        return {"all_passed": False, "checks": checks}

    record("camera_exists", True, {"name": cam_obj.name})

    expected_height = float(camera_cfg["mounting"]["height_above_rail_head_m"])
    measured_height = cam_obj.location.z - track_datum["rail_head_top_z"]
    height_ok = abs(measured_height - expected_height) < 1e-4
    record(
        "camera_height",
        height_ok,
        {
            "expected_height_above_rail_head_m": expected_height,
            "measured_height_above_rail_head_m": round(measured_height, 6),
            "world_z_m": round(cam_obj.location.z, 6),
        },
    )

    # Camera forward direction in world coordinates (Blender local -Z is look direction)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    cam_eval = cam_obj.evaluated_get(depsgraph)
    forward = (cam_eval.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))).normalized()
    measured_pitch_deg = math.degrees(math.asin(max(-1.0, min(1.0, forward.z))))
    expected_pitch_deg = float(camera_cfg["orientation_deg"]["pitch_deg"])

    dir_ok = (
        forward.y > 0.95
        and abs(forward.x) < 1e-4
        and abs(measured_pitch_deg - expected_pitch_deg) < 0.05
    )
    record(
        "camera_orientation",
        dir_ok,
        {
            "forward_vector": [round(forward.x, 6), round(forward.y, 6), round(forward.z, 6)],
            "expected_pitch_deg": expected_pitch_deg,
            "measured_pitch_deg": round(measured_pitch_deg, 4),
        },
    )

    expected_lens = float(camera_cfg["optics"]["focal_length_mm"])
    lens_ok = abs(cam_obj.data.lens - expected_lens) < 1e-4
    record(
        "camera_optics",
        lens_ok,
        {
            "expected_focal_length_mm": expected_lens,
            "measured_focal_length_mm": round(cam_obj.data.lens, 4),
            "horizontal_fov_deg": round(math.degrees(cam_obj.data.angle_x), 3),
            "vertical_fov_deg": round(math.degrees(cam_obj.data.angle_y), 3),
        },
    )

    # Check both rails and sleeper center at probe_distance_m ahead are visible in frame
    y_probe = cam_obj.location.y + probe_distance_m
    pts = {
        "left_rail": Vector((track_datum["left_rail_center_x"], y_probe, track_datum["rail_head_top_z"])),
        "right_rail": Vector((track_datum["right_rail_center_x"], y_probe, track_datum["rail_head_top_z"])),
        "sleeper_center": Vector((0.0, y_probe, track_datum["sleeper_top_z"])),
    }
    projected: dict[str, Any] = {}
    visibility_ok = True
    for key, pt in pts.items():
        ndc = world_to_camera_view(scene, cam_obj, pt)
        in_frame = (0.0 <= ndc.x <= 1.0) and (0.0 <= ndc.y <= 1.0) and (ndc.z > 0.0)
        projected[key] = {
            "ndc_x": round(ndc.x, 4),
            "ndc_y": round(ndc.y, 4),
            "depth_m": round(ndc.z, 4),
            "in_frame": in_frame,
        }
        if not in_frame:
            visibility_ok = False

    record(
        "rails_and_sleepers_in_frame",
        visibility_ok,
        {"probe_distance_m": probe_distance_m, "projections": projected},
    )

    return {"all_passed": all_passed, "checks": checks}

