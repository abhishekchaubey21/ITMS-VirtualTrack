"""Deterministic 250 m railway corridor builder and geometry validator."""

from __future__ import annotations

import math
import random
from typing import Any

import bpy
from mathutils import Vector

from track_components import (
    build_ballast_bed_mesh,
    build_ground_mesh,
    build_marker_post_mesh,
    build_rail_mesh,
    build_scrub_bush_mesh,
    build_sleeper_template_mesh,
    create_ballast_material,
    create_fastener_material,
    create_ground_material,
    create_rail_crack_material,
    create_rail_steel_material,
    create_simple_material,
    create_sleeper_concrete_material,
    ground_elevation_at,
    lateral_misalignment_at_y,
    lateral_misalignment_slope_at_y,
)


def _get_or_create_collection(
    name: str, parent: bpy.types.Collection | None = None
) -> bpy.types.Collection:
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        target = parent if parent is not None else bpy.context.scene.collection
        target.children.link(col)
    return col


def compute_track_datum(sim_cfg: dict[str, Any]) -> dict[str, float]:
    """Compute consistent metric track elevations and rail coordinates from config."""
    track = sim_cfg["track"]
    gauge_m = float(track["gauge_m"])
    rail_cfg = track["rail"]
    sleeper_cfg = track["sleeper"]
    ballast_cfg = track["ballast"]

    head_width_m = float(rail_cfg["head_width_m"])
    rail_height_m = float(rail_cfg["height_m"])
    ballast_height_m = float(ballast_cfg["height_m"])
    sleeper_height_m = float(sleeper_cfg["height_m"])
    embed_depth_m = float(sleeper_cfg["embed_depth_m"])

    # Inner gauge faces sit at +-gauge_m / 2; rail centerlines are offset outward by half head width.
    left_gauge_face_x = -gauge_m * 0.5
    right_gauge_face_x = +gauge_m * 0.5
    left_rail_center_x = left_gauge_face_x - head_width_m * 0.5
    right_rail_center_x = right_gauge_face_x + head_width_m * 0.5

    ballast_top_z = ballast_height_m
    sleeper_base_z = ballast_top_z - embed_depth_m
    sleeper_top_z = sleeper_base_z + sleeper_height_m
    rail_base_z = sleeper_top_z
    rail_head_top_z = rail_base_z + rail_height_m

    return {
        "length_m": float(track["length_m"]),
        "gauge_m": gauge_m,
        "rail_center_to_center_m": right_rail_center_x - left_rail_center_x,
        "left_gauge_face_x": left_gauge_face_x,
        "right_gauge_face_x": right_gauge_face_x,
        "left_rail_center_x": left_rail_center_x,
        "right_rail_center_x": right_rail_center_x,
        "ballast_top_z": ballast_top_z,
        "sleeper_base_z": sleeper_base_z,
        "sleeper_top_z": sleeper_top_z,
        "rail_base_z": rail_base_z,
        "rail_head_top_z": rail_head_top_z,
    }


def build_railway_scene(
    sim_cfg: dict[str, Any],
    crack_cfg: dict[str, Any] | None = None,
    misalignment_cfg: dict[str, Any] | None = None,
    variant_params: dict[str, Any] | None = None,
    seed_offset: int = 0,
) -> dict[str, Any]:
    """Build the complete deterministic 250 m railway corridor inside Blender."""
    seed = int(sim_cfg["project"]["random_seed"]) + int(seed_offset)
    rng = random.Random(seed)
    v_params = variant_params or {}
    track = sim_cfg["track"]
    datum = compute_track_datum(sim_cfg)
    length_m = datum["length_m"]

    rails_col = _get_or_create_collection("Track_Rails")
    sleepers_col = _get_or_create_collection("Track_Sleepers")
    ballast_col = _get_or_create_collection("Track_Ballast")
    env_col = _get_or_create_collection("Environment")

    # Materials
    mat_rail = create_rail_steel_material()
    mat_crack = create_rail_crack_material() if crack_cfg is not None else None
    mat_concrete = create_sleeper_concrete_material(
        tint_scale=float(v_params.get("concrete_tint_scale", 1.0))
    )
    mat_fastener = create_fastener_material()
    mat_ballast = create_ballast_material(
        tint_scale=float(v_params.get("ballast_tint_scale", 1.0))
    )
    mat_ground = create_ground_material(
        tint_scale=float(v_params.get("ground_tint_scale", 1.0))
    )
    mat_post_white = create_simple_material("Mat_Marker_White", (0.86, 0.86, 0.84, 1.0), roughness=0.55)
    mat_post_yellow = create_simple_material("Mat_Marker_Yellow", (0.88, 0.66, 0.08, 1.0), roughness=0.45)
    mat_scrub = create_simple_material("Mat_Scrub_Foliage", (0.15, 0.22, 0.09, 1.0), roughness=0.90)

    # 1. Ground terrain
    ground_mesh = build_ground_mesh(
        "Terrain_Ground_Mesh", length_m, track["surroundings"], mat_ground, rng
    )
    ground_obj = bpy.data.objects.new("Terrain_Ground", ground_mesh)
    env_col.objects.link(ground_obj)

    # 2. Ballast bed
    ballast_mesh = build_ballast_bed_mesh(
        "Ballast_Bed_Mesh", length_m, track["ballast"], mat_ballast, rng
    )
    ballast_obj = bpy.data.objects.new("Ballast_Bed", ballast_mesh)
    ballast_col.objects.link(ballast_obj)

    # 3. Sleepers (shared instanced mesh for low memory footprint)
    sleeper_cfg = track["sleeper"]
    sleeper_mesh = build_sleeper_template_mesh(
        "Sleeper_PSC_Mesh",
        sleeper_cfg,
        (datum["left_rail_center_x"], datum["right_rail_center_x"]),
        float(track["rail"]["foot_width_m"]),
        mat_concrete,
        mat_fastener,
    )
    spacing = float(sleeper_cfg["spacing_m"])
    start_y = float(sleeper_cfg.get("start_offset_m", 0.20))
    half_wid = float(sleeper_cfg["width_m"]) * 0.5

    sleeper_positions: list[float] = []
    mis_affected = str(misalignment_cfg.get("affected_rails", "both")) if misalignment_cfg else ""
    idx = 0
    while True:
        y = round(start_y + idx * spacing, 6)
        if y + half_wid > length_m + 1e-6:
            break
        dx_sleeper = 0.0
        yaw_sleeper = 0.0
        if misalignment_cfg and mis_affected == "both":
            dx_sleeper = lateral_misalignment_at_y(y, misalignment_cfg)
            if abs(dx_sleeper) > 1e-6:
                slope = lateral_misalignment_slope_at_y(y, misalignment_cfg)
                yaw_sleeper = -math.atan(slope)
        s_obj = bpy.data.objects.new(f"Sleeper_{idx:04d}", sleeper_mesh)
        s_obj.location = (dx_sleeper, y, datum["sleeper_base_z"])
        if yaw_sleeper != 0.0:
            s_obj.rotation_euler = (0.0, 0.0, yaw_sleeper)
        sleepers_col.objects.link(s_obj)
        sleeper_positions.append(y)
        idx += 1

    # 4. Left & right continuous rails
    rail_cfg = track["rail"]
    crack_target = str(crack_cfg.get("target_rail", "right")) if crack_cfg else ""
    left_crack = crack_cfg if (crack_cfg and crack_target == "left") else None
    right_crack = crack_cfg if (crack_cfg and crack_target == "right") else None
    left_mis = (
        misalignment_cfg
        if (misalignment_cfg and mis_affected in ("left", "both"))
        else None
    )
    right_mis = (
        misalignment_cfg
        if (misalignment_cfg and mis_affected in ("right", "both"))
        else None
    )

    left_rail_mesh = build_rail_mesh(
        "Rail_Left_Mesh",
        datum["left_rail_center_x"],
        datum["rail_base_z"],
        length_m,
        rail_cfg,
        mat_rail,
        crack_cfg=left_crack,
        crack_material=mat_crack if left_crack else None,
        misalignment_cfg=left_mis,
    )
    left_rail_obj = bpy.data.objects.new("Rail_Left", left_rail_mesh)
    rails_col.objects.link(left_rail_obj)

    right_rail_mesh = build_rail_mesh(
        "Rail_Right_Mesh",
        datum["right_rail_center_x"],
        datum["rail_base_z"],
        length_m,
        rail_cfg,
        mat_rail,
        crack_cfg=right_crack,
        crack_material=mat_crack if right_crack else None,
        misalignment_cfg=right_mis,
    )
    right_rail_obj = bpy.data.objects.new("Rail_Right", right_rail_mesh)
    rails_col.objects.link(right_rail_obj)

    # 5. Trackside reference distance posts & low-poly countryside scrub
    surround = track["surroundings"]
    marker_count = 0
    if surround.get("include_reference_poles", True):
        post_mesh = build_marker_post_mesh("Distance_Marker_Mesh", mat_post_white, mat_post_yellow)
        pole_interval = float(surround.get("pole_interval_m", 25.0))
        pole_x = float(surround.get("pole_lateral_offset_m", 3.6))
        y_pole = 0.0
        while y_pole <= length_m + 1e-6:
            side = 1.0 if int(round(y_pole / pole_interval)) % 2 == 0 else 1.0
            m_obj = bpy.data.objects.new(f"Marker_Post_{int(round(y_pole)):03d}m", post_mesh)
            m_obj.location = (side * pole_x, y_pole, -0.15)
            env_col.objects.link(m_obj)
            marker_count += 1
            y_pole += pole_interval

    scrub_count = 0
    if surround.get("include_scrub_bushes", True):
        scrub_mesh = build_scrub_bush_mesh("Trackside_Scrub_Mesh", mat_scrub)
        target_scrubs = int(surround.get("scrub_count", 60))
        for s_idx in range(target_scrubs):
            side = -1.0 if (s_idx % 2 == 0) else 1.0
            sx = side * rng.uniform(5.8, 28.0)
            sy = rng.uniform(-10.0, length_m + 40.0)
            sz = ground_elevation_at(sx, sy) - 0.12  # Embed base slightly into ground
            scale_xy = rng.uniform(0.7, 2.2)
            scale_z = rng.uniform(0.6, 1.5)
            b_obj = bpy.data.objects.new(f"Scrub_{s_idx:03d}", scrub_mesh)
            b_obj.location = (sx, sy, sz)
            b_obj.scale = (scale_xy, scale_xy * rng.uniform(0.8, 1.3), scale_z)
            b_obj.rotation_euler = (0.0, 0.0, rng.uniform(0.0, 6.283))
            env_col.objects.link(b_obj)
            scrub_count += 1

    return {
        "seed": seed,
        "datum": datum,
        "sleeper_count": len(sleeper_positions),
        "sleeper_first_y_m": sleeper_positions[0] if sleeper_positions else None,
        "sleeper_last_y_m": sleeper_positions[-1] if sleeper_positions else None,
        "marker_post_count": marker_count,
        "scrub_count": scrub_count,
    }


def _world_bounds(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    mat = obj.matrix_world
    pts = [mat @ v.co for v in obj.data.vertices]
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    zs = [p.z for p in pts]
    return Vector((min(xs), min(ys), min(zs))), Vector((max(xs), max(ys), max(zs)))


def validate_railway_geometry(
    sim_cfg: dict[str, Any],
    misalignment_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Inspect actual Blender scene geometry to verify track dimensions and intervals."""
    track = sim_cfg["track"]
    expected_length = float(track["length_m"])
    expected_gauge = float(track["gauge_m"])
    expected_spacing = float(track["sleeper"]["spacing_m"])
    datum = compute_track_datum(sim_cfg)

    left_rail = bpy.data.objects.get("Rail_Left")
    right_rail = bpy.data.objects.get("Rail_Right")
    ballast = bpy.data.objects.get("Ballast_Bed")
    ground = bpy.data.objects.get("Terrain_Ground")
    sleepers = sorted(
        [o for o in bpy.data.objects if o.name.startswith("Sleeper_")],
        key=lambda o: o.location.y,
    )

    checks: dict[str, Any] = {}
    all_passed = True

    def record(name: str, passed: bool, details: dict[str, Any]) -> None:
        nonlocal all_passed
        checks[name] = {"passed": passed, **details}
        if not passed:
            all_passed = False

    # 1. Rails exist and span 250 m
    rails_exist = left_rail is not None and right_rail is not None
    if rails_exist:
        l_min, l_max = _world_bounds(left_rail)
        r_min, r_max = _world_bounds(right_rail)
        l_len = l_max.y - l_min.y
        r_len = r_max.y - r_min.y
        length_ok = (
            abs(l_len - expected_length) < 1e-4
            and abs(r_len - expected_length) < 1e-4
            and abs(l_min.y) < 1e-4
            and abs(r_min.y) < 1e-4
        )
        record(
            "rails_and_track_length",
            length_ok,
            {
                "expected_length_m": expected_length,
                "left_rail_span_m": [round(l_min.y, 6), round(l_max.y, 6)],
                "right_rail_span_m": [round(r_min.y, 6), round(r_max.y, 6)],
                "measured_length_m": round(l_len, 6),
            },
        )

        # 2. Measure inner gauge distance from rail head vertices in nominal track sections
        z_cutoff = datum["rail_head_top_z"] - 0.030
        mis_start = float(misalignment_cfg["start_m"]) - 0.5 if misalignment_cfg else 9999.0
        mis_end = float(misalignment_cfg["end_m"]) + 0.5 if misalignment_cfg else -9999.0

        def in_nominal_section(world_pt: Vector) -> bool:
            return world_pt.z >= z_cutoff and not (mis_start <= world_pt.y <= mis_end)

        l_head_xs = [
            (left_rail.matrix_world @ v.co).x
            for v in left_rail.data.vertices
            if in_nominal_section(left_rail.matrix_world @ v.co)
        ]
        r_head_xs = [
            (right_rail.matrix_world @ v.co).x
            for v in right_rail.data.vertices
            if in_nominal_section(right_rail.matrix_world @ v.co)
        ]
        left_inner_x = max(l_head_xs)
        right_inner_x = min(r_head_xs)
        measured_gauge = right_inner_x - left_inner_x
        gauge_ok = abs(measured_gauge - expected_gauge) < 1e-4
        record(
            "track_gauge",
            gauge_ok,
            {
                "expected_gauge_m": expected_gauge,
                "left_inner_gauge_face_x_m": round(left_inner_x, 6),
                "right_inner_gauge_face_x_m": round(right_inner_x, 6),
                "measured_gauge_m": round(measured_gauge, 6),
            },
        )
    else:
        record("rails_and_track_length", False, {"error": "Missing Rail_Left or Rail_Right"})
        record("track_gauge", False, {"error": "Cannot measure gauge without both rails"})

    # 3. Sleepers exist at expected intervals
    if len(sleepers) >= 2:
        ys = [s.location.y for s in sleepers]
        diffs = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
        max_spacing_err = max(abs(d - expected_spacing) for d in diffs)
        spacing_ok = max_spacing_err < 1e-4 and len(sleepers) >= 400
        record(
            "sleepers_interval",
            spacing_ok,
            {
                "sleeper_count": len(sleepers),
                "expected_spacing_m": expected_spacing,
                "max_spacing_error_m": round(max_spacing_err, 6),
                "first_y_m": round(ys[0], 4),
                "last_y_m": round(ys[-1], 4),
            },
        )
    else:
        record("sleepers_interval", False, {"sleeper_count": len(sleepers)})

    # 4. Ballast & ground exist
    if ballast is not None:
        b_min, b_max = _world_bounds(ballast)
        ballast_ok = abs((b_max.y - b_min.y) - expected_length) < 1e-4
        record(
            "ballast_bed",
            ballast_ok,
            {"span_y_m": [round(b_min.y, 4), round(b_max.y, 4)]},
        )
    else:
        record("ballast_bed", False, {"error": "Missing Ballast_Bed"})

    record("ground_plane", ground is not None, {"exists": ground is not None})

    return {"all_passed": all_passed, "checks": checks}

