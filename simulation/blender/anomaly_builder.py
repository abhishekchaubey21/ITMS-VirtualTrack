"""Modular procedural anomaly generators for ITMS-VirtualTrack (Milestone 3+).

Generates four distinct configurable anomalies at predefined track locations:
  - A1 (50 m)  : Non-graphic 3D animal-shaped obstruction (object detection)
  - A2 (100 m) : Procedural grain spillage pile and kernels (object detection)
  - A3 (150 m) : Controlled visual rail-head crack mask & metadata (segmentation)
  - A4 (200 m) : Smooth lateral rail misalignment profile & mask (geometric analysis)
"""

from __future__ import annotations

import math
import random
from typing import Any

import bmesh
import bpy
from mathutils import Euler, Matrix, Vector

from track_components import (
    _crack_profile_params_at_y,
    get_node_tree,
    lateral_misalignment_at_y,
    lateral_misalignment_slope_at_y,
)


def _get_or_create_subcollection(
    name: str, parent: bpy.types.Collection
) -> bpy.types.Collection:
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        parent.children.link(col)
    return col


def get_anomalies_root_collection() -> bpy.types.Collection:
    root = bpy.data.collections.get("Anomalies")
    if root is None:
        root = bpy.data.collections.new("Anomalies")
        bpy.context.scene.collection.children.link(root)
    return root


def _clear_bsdf(mat: bpy.types.Material) -> tuple[bpy.types.NodeTree, bpy.types.Node]:
    nt = get_node_tree(mat)
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (420, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (120, 0)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return nt, bsdf


def create_animal_hide_material(name: str = "Mat_A1_Animal_Hide") -> bpy.types.Material:
    """Create non-graphic procedural short-fur hide material for A1."""
    mat = bpy.data.materials.new(name=name)
    nt, bsdf = _clear_bsdf(mat)

    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.location = (-640, 0)
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.location = (-420, 60)
    noise.inputs["Scale"].default_value = 14.0
    noise.inputs["Detail"].default_value = 8.0
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])

    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.location = (-180, 100)
    ramp.color_ramp.elements[0].position = 0.28
    ramp.color_ramp.elements[0].color = (0.16, 0.11, 0.07, 1.0)
    ramp.color_ramp.elements[1].position = 0.78
    ramp.color_ramp.elements[1].color = (0.36, 0.25, 0.16, 1.0)
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    bsdf.inputs["Roughness"].default_value = 0.86

    bump = nt.nodes.new("ShaderNodeBump")
    bump.location = (-120, -140)
    bump.inputs["Strength"].default_value = 0.18
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def create_grain_material(
    grain_rgb: list[float], name: str = "Mat_A2_Grain_Spillage"
) -> bpy.types.Material:
    """Create granular golden-tan wheat/grain material contrasting with dark ballast."""
    mat = bpy.data.materials.new(name=name)
    nt, bsdf = _clear_bsdf(mat)

    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.location = (-740, 0)

    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.location = (-500, 120)
    vor.feature = "F1"
    vor.inputs["Scale"].default_value = 135.0
    nt.links.new(tc.outputs["Object"], vor.inputs["Vector"])

    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.location = (-500, -120)
    noise.inputs["Scale"].default_value = 26.0
    noise.inputs["Detail"].default_value = 6.0
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])

    r, g, b = float(grain_rgb[0]), float(grain_rgb[1]), float(grain_rgb[2])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.location = (-200, 120)
    ramp.color_ramp.elements[0].position = 0.20
    ramp.color_ramp.elements[0].color = (r * 0.68, g * 0.64, b * 0.55, 1.0)
    ramp.color_ramp.elements[1].position = 0.80
    ramp.color_ramp.elements[1].color = (min(1.0, r * 1.12), min(1.0, g * 1.10), min(1.0, b * 1.05), 1.0)
    nt.links.new(vor.outputs["Distance"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    bsdf.inputs["Roughness"].default_value = 0.76

    bump = nt.nodes.new("ShaderNodeBump")
    bump.location = (-120, -140)
    bump.inputs["Strength"].default_value = 0.52
    nt.links.new(vor.outputs["Distance"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def _add_ellipsoid(
    bm: bmesh.types.BMesh,
    center: tuple[float, float, float],
    radii: tuple[float, float, float],
    rot_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    subdivisions: int = 2,
) -> None:
    res = bmesh.ops.create_icosphere(bm, subdivisions=subdivisions, radius=1.0)
    rot_mat = Euler([math.radians(a) for a in rot_deg], "XYZ").to_matrix().to_4x4()
    scale_mat = Matrix.Diagonal((radii[0], radii[1], radii[2], 1.0))
    trans_mat = Matrix.Translation(center)
    xform = trans_mat @ rot_mat @ scale_mat
    bmesh.ops.transform(bm, matrix=xform, verts=res["verts"])


def _object_world_bbox(obj: bpy.types.Object) -> dict[str, Any]:
    mat = obj.matrix_world
    pts = [mat @ v.co for v in obj.data.vertices]
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    zs = [p.z for p in pts]
    min_pt = [round(min(xs), 4), round(min(ys), 4), round(min(zs), 4)]
    max_pt = [round(max(xs), 4), round(max(ys), 4), round(max(zs), 4)]
    dims = [
        round(max_pt[0] - min_pt[0], 4),
        round(max_pt[1] - min_pt[1], 4),
        round(max_pt[2] - min_pt[2], 4),
    ]
    center = [
        round(0.5 * (min_pt[0] + max_pt[0]), 4),
        round(0.5 * (min_pt[1] + max_pt[1]), 4),
        round(0.5 * (min_pt[2] + max_pt[2]), 4),
    ]
    return {
        "world_min_m": min_pt,
        "world_max_m": max_pt,
        "world_center_m": center,
        "world_dimensions_m": dims,
    }


def build_animal_carcass(
    a1_cfg: dict[str, Any],
    track_datum: dict[str, float],
    anomalies_root: bpy.types.Collection,
    scale_factor: float = 1.0,
    yaw_delta_deg: float = 0.0,
) -> tuple[bpy.types.Object, dict[str, Any]]:
    """Build A1: Non-graphic 3D side-lying animal obstruction near/on the track."""
    col = _get_or_create_subcollection("A1_Animal_Carcass", anomalies_root)
    mat = create_animal_hide_material()

    bm = bmesh.new()
    # Sculpted quadruped silhouette lying on its side (local +X along body length)
    _add_ellipsoid(bm, (0.00, 0.00, 0.16), (0.34, 0.17, 0.15), (0.0, 2.0, 0.0), 2)   # Torso
    _add_ellipsoid(bm, (0.14, 0.01, 0.18), (0.20, 0.16, 0.15), (0.0, -4.0, 0.0), 2)  # Ribcage
    _add_ellipsoid(bm, (-0.24, -0.02, 0.15), (0.17, 0.15, 0.13), (0.0, 5.0, 0.0), 2) # Haunch
    _add_ellipsoid(bm, (0.34, 0.03, 0.21), (0.13, 0.09, 0.09), (0.0, -22.0, 10.0), 2) # Neck
    _add_ellipsoid(bm, (0.46, 0.06, 0.24), (0.12, 0.085, 0.085), (0.0, 8.0, 18.0), 2) # Head
    _add_ellipsoid(bm, (0.56, 0.08, 0.22), (0.075, 0.055, 0.050), (0.0, 12.0, 18.0), 1) # Snout
    _add_ellipsoid(bm, (0.41, 0.12, 0.30), (0.055, 0.024, 0.022), (20.0, -15.0, 55.0), 1) # Ear L
    _add_ellipsoid(bm, (0.42, -0.01, 0.31), (0.055, 0.024, 0.022), (-15.0, -15.0, -35.0), 1) # Ear R
    # Four folded/outstretched legs resting toward sleeper plane
    _add_ellipsoid(bm, (0.20, 0.19, 0.08), (0.055, 0.15, 0.055), (0.0, 0.0, 22.0), 1)
    _add_ellipsoid(bm, (0.15, 0.13, 0.20), (0.050, 0.14, 0.050), (22.0, 0.0, 34.0), 1)
    _add_ellipsoid(bm, (-0.22, 0.19, 0.08), (0.055, 0.15, 0.055), (0.0, 0.0, -18.0), 1)
    _add_ellipsoid(bm, (-0.25, 0.11, 0.19), (0.050, 0.13, 0.050), (18.0, 0.0, -28.0), 1)
    # Tail
    _add_ellipsoid(bm, (-0.43, -0.05, 0.11), (0.11, 0.028, 0.028), (0.0, 22.0, -18.0), 1)

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

    # Normalize & scale to exact configured dimensions_m (length, width, height)
    target_dims = a1_cfg["dimensions_m"]
    xs = [v.co.x for v in bm.verts]
    ys = [v.co.y for v in bm.verts]
    zs = [v.co.z for v in bm.verts]
    raw_len = max(xs) - min(xs)
    raw_wid = max(ys) - min(ys)
    raw_hgt = max(zs) - min(zs)
    cx = 0.5 * (max(xs) + min(xs))
    cy = 0.5 * (max(ys) + min(ys))
    min_z = min(zs)

    sx = (float(target_dims["length"]) * scale_factor) / raw_len
    sy = (float(target_dims["width"]) * scale_factor) / raw_wid
    sz = (float(target_dims["height"]) * scale_factor) / raw_hgt

    for v in bm.verts:
        v.co.x = (v.co.x - cx) * sx
        v.co.y = (v.co.y - cy) * sy
        v.co.z = (v.co.z - min_z) * sz

    mesh = bpy.data.meshes.new("A1_Animal_Carcass_Mesh")
    bm.to_mesh(mesh)
    bm.free()
    mesh.materials.append(mat)
    for poly in mesh.polygons:
        poly.use_smooth = True

    obj = bpy.data.objects.new("A1_Animal_Carcass", mesh)
    pos_y = float(a1_cfg["position_m"])
    pos_x = float(a1_cfg.get("lateral_offset_m", 0.22))
    pos_z = track_datum["sleeper_top_z"] - 0.015  # Slight settle into sleeper/ballast bay
    yaw_deg = float(a1_cfg.get("yaw_deg", 32.0)) + yaw_delta_deg

    obj.location = (pos_x, pos_y, pos_z)
    obj.rotation_euler = Euler((0.0, 0.0, math.radians(yaw_deg)), "XYZ")
    col.objects.link(obj)
    bpy.context.view_layer.update()

    bbox_info = _object_world_bbox(obj)
    meta = {
        "id": str(a1_cfg["id"]),
        "class": str(a1_cfg["class"]),
        "class_id": int(a1_cfg["class_id"]),
        "detection_method": str(a1_cfg["detection_method"]),
        "yolo_enabled": True,
        "non_graphic": bool(a1_cfg.get("non_graphic", True)),
        "configured_position_m": pos_y,
        "lateral_offset_m": pos_x,
        "yaw_deg": yaw_deg,
        "configured_dimensions_m": target_dims,
        "object_name": obj.name,
        "collection_name": col.name,
        **bbox_info,
    }
    return obj, meta


def build_grain_spillage(
    a2_cfg: dict[str, Any],
    track_datum: dict[str, float],
    anomalies_root: bpy.types.Collection,
    rng: random.Random,
    scale_factor: float = 1.0,
    yaw_delta_deg: float = 0.0,
) -> tuple[bpy.types.Object, dict[str, Any]]:
    """Build A2: Procedural spilled grain mound and scattered kernel clusters."""
    col = _get_or_create_subcollection("A2_Grain_Spillage", anomalies_root)
    grain_rgb = a2_cfg.get("grain_color_rgb", [0.85, 0.72, 0.35])
    mat = create_grain_material(grain_rgb)

    dims = a2_cfg["dimensions_m"]
    length = float(dims["length"]) * scale_factor
    width = float(dims["width"]) * scale_factor
    height = float(dims["height"]) * scale_factor
    particle_count = int(a2_cfg.get("particle_count", 450))

    bm = bmesh.new()

    # 1. Procedural multi-lobed grain mound draped across sleepers and ballast
    nx, ny = 24, 30
    half_w = width * 0.5
    half_l = length * 0.5
    base_drop = track_datum["sleeper_top_z"] - track_datum["ballast_top_z"]  # ~0.13 m
    # Skirt dips slightly to ballast surface while crown rises `height` above sleeper_top_z
    grid: list[list[bmesh.types.BMVert]] = []
    for iy in range(ny + 1):
        v_norm = (iy / ny) * 2.0 - 1.0  # -1..1
        y_loc = v_norm * half_l
        row = []
        for ix in range(nx + 1):
            u_norm = (ix / nx) * 2.0 - 1.0
            x_loc = u_norm * half_w
            r2 = u_norm * u_norm + v_norm * v_norm
            lobe = 1.0 + 0.18 * math.sin(3.4 * u_norm + 1.2) * math.cos(2.7 * v_norm)
            if r2 >= 1.0:
                z_loc = -base_drop * 0.85
            else:
                cone = (1.0 - math.sqrt(r2)) ** 0.78
                z_loc = -base_drop * (1.0 - cone) + height * cone * lobe + rng.uniform(-0.005, 0.005)
            row.append(bm.verts.new((x_loc, y_loc, z_loc)))
        grid.append(row)

    for iy in range(ny):
        for ix in range(nx):
            bm.faces.new(
                (grid[iy][ix], grid[iy][ix + 1], grid[iy + 1][ix + 1], grid[iy + 1][ix])
            )

    # 2. Scattered individual grain kernel clusters around skirt and nearby bays
    for _ in range(particle_count):
        ang = rng.uniform(0.0, 2.0 * math.pi)
        rad = rng.uniform(0.35, 1.35)
        px = rad * half_w * math.cos(ang)
        py = rad * half_l * math.sin(ang)
        # Alternate height smoothly between sleeper bays and ballast surface
        pz = -base_drop * 0.72 if rng.random() < 0.6 else 0.004
        rx = rng.uniform(0.010, 0.024)
        ry = rng.uniform(0.015, 0.036)
        rz = rng.uniform(0.006, 0.016)
        _add_ellipsoid(
            bm,
            (px, py, pz + rz * 0.4),
            (rx, ry, rz),
            (0.0, 0.0, rng.uniform(0.0, 180.0)),
            subdivisions=1,
        )

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new("A2_Grain_Spillage_Mesh")
    bm.to_mesh(mesh)
    bm.free()
    mesh.materials.append(mat)
    for poly in mesh.polygons:
        poly.use_smooth = True

    obj = bpy.data.objects.new("A2_Grain_Spillage", mesh)
    pos_y = float(a2_cfg["position_m"])
    pos_x = float(a2_cfg.get("lateral_offset_m", -0.10))
    pos_z = track_datum["sleeper_top_z"]
    yaw_deg = float(a2_cfg.get("yaw_deg", 12.0)) + yaw_delta_deg

    obj.location = (pos_x, pos_y, pos_z)
    obj.rotation_euler = Euler((0.0, 0.0, math.radians(yaw_deg)), "XYZ")
    col.objects.link(obj)
    bpy.context.view_layer.update()

    bbox_info = _object_world_bbox(obj)
    meta = {
        "id": str(a2_cfg["id"]),
        "class": str(a2_cfg["class"]),
        "class_id": int(a2_cfg["class_id"]),
        "detection_method": str(a2_cfg["detection_method"]),
        "yolo_enabled": True,
        "configured_position_m": pos_y,
        "lateral_offset_m": pos_x,
        "yaw_deg": yaw_deg,
        "particle_count": particle_count,
        "configured_dimensions_m": dims,
        "object_name": obj.name,
        "collection_name": col.name,
        **bbox_info,
    }
    return obj, meta


def build_rail_crack(
    a3_cfg: dict[str, Any],
    sim_cfg: dict[str, Any],
    track_datum: dict[str, float],
    anomalies_root: bpy.types.Collection,
) -> tuple[bpy.types.Object, dict[str, Any]]:
    """Build A3 segmentation geometry matching the carved crack in the target rail.

    Note: The visible surface trough is carved directly into Rail_Right/Rail_Left
    via build_rail_mesh(..., crack_cfg=a3_cfg). A3_Rail_Crack stores exact 3D crack
    surface faces for ground-truth validation and pixel segmentation masks.
    """
    col = _get_or_create_subcollection("A3_Rail_Crack", anomalies_root)
    target_rail = str(a3_cfg.get("target_rail", "right"))
    rail_x = (
        track_datum["right_rail_center_x"]
        if target_rail == "right"
        else track_datum["left_rail_center_x"]
    )
    z_top = track_datum["rail_head_top_z"]
    hw = float(sim_cfg["track"]["rail"]["head_width_m"])
    pos_m = float(a3_cfg["position_m"])
    c_len = float(a3_cfg["crack_length_m"])
    y0 = pos_m - c_len * 0.5
    y1 = pos_m + c_len * 0.5

    n_steps = 40
    bm = bmesh.new()
    rows: list[list[bmesh.types.BMVert]] = []
    for i in range(n_steps + 1):
        y = y0 + i * (y1 - y0) / n_steps
        cx, half_w, depth = _crack_profile_params_at_y(y, a3_cfg, hw)
        eff_hw = max(0.002, half_w)
        # Offset +0.0006 m upward inside groove so mask pass cleanly clears black trough walls
        dz_lift = 0.0006
        pts = [
            (rail_x + cx + eff_hw, y, z_top + dz_lift),
            (rail_x + cx + eff_hw * 0.35, y, z_top - depth + dz_lift),
            (rail_x + cx - eff_hw * 0.35, y, z_top - depth + dz_lift),
            (rail_x + cx - eff_hw, y, z_top + dz_lift),
        ]
        rows.append([bm.verts.new(p) for p in pts])

    for i in range(n_steps):
        r0 = rows[i]
        r1 = rows[i + 1]
        for j in range(3):
            bm.faces.new((r0[j], r0[j + 1], r1[j + 1], r1[j]))

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new("A3_Rail_Crack_MaskMesh")
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new("A3_Rail_Crack", mesh)
    # Hidden in beauty render because Rail_Right renders the carved crack natively;
    # ground_truth_generator temporarily enables it when rendering binary segmentation masks.
    obj.hide_render = True
    col.objects.link(obj)
    bpy.context.view_layer.update()

    bbox_info = _object_world_bbox(obj)
    meta = {
        "id": str(a3_cfg["id"]),
        "class": str(a3_cfg["class"]),
        "class_id": int(a3_cfg["class_id"]),
        "detection_method": str(a3_cfg["detection_method"]),
        "yolo_enabled": False,
        "engineering_disclaimer": (
            "Controlled synthetic visual rail-head crack defect for CV segmentation; "
            "not a structural metallurgical fracture model."
        ),
        "configured_position_m": pos_m,
        "target_rail": target_rail,
        "target_rail_center_x_m": round(rail_x, 4),
        "crack_length_m": c_len,
        "crack_width_m": float(a3_cfg["crack_width_m"]),
        "crack_depth_m": float(a3_cfg["crack_depth_m"]),
        "object_name": obj.name,
        "collection_name": col.name,
        **bbox_info,
    }
    return obj, meta


def build_rail_misalignment(
    a4_cfg: dict[str, Any],
    sim_cfg: dict[str, Any],
    track_datum: dict[str, float],
    anomalies_root: bpy.types.Collection,
) -> tuple[bpy.types.Object, dict[str, Any]]:
    """Build A4 geometry-mask shell and record smooth lateral displacement profile.

    Note: The actual physical lateral track buckle is built smoothly into Rail_Left,
    Rail_Right, and Sleepers across [start_m, end_m] without discontinuities.
    """
    col = _get_or_create_subcollection("A4_Rail_Misalignment", anomalies_root)
    start_m = float(a4_cfg["start_m"])
    end_m = float(a4_cfg["end_m"])
    pos_m = float(a4_cfg["position_m"])
    max_dev = float(a4_cfg["max_lateral_deviation_m"])
    affected_rails = str(a4_cfg.get("affected_rails", "both"))

    hw = float(sim_cfg["track"]["rail"]["head_width_m"])
    hh = float(sim_cfg["track"]["rail"]["head_height_m"])
    z_top = track_datum["rail_head_top_z"]

    rail_centers: list[tuple[str, float]] = []
    if affected_rails in ("left", "both"):
        rail_centers.append(("left", track_datum["left_rail_center_x"]))
    if affected_rails in ("right", "both"):
        rail_centers.append(("right", track_datum["right_rail_center_x"]))

    # Build slightly expanded rail-head strips over [start_m, end_m] for occlusion-aware mask rendering
    bm = bmesh.new()
    n_steps = 32
    expand = 0.0025
    for _side, base_cx in rail_centers:
        rings: list[list[bmesh.types.BMVert]] = []
        for i in range(n_steps + 1):
            y = start_m + i * (end_m - start_m) / n_steps
            dx = lateral_misalignment_at_y(y, a4_cfg)
            cx = base_cx + dx
            pts = [
                (cx - hw * 0.5 - expand, y, z_top - hh),
                (cx + hw * 0.5 + expand, y, z_top - hh),
                (cx + hw * 0.5 + expand, y, z_top + expand),
                (cx - hw * 0.5 - expand, y, z_top + expand),
            ]
            rings.append([bm.verts.new(p) for p in pts])
        for i in range(n_steps):
            r0, r1 = rings[i], rings[i + 1]
            for j in range(4):
                k = (j + 1) % 4
                bm.faces.new((r0[j], r0[k], r1[k], r1[j]))
        bm.faces.new(list(reversed(rings[0])))
        bm.faces.new(rings[-1])

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new("A4_Rail_Misalignment_MaskMesh")
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new("A4_Rail_Misalignment_Mask", mesh)
    obj.hide_render = True
    col.objects.link(obj)
    bpy.context.view_layer.update()

    # Sample physical displacement profile at 0.25 m intervals
    profile_samples: list[dict[str, float]] = []
    sample_step = 0.25
    num_samples = int(round((end_m - start_m) / sample_step)) + 1
    for idx in range(num_samples):
        y = start_m + idx * sample_step
        dx = lateral_misalignment_at_y(y, a4_cfg)
        slope = lateral_misalignment_slope_at_y(y, a4_cfg)
        profile_samples.append(
            {
                "y_m": round(y, 3),
                "lateral_shift_m": round(dx, 6),
                "left_rail_center_x_m": round(
                    track_datum["left_rail_center_x"] + (dx if affected_rails in ("left", "both") else 0.0),
                    6,
                ),
                "right_rail_center_x_m": round(
                    track_datum["right_rail_center_x"] + (dx if affected_rails in ("right", "both") else 0.0),
                    6,
                ),
                "tangent_angle_deg": round(math.degrees(math.atan(slope)), 4),
            }
        )

    bbox_info = _object_world_bbox(obj)
    meta = {
        "id": str(a4_cfg["id"]),
        "class": str(a4_cfg["class"]),
        "class_id": int(a4_cfg["class_id"]),
        "detection_method": str(a4_cfg["detection_method"]),
        "yolo_enabled": False,
        "configured_position_m": pos_m,
        "affected_span_m": float(a4_cfg["affected_span_m"]),
        "start_m": start_m,
        "end_m": end_m,
        "max_lateral_deviation_m": max_dev,
        "affected_rails": affected_rails,
        "continuity": "C1 smooth raised-cosine sin^2(pi * u) profile (zero displacement/slope discontinuity at boundaries)",
        "displacement_profile": profile_samples,
        "object_name": obj.name,
        "collection_name": col.name,
        **bbox_info,
    }
    return obj, meta


def build_anomalies(
    active_ids: list[str],
    anomaly_cfg: dict[str, Any],
    sim_cfg: dict[str, Any],
    track_datum: dict[str, float],
    variant_params: dict[str, Any] | None = None,
    seed_offset: int = 0,
) -> dict[str, Any]:
    """Build requested anomalies (subset or all of A1..A4) and return metadata dict."""
    if not active_ids:
        return {}

    seed = int(sim_cfg["project"]["random_seed"]) + 1007 + int(seed_offset)
    rng = random.Random(seed)
    v_params = variant_params or {}
    scale_factor = float(v_params.get("anomaly_scale_factor", 1.0))
    yaw_delta = float(v_params.get("anomaly_yaw_delta_deg", 0.0))
    lat_jitter = float(v_params.get("anomaly_lateral_jitter_m", 0.0))

    anomalies_root = get_anomalies_root_collection()
    cfgs = anomaly_cfg["anomalies"]
    built_meta: dict[str, Any] = {}

    if "A1" in active_ids:
        a1_cfg = dict(cfgs["A1"])
        a1_cfg["lateral_offset_m"] = float(a1_cfg.get("lateral_offset_m", 0.22)) + lat_jitter
        _, meta_a1 = build_animal_carcass(
            a1_cfg,
            track_datum,
            anomalies_root,
            scale_factor * float(v_params.get("a1_scale_factor", 1.0)),
            yaw_delta + float(v_params.get("a1_yaw_delta_deg", 0.0)),
        )
        built_meta["A1"] = meta_a1

    if "A2" in active_ids:
        a2_cfg = dict(cfgs["A2"])
        a2_cfg["lateral_offset_m"] = float(a2_cfg.get("lateral_offset_m", -0.10)) - 0.6 * lat_jitter
        _, meta_a2 = build_grain_spillage(
            a2_cfg,
            track_datum,
            anomalies_root,
            rng,
            scale_factor * float(v_params.get("a2_scale_factor", 1.0)),
            yaw_delta + float(v_params.get("a2_yaw_delta_deg", 0.0)),
        )
        built_meta["A2"] = meta_a2

    if "A3" in active_ids:
        a3_cfg = dict(cfgs["A3"])
        a3_scale = float(v_params.get("a3_scale_factor", 1.0))
        a3_cfg["crack_length_m"] = float(a3_cfg["crack_length_m"]) * a3_scale
        a3_cfg["crack_width_m"] = float(a3_cfg["crack_width_m"]) * a3_scale
        _, meta_a3 = build_rail_crack(a3_cfg, sim_cfg, track_datum, anomalies_root)
        built_meta["A3"] = meta_a3

    if "A4" in active_ids:
        a4_cfg = dict(cfgs["A4"])
        a4_cfg["max_lateral_deviation_m"] = float(a4_cfg["max_lateral_deviation_m"]) * float(
            v_params.get("a4_deviation_scale", 1.0)
        )
        _, meta_a4 = build_rail_misalignment(a4_cfg, sim_cfg, track_datum, anomalies_root)
        built_meta["A4"] = meta_a4

    return built_meta


def validate_anomalies_in_scene(
    expected_ids: list[str],
    anomaly_meta: dict[str, Any],
    anomaly_cfg: dict[str, Any],
    track_datum: dict[str, float],
) -> dict[str, Any]:
    """Verify expected anomalies exist at configured positions and unexpected ones are absent."""
    expected_objs = {
        "A1": "A1_Animal_Carcass",
        "A2": "A2_Grain_Spillage",
        "A3": "A3_Rail_Crack",
        "A4": "A4_Rail_Misalignment_Mask",
    }
    checks: dict[str, Any] = {}
    all_passed = True

    def record(name: str, passed: bool, details: dict[str, Any]) -> None:
        nonlocal all_passed
        checks[name] = {"passed": passed, **details}
        if not passed:
            all_passed = False

    # Verify absence of unrequested anomalies (critical for baseline & single-anomaly scenes)
    for aid, obj_name in expected_objs.items():
        obj = bpy.data.objects.get(obj_name)
        should_exist = aid in expected_ids
        if not should_exist:
            record(
                f"{aid}_absent",
                obj is None,
                {"expected_absent": True, "found": obj is not None},
            )
            continue

        if obj is None or aid not in anomaly_meta:
            record(f"{aid}_present_and_positioned", False, {"error": f"Missing {obj_name}"})
            continue

        target_pos = float(anomaly_cfg["anomalies"][aid]["position_m"])
        center_y = float(anomaly_meta[aid]["world_center_m"][1])
        pos_err = abs(center_y - target_pos)
        pos_ok = pos_err <= 0.15
        details = {
            "expected_position_m": target_pos,
            "measured_center_y_m": center_y,
            "position_error_m": round(pos_err, 4),
            "world_center_m": anomaly_meta[aid]["world_center_m"],
            "world_dimensions_m": anomaly_meta[aid]["world_dimensions_m"],
        }

        if aid == "A4":
            # Measure actual deformed Rail_Right centerline peak deviation near 200 m
            right_rail = bpy.data.objects.get("Rail_Right")
            a4_cfg = anomaly_cfg["anomalies"]["A4"]
            expected_dev = float(a4_cfg["max_lateral_deviation_m"])
            if right_rail is not None:
                # Top-head center x around Y=200m
                near_verts = [
                    (right_rail.matrix_world @ v.co)
                    for v in right_rail.data.vertices
                    if abs((right_rail.matrix_world @ v.co).y - target_pos) <= 0.6
                    and (right_rail.matrix_world @ v.co).z >= track_datum["rail_head_top_z"] - 0.01
                ]
                avg_x = sum(p.x for p in near_verts) / max(1, len(near_verts))
                measured_dev = avg_x - track_datum["right_rail_center_x"]
                dev_ok = abs(measured_dev - expected_dev) < 0.005
                pos_ok = pos_ok and dev_ok
                details["expected_max_deviation_m"] = expected_dev
                details["measured_peak_deviation_m"] = round(measured_dev, 5)

        record(f"{aid}_present_and_positioned", pos_ok, details)

    return {"all_passed": all_passed, "checks": checks}

