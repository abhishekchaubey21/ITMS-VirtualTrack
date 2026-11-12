"""Procedural railway track components and materials for Blender 5.3+.

All geometry uses metric world coordinates (+Y longitudinal, +X lateral, +Z up).
Materials are 100% procedural (no external textures) and deterministic.
"""

from __future__ import annotations

import math
import random
from typing import Any

import bmesh
import bpy


def get_node_tree(datablock: Any) -> bpy.types.NodeTree:
    """Return a datablock's shader node tree without triggering Blender 6 deprecation warnings."""
    if datablock.node_tree is None:
        datablock.use_nodes = True
    return datablock.node_tree


def _clear_and_init_bsdf(mat: bpy.types.Material) -> tuple[bpy.types.NodeTree, bpy.types.Node, bpy.types.Node]:
    nt = get_node_tree(mat)
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (450, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (120, 0)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return nt, bsdf, out


def create_rail_steel_material(name: str = "Mat_Rail_Steel") -> bpy.types.Material:
    """Create procedural rail steel with polished running band on top and weathered sides."""
    mat = bpy.data.materials.new(name=name)
    nt, bsdf, _ = _clear_and_init_bsdf(mat)

    geom = nt.nodes.new("ShaderNodeNewGeometry")
    geom.location = (-750, 120)

    sep_norm = nt.nodes.new("ShaderNodeSeparateXYZ")
    sep_norm.location = (-540, 220)
    nt.links.new(geom.outputs["Normal"], sep_norm.inputs["Vector"])

    # Top running surface has Normal.Z close to 1.0 -> bright polished steel
    top_ramp = nt.nodes.new("ShaderNodeValToRGB")
    top_ramp.location = (-320, 240)
    top_ramp.color_ramp.elements[0].position = 0.82
    top_ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    top_ramp.color_ramp.elements[1].position = 0.96
    top_ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    nt.links.new(sep_norm.outputs["Z"], top_ramp.inputs["Fac"])

    tex_coord = nt.nodes.new("ShaderNodeTexCoord")
    tex_coord.location = (-750, -140)
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.location = (-540, -140)
    noise.inputs["Scale"].default_value = 18.0
    noise.inputs["Detail"].default_value = 6.0
    nt.links.new(tex_coord.outputs["Object"], noise.inputs["Vector"])

    mix_color = nt.nodes.new("ShaderNodeMix")
    mix_color.data_type = "RGBA"
    mix_color.location = (-80, 120)
    # Factor = top_ramp -> A: weathered iron-oxide side, B: polished silver running surface
    mix_color.inputs["A"].default_value = (0.14, 0.095, 0.075, 1.0)
    mix_color.inputs["B"].default_value = (0.58, 0.60, 0.63, 1.0)
    nt.links.new(top_ramp.outputs["Color"], mix_color.inputs["Factor"])
    nt.links.new(mix_color.outputs["Result"], bsdf.inputs["Base Color"])

    mix_rough = nt.nodes.new("ShaderNodeMix")
    mix_rough.data_type = "FLOAT"
    mix_rough.location = (-80, -100)
    mix_rough.inputs["A"].default_value = 0.68
    mix_rough.inputs["B"].default_value = 0.18
    nt.links.new(top_ramp.outputs["Color"], mix_rough.inputs["Factor"])
    nt.links.new(mix_rough.outputs["Result"], bsdf.inputs["Roughness"])

    bsdf.inputs["Metallic"].default_value = 0.88

    bump = nt.nodes.new("ShaderNodeBump")
    bump.location = (-80, -280)
    bump.inputs["Strength"].default_value = 0.06
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    return mat


def create_sleeper_concrete_material(
    name: str = "Mat_Sleeper_Concrete", tint_scale: float = 1.0
) -> bpy.types.Material:
    """Create procedural prestressed concrete material for railway sleepers."""
    mat = bpy.data.materials.new(name=name)
    nt, bsdf, _ = _clear_and_init_bsdf(mat)

    tex_coord = nt.nodes.new("ShaderNodeTexCoord")
    tex_coord.location = (-680, 0)

    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.location = (-460, 80)
    noise.inputs["Scale"].default_value = 28.0
    noise.inputs["Detail"].default_value = 8.0
    noise.inputs["Roughness"].default_value = 0.65
    nt.links.new(tex_coord.outputs["Object"], noise.inputs["Vector"])

    ts = float(tint_scale)
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.location = (-220, 120)
    ramp.color_ramp.elements[0].position = 0.25
    ramp.color_ramp.elements[0].color = (min(1.0, 0.28 * ts), min(1.0, 0.27 * ts), min(1.0, 0.25 * ts), 1.0)
    ramp.color_ramp.elements[1].position = 0.78
    ramp.color_ramp.elements[1].color = (min(1.0, 0.48 * ts), min(1.0, 0.46 * ts), min(1.0, 0.43 * ts), 1.0)
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    bsdf.inputs["Roughness"].default_value = 0.88

    bump = nt.nodes.new("ShaderNodeBump")
    bump.location = (-160, -140)
    bump.inputs["Strength"].default_value = 0.18
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    return mat


def create_fastener_material(name: str = "Mat_Fastener_Iron") -> bpy.types.Material:
    """Create dark cast-iron clip/pad material for sleeper rail seats."""
    mat = bpy.data.materials.new(name=name)
    _, bsdf, _ = _clear_and_init_bsdf(mat)
    bsdf.inputs["Base Color"].default_value = (0.09, 0.085, 0.08, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.72
    bsdf.inputs["Roughness"].default_value = 0.56
    return mat


def create_ballast_material(
    name: str = "Mat_Ballast_Gravel", tint_scale: float = 1.0
) -> bpy.types.Material:
    """Create procedural crushed granite ballast material using multi-scale Voronoi stones."""
    mat = bpy.data.materials.new(name=name)
    nt, bsdf, _ = _clear_and_init_bsdf(mat)

    tex_coord = nt.nodes.new("ShaderNodeTexCoord")
    tex_coord.location = (-860, 0)

    vor_coarse = nt.nodes.new("ShaderNodeTexVoronoi")
    vor_coarse.location = (-600, 180)
    vor_coarse.feature = "F1"
    vor_coarse.inputs["Scale"].default_value = 24.0
    nt.links.new(tex_coord.outputs["Object"], vor_coarse.inputs["Vector"])

    vor_fine = nt.nodes.new("ShaderNodeTexVoronoi")
    vor_fine.location = (-600, -100)
    vor_fine.feature = "DISTANCE_TO_EDGE"
    vor_fine.inputs["Scale"].default_value = 24.0
    nt.links.new(tex_coord.outputs["Object"], vor_fine.inputs["Vector"])

    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.location = (-600, -320)
    noise.inputs["Scale"].default_value = 45.0
    noise.inputs["Detail"].default_value = 6.0
    nt.links.new(tex_coord.outputs["Object"], noise.inputs["Vector"])

    # Stone-to-stone color variation from Voronoi Color output (R channel)
    sep_col = nt.nodes.new("ShaderNodeSeparateColor")
    sep_col.location = (-380, 180)
    nt.links.new(vor_coarse.outputs["Color"], sep_col.inputs["Color"])

    ts = float(tint_scale)
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.location = (-160, 200)
    ramp.color_ramp.elements[0].position = 0.15
    ramp.color_ramp.elements[0].color = (min(1.0, 0.11 * ts), min(1.0, 0.105 * ts), min(1.0, 0.10 * ts), 1.0)
    mid = ramp.color_ramp.elements.new(0.55)
    mid.color = (min(1.0, 0.21 * ts), min(1.0, 0.195 * ts), min(1.0, 0.18 * ts), 1.0)
    ramp.color_ramp.elements[2].position = 0.90
    ramp.color_ramp.elements[2].color = (min(1.0, 0.31 * ts), min(1.0, 0.285 * ts), min(1.0, 0.255 * ts), 1.0)
    nt.links.new(sep_col.outputs["Red"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    bsdf.inputs["Roughness"].default_value = 0.92

    # Combine Voronoi edge relief and fine grain for strong gravel bump
    mix_height = nt.nodes.new("ShaderNodeMix")
    mix_height.data_type = "FLOAT"
    mix_height.location = (-300, -180)
    mix_height.inputs["Factor"].default_value = 0.35
    nt.links.new(vor_fine.outputs["Distance"], mix_height.inputs["A"])
    nt.links.new(noise.outputs["Fac"], mix_height.inputs["B"])

    bump = nt.nodes.new("ShaderNodeBump")
    bump.location = (-80, -180)
    bump.inputs["Strength"].default_value = 0.72
    bump.inputs["Distance"].default_value = 0.04
    nt.links.new(mix_height.outputs["Result"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    return mat


def create_ground_material(
    name: str = "Mat_Terrain_Ground", tint_scale: float = 1.0
) -> bpy.types.Material:
    """Create procedural dry countryside soil and grass material."""
    mat = bpy.data.materials.new(name=name)
    nt, bsdf, _ = _clear_and_init_bsdf(mat)

    tex_coord = nt.nodes.new("ShaderNodeTexCoord")
    tex_coord.location = (-700, 0)

    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.location = (-480, 100)
    noise.inputs["Scale"].default_value = 0.35
    noise.inputs["Detail"].default_value = 8.0
    nt.links.new(tex_coord.outputs["Object"], noise.inputs["Vector"])

    gs = float(tint_scale)
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.location = (-220, 120)
    ramp.color_ramp.elements[0].position = 0.25
    ramp.color_ramp.elements[0].color = (min(1.0, 0.14 * gs), min(1.0, 0.12 * gs), min(1.0, 0.08 * gs), 1.0)
    mid = ramp.color_ramp.elements.new(0.55)
    mid.color = (min(1.0, 0.19 * gs), min(1.0, 0.21 * gs), min(1.0, 0.10 * gs), 1.0)
    ramp.color_ramp.elements[2].position = 0.85
    ramp.color_ramp.elements[2].color = (min(1.0, 0.26 * gs), min(1.0, 0.24 * gs), min(1.0, 0.14 * gs), 1.0)
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    bsdf.inputs["Roughness"].default_value = 0.95

    bump = nt.nodes.new("ShaderNodeBump")
    bump.location = (-160, -140)
    bump.inputs["Strength"].default_value = 0.25
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    return mat


def create_simple_material(
    name: str, rgba: tuple[float, float, float, float], roughness: float = 0.6, metallic: float = 0.0
) -> bpy.types.Material:
    """Create a clean solid-color Principled BSDF material."""
    mat = bpy.data.materials.new(name=name)
    _, bsdf, _ = _clear_and_init_bsdf(mat)
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    return mat


def create_rail_crack_material(name: str = "Mat_Rail_Crack_Defect") -> bpy.types.Material:
    """Create dark oxidized carbon-steel fracture material for the simulated rail crack."""
    mat = bpy.data.materials.new(name=name)
    nt, bsdf, _ = _clear_and_init_bsdf(mat)

    tex_coord = nt.nodes.new("ShaderNodeTexCoord")
    tex_coord.location = (-600, 0)
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.location = (-380, 0)
    noise.inputs["Scale"].default_value = 95.0
    noise.inputs["Detail"].default_value = 8.0
    nt.links.new(tex_coord.outputs["Object"], noise.inputs["Vector"])

    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.location = (-160, 80)
    ramp.color_ramp.elements[0].position = 0.25
    ramp.color_ramp.elements[0].color = (0.015, 0.012, 0.012, 1.0)
    ramp.color_ramp.elements[1].position = 0.85
    ramp.color_ramp.elements[1].color = (0.09, 0.045, 0.025, 1.0)
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    bsdf.inputs["Metallic"].default_value = 0.25
    bsdf.inputs["Roughness"].default_value = 0.92

    bump = nt.nodes.new("ShaderNodeBump")
    bump.location = (-140, -140)
    bump.inputs["Strength"].default_value = 0.65
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def lateral_misalignment_at_y(
    y: float, misalignment_cfg: dict[str, Any] | None
) -> float:
    """Return smooth raised-cosine lateral shift (m) at longitudinal coordinate y."""
    if not misalignment_cfg:
        return 0.0
    start_m = float(misalignment_cfg["start_m"])
    end_m = float(misalignment_cfg["end_m"])
    if y <= start_m or y >= end_m or end_m <= start_m:
        return 0.0
    u = (y - start_m) / (end_m - start_m)
    max_dev = float(misalignment_cfg["max_lateral_deviation_m"])
    return max_dev * (math.sin(math.pi * u) ** 2)


def lateral_misalignment_slope_at_y(
    y: float, misalignment_cfg: dict[str, Any] | None
) -> float:
    """Return derivative dx/dy of the lateral misalignment curve at y."""
    if not misalignment_cfg:
        return 0.0
    start_m = float(misalignment_cfg["start_m"])
    end_m = float(misalignment_cfg["end_m"])
    span = end_m - start_m
    if y <= start_m or y >= end_m or span <= 0.0:
        return 0.0
    u = (y - start_m) / span
    max_dev = float(misalignment_cfg["max_lateral_deviation_m"])
    return max_dev * (math.pi / span) * math.sin(2.0 * math.pi * u)


def _crack_profile_params_at_y(
    y: float, crack_cfg: dict[str, Any] | None, rail_head_width: float
) -> tuple[float, float, float]:
    """Return (center_x_offset, half_width, depth) for a localized rail-head crack at y."""
    if not crack_cfg:
        return 0.0, 0.0, 0.0
    pos_m = float(crack_cfg["position_m"])
    c_len = float(crack_cfg["crack_length_m"])
    half_len = c_len * 0.5
    if y <= pos_m - half_len or y >= pos_m + half_len:
        return 0.0, 0.0, 0.0

    u = (y - (pos_m - half_len)) / c_len  # 0..1 across crack
    envelope = math.sin(math.pi * u) ** 0.55
    half_w = 0.5 * float(crack_cfg["crack_width_m"]) * envelope
    depth = float(crack_cfg["crack_depth_m"]) * envelope
    # Deterministic jagged lateral wander inside rail head bounds
    wander_amp = max(0.004, (rail_head_width * 0.5) - half_w - 0.008)
    cx = wander_amp * (0.6 * math.sin(5.0 * math.pi * u) + 0.4 * math.sin(13.0 * math.pi * u))
    return cx, half_w, depth


def build_rail_mesh(
    name: str,
    x_center: float,
    z_base: float,
    length_m: float,
    rail_cfg: dict[str, Any],
    material: bpy.types.Material,
    segment_length_m: float = 0.5,
    crack_cfg: dict[str, Any] | None = None,
    crack_material: bpy.types.Material | None = None,
    misalignment_cfg: dict[str, Any] | None = None,
) -> bpy.types.Mesh:
    """Build an extruded I-profile railway rail spanning Y in [0.0, length_m].

    Supports optional localized rail-head crack groove (A3) and smooth lateral
    track misalignment (A4) while preserving exact topology elsewhere.
    """
    h = float(rail_cfg["height_m"])
    hw = float(rail_cfg["head_width_m"])
    hh = float(rail_cfg["head_height_m"])
    fw = float(rail_cfg["foot_width_m"])
    fh = float(rail_cfg["foot_height_m"])
    wt = float(rail_cfg["web_thickness_m"])

    chamfer_x = 0.004
    chamfer_z = 0.004

    # Build deterministic longitudinal Y stations, refining around crack if active
    base_steps = max(1, int(round(length_m / segment_length_m)))
    y_stations = sorted({round(i * (length_m / base_steps), 6) for i in range(base_steps + 1)})
    use_crack_profile = crack_cfg is not None
    if use_crack_profile:
        pos_m = float(crack_cfg["position_m"])
        half_span = float(crack_cfg["crack_length_m"]) * 0.5 + 0.10
        fine_n = 48
        y0 = max(0.0, pos_m - half_span)
        y1 = min(length_m, pos_m + half_span)
        extra = [round(y0 + k * (y1 - y0) / fine_n, 6) for k in range(fine_n + 1)]
        y_stations = sorted(set(y_stations).union(extra))

    def ring_points_at_y(y: float) -> list[tuple[float, float]]:
        dx_mis = lateral_misalignment_at_y(y, misalignment_cfg)
        top_r = hw * 0.5 - chamfer_x
        top_l = -hw * 0.5 + chamfer_x

        if not use_crack_profile:
            pts_xz = [
                (-fw * 0.5, 0.0),
                (+fw * 0.5, 0.0),
                (+fw * 0.5, fh * 0.55),
                (+wt * 0.5, fh),
                (+wt * 0.5, h - hh),
                (+hw * 0.5, h - hh * 0.78),
                (+hw * 0.5, h - chamfer_z),
                (top_r, h),
                (top_l, h),
                (-hw * 0.5, h - chamfer_z),
                (-hw * 0.5, h - hh * 0.78),
                (-wt * 0.5, h - hh),
                (-wt * 0.5, fh),
                (-fw * 0.5, fh * 0.55),
            ]
        else:
            cx, half_w, depth = _crack_profile_params_at_y(y, crack_cfg, hw)
            eff_half_w = max(0.003, half_w)
            cr_r = min(top_r - 0.002, cx + eff_half_w)
            cr_l = max(top_l + 0.002, cx - eff_half_w)
            cr_bot_r = min(cr_r - 0.0005, cx + eff_half_w * 0.35)
            cr_bot_l = max(cr_l + 0.0005, cx - eff_half_w * 0.35)
            pts_xz = [
                (-fw * 0.5, 0.0),
                (+fw * 0.5, 0.0),
                (+fw * 0.5, fh * 0.55),
                (+wt * 0.5, fh),
                (+wt * 0.5, h - hh),
                (+hw * 0.5, h - hh * 0.78),
                (+hw * 0.5, h - chamfer_z),
                (top_r, h),
                (cr_r, h),
                (cr_bot_r, h - depth),
                (cr_bot_l, h - depth),
                (cr_l, h),
                (top_l, h),
                (-hw * 0.5, h - chamfer_z),
                (-hw * 0.5, h - hh * 0.78),
                (-wt * 0.5, h - hh),
                (-wt * 0.5, fh),
                (-fw * 0.5, fh * 0.55),
            ]
        return [(x_center + dx_mis + px, z_base + pz) for (px, pz) in pts_xz]

    n_pts = 18 if use_crack_profile else 14
    crack_face_indices = {8, 9, 10} if use_crack_profile else set()
    bm = bmesh.new()
    rings: list[list[bmesh.types.BMVert]] = []
    for y in y_stations:
        ring = [bm.verts.new((wx, y, wz)) for (wx, wz) in ring_points_at_y(y)]
        rings.append(ring)

    bm.verts.ensure_lookup_table()

    for seg_idx in range(len(y_stations) - 1):
        r0 = rings[seg_idx]
        r1 = rings[seg_idx + 1]
        y_mid = 0.5 * (y_stations[seg_idx] + y_stations[seg_idx + 1])
        _, _, depth_mid = _crack_profile_params_at_y(y_mid, crack_cfg, hw)
        for i in range(n_pts):
            j = (i + 1) % n_pts
            face = bm.faces.new((r0[i], r0[j], r1[j], r1[i]))
            if crack_material is not None and depth_mid > 1e-4 and i in crack_face_indices:
                face.material_index = 1

    # End caps at Y=0 and Y=length_m
    bm.faces.new(list(reversed(rings[0])))
    bm.faces.new(rings[-1])

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    mesh.materials.append(material)
    if crack_material is not None:
        mesh.materials.append(crack_material)
    for poly in mesh.polygons:
        poly.use_smooth = False
    return mesh


def build_sleeper_template_mesh(
    name: str,
    sleeper_cfg: dict[str, Any],
    rail_x_centers: tuple[float, float],
    rail_foot_width_m: float,
    concrete_mat: bpy.types.Material,
    fastener_mat: bpy.types.Material,
) -> bpy.types.Mesh:
    """Create a tapered PSC concrete sleeper template mesh (origin at bottom center)."""
    s_len = float(sleeper_cfg["length_m"])
    s_wid = float(sleeper_cfg["width_m"])
    s_hgt = float(sleeper_cfg["height_m"])

    top_len = s_len - 0.14
    top_wid = s_wid - 0.05

    bm = bmesh.new()

    # Main concrete sleeper frustum box
    b_verts = [
        bm.verts.new((-s_len * 0.5, -s_wid * 0.5, 0.0)),
        bm.verts.new((+s_len * 0.5, -s_wid * 0.5, 0.0)),
        bm.verts.new((+s_len * 0.5, +s_wid * 0.5, 0.0)),
        bm.verts.new((-s_len * 0.5, +s_wid * 0.5, 0.0)),
    ]
    t_verts = [
        bm.verts.new((-top_len * 0.5, -top_wid * 0.5, s_hgt)),
        bm.verts.new((+top_len * 0.5, -top_wid * 0.5, s_hgt)),
        bm.verts.new((+top_len * 0.5, +top_wid * 0.5, s_hgt)),
        bm.verts.new((-top_len * 0.5, +top_wid * 0.5, s_hgt)),
    ]

    bm.faces.new((b_verts[3], b_verts[2], b_verts[1], b_verts[0]))
    bm.faces.new((t_verts[0], t_verts[1], t_verts[2], t_verts[3]))
    for i in range(4):
        j = (i + 1) % 4
        bm.faces.new((b_verts[i], b_verts[j], t_verts[j], t_verts[i]))

    # Rail-seat clip shoulders (material index 1) on either side of each rail foot
    clip_w = 0.045
    clip_y = 0.09
    clip_h = 0.028
    half_foot = rail_foot_width_m * 0.5
    for rx in rail_x_centers:
        for side in (-1.0, 1.0):
            cx = rx + side * (half_foot + clip_w * 0.5 + 0.004)
            res = bmesh.ops.create_cube(bm, size=1.0)
            for v in res["verts"]:
                v.co.x = cx + v.co.x * clip_w
                v.co.y = v.co.y * clip_y
                v.co.z = s_hgt + (v.co.z + 0.5) * clip_h
            for f in bm.faces:
                if all(v in res["verts"] for v in f.verts):
                    f.material_index = 1

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    mesh.materials.append(concrete_mat)
    mesh.materials.append(fastener_mat)
    return mesh


def build_ballast_bed_mesh(
    name: str,
    length_m: float,
    ballast_cfg: dict[str, Any],
    material: bpy.types.Material,
    rng: random.Random,
) -> bpy.types.Mesh:
    """Create trapezoidal ballast bed spanning Y in [0.0, length_m] with subtle relief."""
    h = float(ballast_cfg["height_m"])
    tw = float(ballast_cfg["top_width_m"])
    bw = float(ballast_cfg["bottom_width_m"])
    step_y = float(ballast_cfg.get("subdivision_y_m", 1.0))
    noise_amp = float(ballast_cfg.get("noise_amplitude_m", 0.012))

    # Cross-section lateral stations from left toe to right toe
    x_stations = [
        (-bw * 0.5, 0.0, False),
        (-tw * 0.5, h, True),
        (-tw * 0.30, h, True),
        (0.0, h + 0.005, True),
        (+tw * 0.30, h, True),
        (+tw * 0.5, h, True),
        (+bw * 0.5, 0.0, False),
    ]

    num_segments = max(1, int(round(length_m / step_y)))
    actual_step = length_m / num_segments

    bm = bmesh.new()
    rows: list[list[bmesh.types.BMVert]] = []
    for seg_idx in range(num_segments + 1):
        y = seg_idx * actual_step
        row = []
        for x, z, allow_jitter in x_stations:
            dz = rng.uniform(-noise_amp, noise_amp) if allow_jitter else 0.0
            row.append(bm.verts.new((x, y, max(0.0, z + dz))))
        rows.append(row)

    n_cols = len(x_stations)
    for seg_idx in range(num_segments):
        r0 = rows[seg_idx]
        r1 = rows[seg_idx + 1]
        for c in range(n_cols - 1):
            bm.faces.new((r0[c], r0[c + 1], r1[c + 1], r1[c]))

    # Close underside/end caps minimally
    bm.faces.new(list(reversed(rows[0])))
    bm.faces.new(rows[-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    mesh.materials.append(material)
    return mesh


def ground_elevation_at(x: float, y: float) -> float:
    """Deterministic base terrain elevation at (x, y) outside the ballast corridor."""
    if abs(x) <= 2.6:
        return 0.0
    if abs(x) <= 4.5:
        return -0.18
    return 0.02 + 0.14 * math.sin(0.08 * y + 0.15 * x)


def build_ground_mesh(
    name: str,
    length_m: float,
    surround_cfg: dict[str, Any],
    material: bpy.types.Material,
    rng: random.Random,
) -> bpy.types.Mesh:
    """Create surrounding terrain ground plane with drainage berm outside track corridor."""
    width = float(surround_cfg["ground_width_m"])
    y_min = -float(surround_cfg["ground_pre_buffer_m"])
    y_max = length_m + float(surround_cfg["ground_post_buffer_m"])

    # Lateral stations keep |X| <= 2.6m flat at Z=0.0 under the ballast bed
    x_stations = [
        -width * 0.5,
        -80.0,
        -28.0,
        -10.0,
        -4.5,
        -2.6,
        0.0,
        2.6,
        4.5,
        10.0,
        28.0,
        80.0,
        width * 0.5,
    ]
    y_steps = 60
    dy = (y_max - y_min) / y_steps

    bm = bmesh.new()
    rows: list[list[bmesh.types.BMVert]] = []
    for yi in range(y_steps + 1):
        y = y_min + yi * dy
        row = []
        for x in x_stations:
            base_z = ground_elevation_at(x, y)
            if abs(x) > 4.5 and abs(x) < width * 0.45:
                z = base_z + rng.uniform(-0.05, 0.05)
            else:
                z = base_z
            row.append(bm.verts.new((x, y, z)))
        rows.append(row)

    for yi in range(y_steps):
        for xi in range(len(x_stations) - 1):
            bm.faces.new((rows[yi][xi], rows[yi][xi + 1], rows[yi + 1][xi + 1], rows[yi + 1][xi]))

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    mesh.materials.append(material)
    return mesh


def build_marker_post_mesh(
    name: str,
    post_mat: bpy.types.Material,
    cap_mat: bpy.types.Material,
) -> bpy.types.Mesh:
    """Create a lightweight railway distance marker post (material 0=white post, 1=yellow cap)."""
    bm = bmesh.new()
    res_stem = bmesh.ops.create_cube(bm, size=1.0)
    for v in res_stem["verts"]:
        v.co.x *= 0.10
        v.co.y *= 0.06
        v.co.z = (v.co.z + 0.5) * 1.10

    res_cap = bmesh.ops.create_cube(bm, size=1.0)
    for v in res_cap["verts"]:
        v.co.x *= 0.24
        v.co.y *= 0.08
        v.co.z = 1.10 + (v.co.z + 0.5) * 0.22
    for f in bm.faces:
        if all(v in res_cap["verts"] for v in f.verts):
            f.material_index = 1

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    mesh.materials.append(post_mat)
    mesh.materials.append(cap_mat)
    return mesh


def build_scrub_bush_mesh(
    name: str, material: bpy.types.Material
) -> bpy.types.Mesh:
    """Create a low-poly icosphere scrub cluster for peripheral trackside context."""
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=1, radius=0.45)
    for v in bm.verts:
        v.co.z = max(0.0, v.co.z * 0.75 + 0.28)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    mesh.materials.append(material)
    return mesh

