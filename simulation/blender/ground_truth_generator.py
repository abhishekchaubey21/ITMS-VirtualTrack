"""Camera-space 3D projection, occlusion checks, and segmentation mask rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

from track_components import get_node_tree


def _sample_world_points(obj: bpy.types.Object, max_points: int = 800) -> list[Vector]:
    verts = obj.data.vertices
    n = len(verts)
    if n == 0:
        return []
    step = max(1, n // max_points)
    mat = obj.matrix_world
    return [mat @ verts[i].co for i in range(0, n, step)]


def _check_point_occlusion(
    scene: bpy.types.Scene,
    depsgraph: bpy.types.Depsgraph,
    cam_origin: Vector,
    target_pt: Vector,
    target_obj_names: set[str],
    tol_m: float = 0.08,
) -> bool:
    """Return True if ray from camera reaches target_pt without earlier occluder."""
    vec = target_pt - cam_origin
    dist = vec.length
    if dist < 1e-4:
        return True
    direction = vec / dist
    hit, hit_loc, _norm, _idx, hit_obj, _mat = scene.ray_cast(
        depsgraph, cam_origin, direction, distance=dist + 0.05
    )
    if not hit:
        return True
    if hit_obj is not None and hit_obj.name in target_obj_names:
        return True
    hit_dist = (hit_loc - cam_origin).length
    return hit_dist >= (dist - tol_m)


def project_anomaly_to_camera(
    scene: bpy.types.Scene,
    cam_obj: bpy.types.Object,
    anomaly_obj: bpy.types.Object,
    anomaly_meta: dict[str, Any],
    gt_rules: dict[str, Any],
    occluder_exempt_names: set[str] | None = None,
) -> dict[str, Any]:
    """Project 3D anomaly geometry into current camera frame with occlusion check."""
    render = scene.render
    res_x = int(render.resolution_x)
    res_y = int(render.resolution_y)
    clip_start = float(cam_obj.data.clip_start)

    pts = _sample_world_points(anomaly_obj, max_points=600)
    if not pts:
        return {
            "id": anomaly_meta["id"],
            "class": anomaly_meta["class"],
            "class_id": anomaly_meta["class_id"],
            "detection_method": anomaly_meta["detection_method"],
            "position_m": anomaly_meta["configured_position_m"],
            "visible": False,
            "reason": "empty_geometry",
        }

    in_front_pts: list[tuple[float, float, float, Vector]] = []
    for pt in pts:
        ndc = world_to_camera_view(scene, cam_obj, pt)
        if ndc.z > clip_start:
            px = float(ndc.x) * res_x
            py = (1.0 - float(ndc.y)) * res_y
            in_front_pts.append((px, py, float(ndc.z), pt))

    if not in_front_pts:
        return {
            "id": anomaly_meta["id"],
            "class": anomaly_meta["class"],
            "class_id": anomaly_meta["class_id"],
            "detection_method": anomaly_meta["detection_method"],
            "yolo_enabled": bool(anomaly_meta.get("yolo_enabled", False)),
            "position_m": anomaly_meta["configured_position_m"],
            "camera_distance_m": round(
                float(anomaly_meta["configured_position_m"]) - cam_obj.location.y, 4
            ),
            "in_frustum": False,
            "visible": False,
            "sufficiently_visible": False,
            "occluded": False,
            "reason": "behind_camera",
        }

    raw_xmin = min(p[0] for p in in_front_pts)
    raw_ymin = min(p[1] for p in in_front_pts)
    raw_xmax = max(p[0] for p in in_front_pts)
    raw_ymax = max(p[1] for p in in_front_pts)
    min_depth = min(p[2] for p in in_front_pts)
    max_dist = float(gt_rules.get("max_annotation_distance_m", 120.0))

    # Clamp to image bounds
    xmin = max(0.0, min(float(res_x), raw_xmin))
    ymin = max(0.0, min(float(res_y), raw_ymin))
    xmax = max(0.0, min(float(res_x), raw_xmax))
    ymax = max(0.0, min(float(res_y), raw_ymax))

    bbox_w = xmax - xmin
    bbox_h = ymax - ymin
    bbox_area = bbox_w * bbox_h
    in_frustum = (bbox_w > 0.0) and (bbox_h > 0.0)

    raw_area = max(1e-6, (raw_xmax - raw_xmin) * (raw_ymax - raw_ymin))
    in_frame_fraction = bbox_area / raw_area
    partially_clipped = (
        raw_xmin < 0.0 or raw_ymin < 0.0 or raw_xmax > res_x or raw_ymax > res_y
    )

    # Occlusion check using raycast on sampled points (only when inside frustum & range)
    visible_fraction = 1.0
    occluded = False
    if (
        in_frustum
        and min_depth <= max_dist
        and gt_rules.get("check_occlusion", True)
        and not anomaly_obj.hide_render
    ):
        depsgraph = bpy.context.evaluated_depsgraph_get()
        cam_origin = cam_obj.matrix_world.translation
        ray_step = max(1, len(in_front_pts) // 24)
        sample_pts = [p[3] for p in in_front_pts[::ray_step]]
        exempt = {anomaly_obj.name}
        if occluder_exempt_names:
            exempt.update(occluder_exempt_names)
        unoccluded = sum(
            1
            for pt in sample_pts
            if _check_point_occlusion(scene, depsgraph, cam_origin, pt, exempt)
        )
        visible_fraction = unoccluded / max(1, len(sample_pts))
        occluded = visible_fraction < 0.99

    min_w = float(gt_rules.get("min_bbox_width_px", 8))
    min_h = float(gt_rules.get("min_bbox_height_px", 8))
    min_area = float(gt_rules.get("min_bbox_area_px", 80))
    min_vis = float(gt_rules.get("min_visible_fraction", 0.15))

    sufficiently_visible = (
        in_frustum
        and min_depth <= max_dist
        and bbox_w >= min_w
        and bbox_h >= min_h
        and bbox_area >= min_area
        and visible_fraction >= min_vis
        and in_frame_fraction >= 0.05
    )

    yolo_bbox = None
    if sufficiently_visible and anomaly_meta.get("yolo_enabled", False):
        cx = ((xmin + xmax) * 0.5) / res_x
        cy = ((ymin + ymax) * 0.5) / res_y
        nw = bbox_w / res_x
        nh = bbox_h / res_y
        yolo_bbox = [round(cx, 6), round(cy, 6), round(nw, 6), round(nh, 6)]

    return {
        "id": anomaly_meta["id"],
        "class": anomaly_meta["class"],
        "class_id": anomaly_meta["class_id"],
        "detection_method": anomaly_meta["detection_method"],
        "yolo_enabled": bool(anomaly_meta.get("yolo_enabled", False)),
        "position_m": anomaly_meta["configured_position_m"],
        "camera_distance_m": round(float(anomaly_meta["configured_position_m"]) - cam_obj.location.y, 4),
        "min_depth_m": round(min_depth, 4),
        "in_frustum": bool(in_frustum),
        "visible": bool(sufficiently_visible),
        "sufficiently_visible": bool(sufficiently_visible),
        "occluded": bool(occluded),
        "partially_clipped": bool(partially_clipped),
        "visible_fraction": round(visible_fraction, 4),
        "bbox": [round(xmin, 2), round(ymin, 2), round(xmax, 2), round(ymax, 2)]
        if in_frustum
        else None,
        "yolo_bbox": yolo_bbox,
    }


def _create_unlit_material(name: str, rgba: tuple[float, float, float, float]) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    nt = get_node_tree(mat)
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = rgba
    em.inputs["Strength"].default_value = 1.0
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return mat


def render_segmentation_mask(
    scene: bpy.types.Scene,
    target_obj: bpy.types.Object,
    output_mask_path: Path,
) -> str:
    """Render an occlusion-aware binary segmentation mask (0/255) from active camera."""
    results = render_segmentation_mask_sequence(
        scene, target_obj, [(scene.frame_current, output_mask_path)]
    )
    return results[0]


def render_segmentation_mask_sequence(
    scene: bpy.types.Scene,
    target_obj: bpy.types.Object,
    frame_to_mask_path: list[tuple[int, Path]],
) -> list[str]:
    """Render occlusion-aware binary segmentation masks across multiple frames cleanly."""
    if not frame_to_mask_path:
        return []

    prev_frame = scene.frame_current
    prev_engine = scene.render.engine
    prev_samples = getattr(scene.eevee, "taa_render_samples", 8)
    prev_vt = scene.view_settings.view_transform
    prev_exp = scene.view_settings.exposure
    prev_mb = scene.render.use_motion_blur
    prev_filter = scene.render.filter_size
    prev_dither = scene.render.dither_intensity
    prev_media = scene.render.image_settings.media_type
    prev_fmt = scene.render.image_settings.file_format
    prev_color_mode = scene.render.image_settings.color_mode
    prev_filepath = scene.render.filepath

    mat_white = _create_unlit_material("_Mask_White_Tmp", (1.0, 1.0, 1.0, 1.0))
    mat_black = _create_unlit_material("_Mask_Black_Tmp", (0.0, 0.0, 0.0, 1.0))

    saved_obj_hide: list[tuple[bpy.types.Object, bool]] = []
    for obj in scene.objects:
        if obj.type == "MESH":
            saved_obj_hide.append((obj, obj.hide_render))

    saved_mesh_mats: list[tuple[bpy.types.Mesh, list[bpy.types.Material | None]]] = [
        (mesh, list(mesh.materials)) for mesh in bpy.data.meshes
    ]

    world_bg_node = None
    prev_world_strength = None
    prev_world_color = None
    world_links_backup: list[tuple[Any, Any]] = []
    if scene.world and scene.world.node_tree:
        nt = scene.world.node_tree
        bg = next((n for n in nt.nodes if n.type == "BACKGROUND"), None)
        if bg is not None:
            world_bg_node = bg
            prev_world_strength = bg.inputs["Strength"].default_value
            prev_world_color = tuple(bg.inputs["Color"].default_value)
            for link in list(nt.links):
                if link.to_node == bg and link.to_socket == bg.inputs["Color"]:
                    world_links_backup.append((link.from_socket, link.to_socket))
                    nt.links.remove(link)
            bg.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
            bg.inputs["Strength"].default_value = 0.0

    rendered_paths: list[str] = []
    try:
        target_mesh = target_obj.data
        for mesh, _orig_mats in saved_mesh_mats:
            mesh.materials.clear()
            mesh.materials.append(mat_white if mesh == target_mesh else mat_black)

        for obj, orig_hide in saved_obj_hide:
            if obj == target_obj:
                obj.hide_render = False
            elif orig_hide:
                obj.hide_render = True
            else:
                obj.hide_render = False

        scene.render.engine = "BLENDER_EEVEE"
        scene.eevee.taa_render_samples = 1
        scene.render.use_motion_blur = False
        scene.render.filter_size = 0.01
        scene.render.dither_intensity = 0.0
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.exposure = 0.0
        scene.render.image_settings.media_type = "IMAGE"
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "BW"

        for f_num, out_path in frame_to_mask_path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            scene.frame_set(int(f_num))
            scene.render.filepath = str(out_path)
            bpy.ops.render.render(write_still=True)
            rendered_paths.append(str(out_path))
    finally:
        for obj, orig_hide in saved_obj_hide:
            obj.hide_render = orig_hide
        for mesh, orig_mats in saved_mesh_mats:
            mesh.materials.clear()
            for m in orig_mats:
                mesh.materials.append(m)

        if world_bg_node is not None and scene.world and scene.world.node_tree:
            nt = scene.world.node_tree
            world_bg_node.inputs["Strength"].default_value = prev_world_strength
            world_bg_node.inputs["Color"].default_value = prev_world_color
            for from_sock, to_sock in world_links_backup:
                nt.links.new(from_sock, to_sock)

        scene.frame_set(prev_frame)
        scene.render.engine = prev_engine
        scene.eevee.taa_render_samples = prev_samples
        scene.view_settings.view_transform = prev_vt
        scene.view_settings.exposure = prev_exp
        scene.render.use_motion_blur = prev_mb
        scene.render.filter_size = prev_filter
        scene.render.dither_intensity = prev_dither
        scene.render.image_settings.media_type = prev_media
        scene.render.image_settings.file_format = prev_fmt
        scene.render.image_settings.color_mode = prev_color_mode
        scene.render.filepath = prev_filepath

        bpy.data.materials.remove(mat_white, do_unlink=True)
        bpy.data.materials.remove(mat_black, do_unlink=True)

    return rendered_paths
