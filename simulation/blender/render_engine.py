"""Blender 5.3 render engine, lighting, and output configuration."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import bpy
from mathutils import Euler

from track_components import get_node_tree


def configure_world_and_lighting(
    render_cfg: dict[str, Any],
    variant_params: dict[str, Any] | None = None,
) -> None:
    """Configure deterministic daylight/weather world sky and aligned sun lamp."""
    scene = bpy.context.scene
    light_cfg = render_cfg["lighting"]
    vp = variant_params or {}

    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("ITMS_World")
        scene.world = world

    nt = get_node_tree(world)
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputWorld")
    out.location = (400, 0)
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.location = (180, 0)
    bg.inputs["Strength"].default_value = float(
        vp.get("sky_strength", light_cfg.get("sky_strength", 0.12))
    )

    sky = nt.nodes.new("ShaderNodeTexSky")
    sky.location = (-120, 0)
    sky.sky_type = str(light_cfg.get("sky_type", "MULTIPLE_SCATTERING"))
    sky.sun_disc = False  # Use explicit SUN lamp for deterministic shadow control
    elev_deg = float(vp.get("sun_elevation_deg", light_cfg.get("sun_elevation_deg", 32.0)))
    azim_deg = float(vp.get("sun_azimuth_deg", light_cfg.get("sun_azimuth_deg", 125.0)))
    elev_rad = math.radians(elev_deg)
    azim_rad = math.radians(azim_deg)
    sky.sun_elevation = elev_rad
    sky.sun_rotation = azim_rad
    sky.turbidity = float(vp.get("turbidity", 2.2))
    sky.ground_albedo = 0.28

    nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])

    # Dedicated Sun lamp aligned with sky angles
    sun_data = bpy.data.lights.new(name="Sun_Key_Data", type="SUN")
    sun_data.energy = float(vp.get("sun_energy", light_cfg.get("sun_energy", 4.5)))
    sun_data.angle = math.radians(1.2 if float(vp.get("turbidity", 2.2)) < 4.0 else 4.5)
    sun_data.color = (1.0, 0.97, 0.92)

    sun_obj = bpy.data.objects.new(name="Sun_Key", object_data=sun_data)
    # Rotate sun vector (0, 0, -1) downward along elevation and azimuth
    sun_obj.rotation_mode = "XYZ"
    sun_obj.rotation_euler = Euler((math.pi * 0.5 - elev_rad, 0.0, azim_rad), "XYZ")

    env_col = bpy.data.collections.get("Environment") or scene.collection
    env_col.objects.link(sun_obj)


def configure_render_settings(
    camera_cfg: dict[str, Any],
    render_cfg: dict[str, Any],
    mode: str,
    temp_dir: Path,
    motion_blur_override: bool | None = None,
    fps_override: int | None = None,
    exposure_override: float | None = None,
) -> dict[str, Any]:
    """Configure engine, resolution, FPS, color management, and temporary dir."""
    scene = bpy.context.scene
    temp_dir.mkdir(parents=True, exist_ok=True)
    bpy.context.preferences.filepaths.temporary_directory = str(temp_dir)

    engine = str(render_cfg.get("engine", "BLENDER_EEVEE"))
    try:
        scene.render.engine = engine
    except Exception:
        fallback = str(render_cfg.get("fallback_engine", "CYCLES"))
        scene.render.engine = fallback
        engine = fallback

    if engine == "BLENDER_EEVEE":
        eevee_cfg = render_cfg["eevee"]
        samples = int(
            eevee_cfg["preview_samples"] if mode == "preview" else eevee_cfg["production_samples"]
        )
        scene.eevee.taa_render_samples = samples
        scene.eevee.use_raytracing = bool(eevee_cfg.get("use_raytracing", False))
        scene.eevee.use_shadows = bool(eevee_cfg.get("use_shadows", True))
        scene.eevee.shadow_resolution_scale = float(eevee_cfg.get("shadow_resolution_scale", 1.0))
    elif engine == "CYCLES":
        cyc_cfg = render_cfg["cycles"]
        scene.cycles.samples = int(cyc_cfg.get("samples", 32))
        scene.cycles.device = str(cyc_cfg.get("device", "GPU"))
        scene.cycles.use_denoising = bool(cyc_cfg.get("use_denoising", True))

    res_cfg = camera_cfg["resolution"]["preview" if mode == "preview" else "production"]
    width = int(res_cfg["width"])
    height = int(res_cfg["height"])
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100

    timing_cfg = camera_cfg["timing"]
    fps = int(
        fps_override
        if fps_override is not None
        else (
            timing_cfg["preview_fps"]
            if mode == "preview"
            else timing_cfg["production_fps"]
        )
    )
    scene.render.fps = fps
    scene.render.fps_base = float(timing_cfg.get("fps_base", 1.0))

    # Motion blur
    mb_cfg = camera_cfg.get("motion_blur", {})
    mb_enabled = (
        bool(motion_blur_override)
        if motion_blur_override is not None
        else bool(mb_cfg.get("enabled", False))
    )
    scene.render.use_motion_blur = mb_enabled
    if scene.render.use_motion_blur:
        scene.render.motion_blur_shutter = float(mb_cfg.get("shutter_fraction", 0.5))
        scene.render.motion_blur_position = str(mb_cfg.get("position", "CENTER"))
        if hasattr(scene.eevee, "motion_blur_steps"):
            scene.eevee.motion_blur_steps = int(mb_cfg.get("eevee_steps", 1))

    # Color management
    cm_cfg = render_cfg.get("color_management", {})
    scene.display_settings.display_device = str(cm_cfg.get("display_device", "sRGB"))
    scene.view_settings.view_transform = str(cm_cfg.get("view_transform", "AgX"))
    scene.view_settings.look = str(cm_cfg.get("look", "None"))
    default_ev = float(camera_cfg.get("exposure", {}).get("exposure_ev", cm_cfg.get("exposure", 0.0)))
    scene.view_settings.exposure = float(exposure_override) if exposure_override is not None else default_ev

    return {
        "engine": engine,
        "mode": mode,
        "width": width,
        "height": height,
        "fps": fps,
        "motion_blur": scene.render.use_motion_blur,
    }


def render_still_keyframes(
    scene: bpy.types.Scene,
    cam_obj: bpy.types.Object,
    positions_m: list[float],
    output_dir: Path,
) -> list[str]:
    """Render deterministic PNG still frames at requested track positions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prev_y = cam_obj.location.y
    prev_anim = cam_obj.animation_data.action if cam_obj.animation_data else None

    # Detach action temporarily so manual cam_obj.location updates stick per still render
    if cam_obj.animation_data is not None:
        cam_obj.animation_data.action = None

    scene.render.image_settings.media_type = "IMAGE"
    scene.render.image_settings.file_format = "PNG"

    rendered_paths: list[str] = []
    for pos_m in positions_m:
        cam_obj.location.y = float(pos_m)
        out_path = output_dir / f"preview_{int(round(pos_m)):03d}m.png"
        scene.render.filepath = str(out_path)
        bpy.ops.render.render(write_still=True)
        rendered_paths.append(str(out_path))

    if cam_obj.animation_data is not None and prev_anim is not None:
        cam_obj.animation_data.action = prev_anim
    cam_obj.location.y = prev_y
    return rendered_paths


def render_preview_video(
    scene: bpy.types.Scene,
    render_cfg: dict[str, Any],
    output_video_path: Path,
    frame_start: int,
    frame_end: int,
) -> str:
    """Render an H.264 MPEG4 video from frame_start to frame_end."""
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    out_cfg = render_cfg["output"]

    scene.frame_start = int(frame_start)
    scene.frame_end = int(frame_end)
    scene.frame_step = 1

    scene.render.image_settings.media_type = str(out_cfg.get("media_type", "VIDEO"))
    scene.render.image_settings.file_format = str(out_cfg.get("file_format", "FFMPEG"))
    scene.render.ffmpeg.format = str(out_cfg.get("ffmpeg_format", "MPEG4"))
    scene.render.ffmpeg.codec = str(out_cfg.get("video_codec", "H264"))
    scene.render.ffmpeg.constant_rate_factor = str(out_cfg.get("constant_rate_factor", "MEDIUM"))
    scene.render.ffmpeg.ffmpeg_preset = str(out_cfg.get("ffmpeg_preset", "GOOD"))
    scene.render.ffmpeg.gopsize = int(out_cfg.get("gopsize", 12))
    scene.render.filepath = str(output_video_path)

    bpy.ops.render.render(animation=True)
    return str(output_video_path)

