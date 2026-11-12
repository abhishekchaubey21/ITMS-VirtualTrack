"""Scene orchestration for building, validating, saving, and rendering ITMS-VirtualTrack."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import bpy

from anomaly_builder import build_anomalies, validate_anomalies_in_scene
from ground_truth_generator import (
    project_anomaly_to_camera,
    render_segmentation_mask,
    render_segmentation_mask_sequence,
)
from railway_builder import (
    build_railway_scene,
    compute_track_datum,
    validate_railway_geometry,
)
from render_engine import (
    configure_render_settings,
    configure_world_and_lighting,
    render_preview_video,
    render_still_keyframes,
)
from speed_controller import (
    apply_speed_trajectory,
    compute_speed_schedule,
    evaluate_trajectory_and_visibility,
)
from train_camera import (
    configure_linear_camera_motion,
    create_train_camera,
    validate_camera,
)


def load_configs(project_root: Path) -> dict[str, dict[str, Any]]:
    cfg_dir = project_root / "config"
    configs = {}
    for name in ("simulation", "camera", "anomaly", "rendering"):
        path = cfg_dir / f"{name}_config.json"
        with path.open("r", encoding="utf-8") as handle:
            configs[name] = json.load(handle)
    return configs


def reset_empty_scene() -> None:
    """Reset Blender to a clean empty scene with unit metric scale."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0


def run_milestone2_pipeline(
    project_root: Path,
    mode: str = "preview",
    render_video: bool = True,
    render_keyframes: bool = True,
    save_blend: bool = True,
) -> dict[str, Any]:
    """Build 250 m railway, place camera, validate geometry, save .blend, and render preview."""
    t0 = time.perf_counter()
    configs = load_configs(project_root)
    sim_cfg = configs["simulation"]
    cam_cfg = configs["camera"]
    ren_cfg = configs["rendering"]

    reset_empty_scene()
    scene = bpy.context.scene

    build_info = build_railway_scene(sim_cfg)
    datum = build_info["datum"]

    preview_sim = ren_cfg["preview_simulation"]
    start_pos_m = float(preview_sim.get("start_position_m", 0.0))
    cam_obj = create_train_camera(cam_cfg, datum, start_pos_m=start_pos_m)

    configure_world_and_lighting(ren_cfg)

    temp_dir = project_root / sim_cfg["paths"].get("temp_dir", "results/_tmp")
    render_summary = configure_render_settings(cam_cfg, ren_cfg, mode=mode, temp_dir=temp_dir)

    fps = render_summary["fps"]
    duration_s = float(preview_sim.get("duration_s", 2.0))
    speed_kmh = float(preview_sim.get("speed_kmh", 60.0))
    total_frames = max(2, int(round(duration_s * fps)))
    frame_start = 1
    frame_end = frame_start + total_frames - 1

    motion_info = configure_linear_camera_motion(
        cam_obj=cam_obj,
        start_y_m=start_pos_m,
        speed_kmh=speed_kmh,
        fps=fps,
        frame_start=frame_start,
        frame_end=frame_end,
    )

    # Geometry & camera checks inside Blender
    geo_validation = validate_railway_geometry(sim_cfg)
    cam_validation = validate_camera(scene, cam_cfg, datum, probe_distance_m=15.0)

    out_cfg = ren_cfg["output"]
    blend_path: str | None = None
    if save_blend and out_cfg.get("save_blend_file", True):
        target_blend = project_root / out_cfg.get(
            "blend_filename", "results/scenes/railway_250m_base.blend"
        )
        target_blend.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(target_blend))
        blend_path = str(target_blend)

    keyframe_paths: list[str] = []
    if render_keyframes and preview_sim.get("render_keyframes", True):
        kf_dir = project_root / out_cfg.get(
            "preview_keyframe_dir", "results/videos/m2_preview_frames"
        )
        kf_positions = [float(p) for p in preview_sim.get("keyframe_positions_m", [0.0, 25.0, 100.0])]
        keyframe_paths = render_still_keyframes(scene, cam_obj, kf_positions, kf_dir)

    video_path: str | None = None
    if render_video:
        target_vid = project_root / out_cfg.get(
            "preview_video_filename", "results/videos/m2_railway_preview.mp4"
        )
        video_path = render_preview_video(
            scene=scene,
            render_cfg=ren_cfg,
            output_video_path=target_vid,
            frame_start=frame_start,
            frame_end=frame_end,
        )

    elapsed_s = round(time.perf_counter() - t0, 2)

    metadata = {
        "data_label": "SYNTHETIC",
        "milestone": "M2_BASIC_RAILWAY_SIMULATION",
        "blender_version": bpy.app.version_string,
        "elapsed_seconds": elapsed_s,
        "scene_build": build_info,
        "camera_configuration": {
            "mounting": cam_cfg["mounting"],
            "orientation_deg": cam_cfg["orientation_deg"],
            "optics": cam_cfg["optics"],
            "render_settings": render_summary,
            "preview_motion": motion_info,
        },
        "outputs": {
            "blend_file": blend_path,
            "preview_video": video_path,
            "preview_keyframes": keyframe_paths,
        },
    }

    validation = {
        "data_label": "SYNTHETIC",
        "milestone": "M2_BASIC_RAILWAY_SIMULATION",
        "geometry_validation": geo_validation,
        "camera_validation": cam_validation,
        "blender_all_passed": geo_validation["all_passed"] and cam_validation["all_passed"],
    }

    metrics_dir = project_root / sim_cfg["paths"].get("results_dir", "results") / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with (metrics_dir / "m2_scene_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    with (metrics_dir / "m2_blender_validation.json").open("w", encoding="utf-8") as handle:
        json.dump(validation, handle, indent=2)

    return {"metadata": metadata, "validation": validation}


def _render_single_camera_view(
    scene: bpy.types.Scene,
    cam_obj: bpy.types.Object,
    cam_y_m: float,
    output_png: Path,
) -> str:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    cam_obj.location.y = float(cam_y_m)
    bpy.context.view_layer.update()
    scene.render.image_settings.media_type = "IMAGE"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.filepath = str(output_png)
    bpy.ops.render.render(write_still=True)
    return str(output_png)


def _build_and_inspect_variant(
    project_root: Path,
    configs: dict[str, dict[str, Any]],
    variant_name: str,
    active_ids: list[str],
    blend_rel_path: str,
    inspection_views: list[tuple[str, float, str]],
    mode: str = "preview",
) -> dict[str, Any]:
    """Build one deterministic scene variant, save .blend, render views/masks, and validate."""
    sim_cfg = configs["simulation"]
    cam_cfg = configs["camera"]
    anom_cfg = configs["anomaly"]
    ren_cfg = configs["rendering"]

    reset_empty_scene()
    scene = bpy.context.scene

    crack_cfg = anom_cfg["anomalies"]["A3"] if "A3" in active_ids else None
    mis_cfg = anom_cfg["anomalies"]["A4"] if "A4" in active_ids else None

    build_info = build_railway_scene(
        sim_cfg, crack_cfg=crack_cfg, misalignment_cfg=mis_cfg
    )
    datum = build_info["datum"]
    anomaly_meta = build_anomalies(
        active_ids=active_ids,
        anomaly_cfg=anom_cfg,
        sim_cfg=sim_cfg,
        track_datum=datum,
    )

    cam_obj = create_train_camera(cam_cfg, datum, start_pos_m=0.0)
    configure_world_and_lighting(ren_cfg)

    temp_dir = project_root / sim_cfg["paths"].get("temp_dir", "results/_tmp")
    render_summary = configure_render_settings(cam_cfg, ren_cfg, mode=mode, temp_dir=temp_dir)

    geo_val = validate_railway_geometry(sim_cfg, misalignment_cfg=mis_cfg)
    cam_val = validate_camera(scene, cam_cfg, datum, probe_distance_m=15.0)
    anom_val = validate_anomalies_in_scene(active_ids, anomaly_meta, anom_cfg, datum)

    # Save .blend before rendering view-specific inspection stills (camera stays at 0.0m default)
    blend_path = project_root / blend_rel_path
    blend_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))

    insp_dir = project_root / "results" / "videos" / "m3_inspection"
    gt_rules = anom_cfg["ground_truth_rules"]
    rendered_views: list[dict[str, Any]] = []
    projection_checks_ok = True

    for target_id, cam_y_m, img_filename in inspection_views:
        img_path = _render_single_camera_view(
            scene, cam_obj, cam_y_m, insp_dir / img_filename
        )
        view_entry: dict[str, Any] = {
            "target_id": target_id,
            "camera_y_m": cam_y_m,
            "image_path": img_path,
            "projections": [],
            "mask_path": None,
        }

        for aid in active_ids:
            obj_name = anomaly_meta[aid]["object_name"]
            a_obj = bpy.data.objects.get(obj_name)
            if a_obj is None:
                continue
            exempt = {"Rail_Right", "Rail_Left"} if aid in ("A3", "A4") else None
            proj = project_anomaly_to_camera(
                scene, cam_obj, a_obj, anomaly_meta[aid], gt_rules, occluder_exempt_names=exempt
            )
            view_entry["projections"].append(proj)
            if aid == target_id and not proj["visible"]:
                projection_checks_ok = False

        if target_id in ("A3", "A4") and target_id in active_ids:
            mask_obj = bpy.data.objects.get(anomaly_meta[target_id]["object_name"])
            if mask_obj is not None:
                mask_name = Path(img_filename).stem + "_mask.png"
                view_entry["mask_path"] = render_segmentation_mask(
                    scene, mask_obj, insp_dir / mask_name
                )

        rendered_views.append(view_entry)

    return {
        "variant": variant_name,
        "active_anomalies": active_ids,
        "blend_file": str(blend_path),
        "render_settings": render_summary,
        "datum": datum,
        "anomalies_metadata": anomaly_meta,
        "inspection_views": rendered_views,
        "validation": {
            "geometry_validation": geo_val,
            "camera_validation": cam_val,
            "anomaly_validation": anom_val,
            "projection_visible_ok": projection_checks_ok,
            "all_passed": bool(
                geo_val["all_passed"]
                and cam_val["all_passed"]
                and anom_val["all_passed"]
                and projection_checks_ok
            ),
        },
    }


def run_milestone3_pipeline(
    project_root: Path,
    mode: str = "preview",
) -> dict[str, Any]:
    """Generate clean baseline, 4 individual anomaly scenes, and combined 4-anomaly scene."""
    t0 = time.perf_counter()
    configs = load_configs(project_root)
    sim_cfg = configs["simulation"]
    anom_cfg = configs["anomaly"]["anomalies"]

    # Inspection camera distances tailored so every anomaly fills clear central framing
    pos_a1 = float(anom_cfg["A1"]["position_m"])
    pos_a2 = float(anom_cfg["A2"]["position_m"])
    pos_a3 = float(anom_cfg["A3"]["position_m"])
    pos_a4 = float(anom_cfg["A4"]["position_m"])

    cam_y_a1 = max(0.0, pos_a1 - 7.5)
    cam_y_a2 = max(0.0, pos_a2 - 8.5)
    cam_y_a3 = max(0.0, pos_a3 - 6.2)
    cam_y_a4 = max(0.0, pos_a4 - 14.0)

    specs = [
        (
            "baseline",
            [],
            "results/scenes/railway_250m_baseline.blend",
            [
                ("NONE", cam_y_a1, "baseline_50m.png"),
                ("NONE", cam_y_a4, "baseline_200m.png"),
            ],
        ),
        (
            "A1_only",
            ["A1"],
            "results/scenes/railway_250m_A1_only.blend",
            [("A1", cam_y_a1, "A1_only_50m.png")],
        ),
        (
            "A2_only",
            ["A2"],
            "results/scenes/railway_250m_A2_only.blend",
            [("A2", cam_y_a2, "A2_only_100m.png")],
        ),
        (
            "A3_only",
            ["A3"],
            "results/scenes/railway_250m_A3_only.blend",
            [("A3", cam_y_a3, "A3_only_150m.png")],
        ),
        (
            "A4_only",
            ["A4"],
            "results/scenes/railway_250m_A4_only.blend",
            [("A4", cam_y_a4, "A4_only_200m.png")],
        ),
        (
            "combined_all",
            ["A1", "A2", "A3", "A4"],
            "results/scenes/railway_250m_anomalies.blend",
            [
                ("A1", cam_y_a1, "combined_A1_50m.png"),
                ("A2", cam_y_a2, "combined_A2_100m.png"),
                ("A3", cam_y_a3, "combined_A3_150m.png"),
                ("A4", cam_y_a4, "combined_A4_200m.png"),
            ],
        ),
    ]

    variant_results: dict[str, Any] = {}
    all_variants_passed = True
    for variant_name, active_ids, blend_rel, views in specs:
        res = _build_and_inspect_variant(
            project_root=project_root,
            configs=configs,
            variant_name=variant_name,
            active_ids=active_ids,
            blend_rel_path=blend_rel,
            inspection_views=views,
            mode=mode,
        )
        variant_results[variant_name] = res
        if not res["validation"]["all_passed"]:
            all_variants_passed = False

    elapsed_s = round(time.perf_counter() - t0, 2)
    metrics_dir = project_root / sim_cfg["paths"].get("results_dir", "results") / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Save snapshot of exact scene & anomaly configuration used
    config_snapshot = {
        "data_label": "SYNTHETIC",
        "milestone": "M3_ANOMALY_GENERATION",
        "blender_version": bpy.app.version_string,
        "configs": configs,
    }
    with (metrics_dir / "m3_scene_config_snapshot.json").open("w", encoding="utf-8") as handle:
        json.dump(config_snapshot, handle, indent=2)

    anomaly_metadata_payload = {
        "data_label": "SYNTHETIC",
        "milestone": "M3_ANOMALY_GENERATION",
        "blender_version": bpy.app.version_string,
        "elapsed_seconds": elapsed_s,
        "class_taxonomy": configs["anomaly"]["classes"],
        "combined_scene": variant_results["combined_all"]["blend_file"],
        "baseline_scene": variant_results["baseline"]["blend_file"],
        "individual_scenes": {
            k: variant_results[k]["blend_file"]
            for k in ("A1_only", "A2_only", "A3_only", "A4_only")
        },
        "anomalies": variant_results["combined_all"]["anomalies_metadata"],
        "variants": {
            name: {
                "active_anomalies": res["active_anomalies"],
                "blend_file": res["blend_file"],
                "inspection_views": res["inspection_views"],
            }
            for name, res in variant_results.items()
        },
    }
    with (metrics_dir / "m3_anomaly_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(anomaly_metadata_payload, handle, indent=2)

    blender_validation_payload = {
        "data_label": "SYNTHETIC",
        "milestone": "M3_ANOMALY_GENERATION",
        "blender_all_passed": all_variants_passed,
        "variants": {
            name: res["validation"] for name, res in variant_results.items()
        },
    }
    with (metrics_dir / "m3_blender_validation.json").open("w", encoding="utf-8") as handle:
        json.dump(blender_validation_payload, handle, indent=2)

    return {
        "metadata": anomaly_metadata_payload,
        "validation": blender_validation_payload,
    }


def run_milestone4_speed_run(
    project_root: Path,
    speed_kmh: float,
    mode: str = "preview",
    motion_blur: bool | None = None,
    skip_render: bool = False,
) -> dict[str, Any]:
    """Run a single speed simulation on the preserved M3 combined anomaly scene."""
    t0 = time.perf_counter()
    configs = load_configs(project_root)
    sim_cfg = configs["simulation"]
    cam_cfg = configs["camera"]
    anom_cfg = configs["anomaly"]
    ren_cfg = configs["rendering"]

    metrics_dir = project_root / sim_cfg["paths"].get("results_dir", "results") / "metrics"
    m3_meta_path = metrics_dir / "m3_anomaly_metadata.json"
    if not m3_meta_path.is_file():
        raise FileNotFoundError(
            f"Missing {m3_meta_path}. Run Milestone 3 before Milestone 4."
        )
    with m3_meta_path.open("r", encoding="utf-8") as handle:
        m3_meta = json.load(handle)

    combined_blend = Path(m3_meta["combined_scene"])
    if not combined_blend.is_file():
        combined_blend = project_root / "results" / "scenes" / "railway_250m_anomalies.blend"
    if not combined_blend.is_file():
        raise FileNotFoundError(f"Combined anomaly scene not found: {combined_blend}")

    # Open preserved M3 combined scene read-only (never overwrite it)
    bpy.ops.wm.open_mainfile(filepath=str(combined_blend))
    scene = bpy.context.scene
    cam_obj = scene.camera or bpy.data.objects.get("Train_Camera")
    if cam_obj is None:
        raise RuntimeError("Train_Camera not found in loaded combined anomaly scene.")
    scene.camera = cam_obj

    temp_dir = project_root / sim_cfg["paths"].get("temp_dir", "results/_tmp")
    render_summary = configure_render_settings(
        camera_cfg=cam_cfg,
        render_cfg=ren_cfg,
        mode=mode,
        temp_dir=temp_dir,
        motion_blur_override=motion_blur,
    )
    fps = int(render_summary["fps"])

    datum = compute_track_datum(sim_cfg)
    mis_cfg = anom_cfg["anomalies"]["A4"]
    geo_val = validate_railway_geometry(sim_cfg, misalignment_cfg=mis_cfg)
    cam_val = validate_camera(scene, cam_cfg, datum, probe_distance_m=15.0)

    anomalies_meta = m3_meta["anomalies"]
    anom_val = validate_anomalies_in_scene(
        ["A1", "A2", "A3", "A4"], anomalies_meta, anom_cfg, datum
    )

    anomaly_objects: dict[str, tuple[bpy.types.Object, dict[str, Any]]] = {}
    for aid, meta in anomalies_meta.items():
        obj = bpy.data.objects.get(meta["object_name"])
        if obj is not None:
            anomaly_objects[aid] = (obj, meta)

    start_m = float(sim_cfg["simulation"]["start_position_m"])
    end_m = float(sim_cfg["simulation"]["end_position_m"])
    schedule = compute_speed_schedule(speed_kmh, start_m, end_m, fps)
    apply_speed_trajectory(scene, cam_obj, schedule)

    eval_res = evaluate_trajectory_and_visibility(
        scene=scene,
        cam_obj=cam_obj,
        schedule=schedule,
        anomaly_objects=anomaly_objects,
        gt_rules=anom_cfg["ground_truth_rules"],
    )

    speed_tag = int(schedule["speed_kmh"])
    out_dir = project_root / "results" / "videos" / "m4_speed_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_preview" if mode == "preview" else ""
    video_filename = f"railway_{speed_tag}kmh{suffix}.mp4"
    video_path = out_dir / video_filename

    render_time_s = 0.0
    rendered_path: str | None = None
    video_size_bytes = 0
    if not skip_render:
        t_ren0 = time.perf_counter()
        rendered_path = render_preview_video(
            scene=scene,
            render_cfg=ren_cfg,
            output_video_path=video_path,
            frame_start=schedule["frame_start"],
            frame_end=schedule["frame_end"],
        )
        render_time_s = round(time.perf_counter() - t_ren0, 3)
        if video_path.is_file():
            video_size_bytes = int(video_path.stat().st_size)

    total_elapsed_s = round(time.perf_counter() - t0, 3)

    anomaly_world_positions = {
        aid: {
            "class": meta["class"],
            "detection_method": meta["detection_method"],
            "position_m": meta["configured_position_m"],
            "world_center_m": meta["world_center_m"],
            "world_dimensions_m": meta["world_dimensions_m"],
        }
        for aid, meta in anomalies_meta.items()
    }

    run_metadata = {
        "synthetic": True,
        "data_label": "SYNTHETIC",
        "milestone": "M4_SPEED_SIMULATION",
        "source_scene_blend": str(combined_blend),
        "video_filename": video_filename,
        "video_path": rendered_path,
        "video_size_bytes": video_size_bytes,
        "speed_kmh": schedule["speed_kmh"],
        "speed_mps": schedule["speed_mps"],
        "track_length_m": schedule["track_length_m"],
        "expected_duration_s": schedule["expected_duration_s"],
        "actual_duration_s": schedule["actual_travel_duration_s"],
        "video_container_duration_s": schedule["video_container_duration_s"],
        "fps": schedule["fps"],
        "frame_interval_s": schedule["frame_interval_s"],
        "distance_per_frame_m": schedule["distance_per_frame_m"],
        "num_intervals": schedule["num_intervals"],
        "frame_count": schedule["frame_count"],
        "camera_start_position_m": eval_res["trajectory_validation"]["first_frame_y_m"],
        "camera_end_position_m": eval_res["trajectory_validation"]["last_frame_y_m"],
        "camera_configuration": {
            "height_above_rail_head_m": cam_cfg["mounting"]["height_above_rail_head_m"],
            "focal_length_mm": cam_cfg["optics"]["focal_length_mm"],
            "pitch_deg": cam_cfg["orientation_deg"]["pitch_deg"],
            "horizontal_fov_deg": cam_val["checks"]["camera_optics"]["horizontal_fov_deg"],
            "vertical_fov_deg": cam_val["checks"]["camera_optics"]["vertical_fov_deg"],
        },
        "anomaly_world_positions": anomaly_world_positions,
        "blender_render_configuration": {
            "blender_version": bpy.app.version_string,
            **render_summary,
            "samples": (
                ren_cfg["eevee"]["preview_samples"]
                if mode == "preview"
                else ren_cfg["eevee"]["production_samples"]
            ),
            "render_time_seconds": render_time_s,
            "total_elapsed_seconds": total_elapsed_s,
            "ms_per_frame": round(
                (render_time_s * 1000.0) / max(1, schedule["frame_count"]), 2
            )
            if not skip_render
            else 0.0,
        },
        "validation": {
            "geometry_passed": geo_val["all_passed"],
            "camera_passed": cam_val["all_passed"],
            "anomalies_passed": anom_val["all_passed"],
            "trajectory_validation": eval_res["trajectory_validation"],
            "all_passed": bool(
                geo_val["all_passed"]
                and cam_val["all_passed"]
                and anom_val["all_passed"]
                and eval_res["trajectory_validation"]["passed"]
            ),
        },
        "visibility_summary": eval_res["visibility_summary"],
        "frames": eval_res["frames"],
    }

    run_meta_dir = metrics_dir / "m4_speed_runs"
    run_meta_dir.mkdir(parents=True, exist_ok=True)
    meta_out_path = run_meta_dir / f"railway_{speed_tag}kmh{suffix}_metadata.json"
    with meta_out_path.open("w", encoding="utf-8") as handle:
        json.dump(run_metadata, handle, indent=2)

    return run_metadata


def run_milestone5_masks(
    project_root: Path,
    speed_kmh: float,
    mode: str,
    seg_out_dir: Path,
    stem_prefix: str,
) -> dict[str, Any]:
    """Render aligned A3/A4 binary segmentation masks for an existing M4 speed run."""
    configs = load_configs(project_root)
    sim_cfg = configs["simulation"]
    cam_cfg = configs["camera"]
    ren_cfg = configs["rendering"]

    speed_tag = int(speed_kmh) if float(speed_kmh).is_integer() else speed_kmh
    suffix = "_preview" if mode == "preview" else ""
    m4_meta_path = (
        project_root
        / "results"
        / "metrics"
        / "m4_speed_runs"
        / f"railway_{speed_tag}kmh{suffix}_metadata.json"
    )
    if not m4_meta_path.is_file():
        raise FileNotFoundError(f"Missing M4 metadata: {m4_meta_path}")

    with m4_meta_path.open("r", encoding="utf-8") as handle:
        m4_meta = json.load(handle)

    combined_blend = Path(m4_meta["source_scene_blend"])
    if not combined_blend.is_file():
        combined_blend = project_root / "results" / "scenes" / "railway_250m_anomalies.blend"

    bpy.ops.wm.open_mainfile(filepath=str(combined_blend))
    scene = bpy.context.scene
    cam_obj = scene.camera or bpy.data.objects.get("Train_Camera")
    if cam_obj is None:
        raise RuntimeError("Train_Camera not found in loaded scene.")
    scene.camera = cam_obj

    temp_dir = project_root / sim_cfg["paths"].get("temp_dir", "results/_tmp")
    configure_render_settings(cam_cfg, ren_cfg, mode=mode, temp_dir=temp_dir)

    schedule = compute_speed_schedule(
        speed_kmh=speed_kmh,
        start_m=float(sim_cfg["simulation"]["start_position_m"]),
        end_m=float(sim_cfg["simulation"]["end_position_m"]),
        fps=int(m4_meta["fps"]),
    )
    apply_speed_trajectory(scene, cam_obj, schedule)

    a3_tasks: list[tuple[int, Path]] = []
    a4_tasks: list[tuple[int, Path]] = []
    for f_entry in m4_meta["frames"]:
        f_num = int(f_entry["frame"])
        for anom in f_entry["anomalies"]:
            if anom["id"] == "A3" and anom.get("sufficiently_visible"):
                a3_tasks.append(
                    (f_num, seg_out_dir / "a3_masks" / f"{stem_prefix}_f{f_num:04d}_a3_mask.png")
                )
            elif anom["id"] == "A4" and anom.get("sufficiently_visible"):
                a4_tasks.append(
                    (f_num, seg_out_dir / "a4_masks" / f"{stem_prefix}_f{f_num:04d}_a4_mask.png")
                )

    a3_obj = bpy.data.objects.get("A3_Rail_Crack")
    a4_obj = bpy.data.objects.get("A4_Rail_Misalignment_Mask")

    a3_paths = (
        render_segmentation_mask_sequence(scene, a3_obj, a3_tasks)
        if a3_obj is not None
        else []
    )
    a4_paths = (
        render_segmentation_mask_sequence(scene, a4_obj, a4_tasks)
        if a4_obj is not None
        else []
    )

    return {
        "speed_kmh": speed_tag,
        "a3_mask_count": len(a3_paths),
        "a4_mask_count": len(a4_paths),
    }


def _scale_anomaly_cfgs_for_variant(
    anom_cfg: dict[str, Any], variant_params: dict[str, Any]
) -> dict[str, Any]:
    """Return deep-copied anomaly config scaled deterministically for a variant."""
    scaled = json.loads(json.dumps(anom_cfg))
    a3_scale = float(variant_params.get("a3_scale_factor", 1.0))
    a4_scale = float(variant_params.get("a4_deviation_scale", 1.0))
    scaled["anomalies"]["A3"]["crack_length_m"] = round(
        float(scaled["anomalies"]["A3"]["crack_length_m"]) * a3_scale, 4
    )
    scaled["anomalies"]["A3"]["crack_width_m"] = round(
        float(scaled["anomalies"]["A3"]["crack_width_m"]) * a3_scale, 4
    )
    scaled["anomalies"]["A4"]["max_lateral_deviation_m"] = round(
        float(scaled["anomalies"]["A4"]["max_lateral_deviation_m"]) * a4_scale, 5
    )
    return scaled


def run_milestone5_variant(
    project_root: Path,
    variant_cfg_path: Path,
    mode: str = "preview",
) -> dict[str, Any]:
    """Generate one independent randomized scene variant, video, masks, and GT JSON."""
    t0 = time.perf_counter()
    with variant_cfg_path.open("r", encoding="utf-8") as handle:
        v_cfg = json.load(handle)

    configs = load_configs(project_root)
    sim_cfg = configs["simulation"]
    cam_cfg = configs["camera"]
    ren_cfg = configs["rendering"]
    base_anom_cfg = configs["anomaly"]

    vp = dict(v_cfg.get("variant_params", {}))
    # Prevent double-scaling inside build_anomalies since we scale configs upfront
    anom_cfg = _scale_anomaly_cfgs_for_variant(base_anom_cfg, vp)
    vp_for_builder = dict(vp)
    vp_for_builder["a3_scale_factor"] = 1.0
    vp_for_builder["a4_deviation_scale"] = 1.0

    active_ids = list(v_cfg.get("active_anomalies", ["A1", "A2", "A3", "A4"]))
    seed_offset = int(v_cfg["seed_offset"])
    speed_kmh = float(v_cfg["speed_kmh"])
    variant_id = str(v_cfg["variant_id"])
    stem_prefix = str(v_cfg["stem_prefix"])

    reset_empty_scene()
    scene = bpy.context.scene

    crack_cfg = anom_cfg["anomalies"]["A3"] if "A3" in active_ids else None
    mis_cfg = anom_cfg["anomalies"]["A4"] if "A4" in active_ids else None

    build_info = build_railway_scene(
        sim_cfg,
        crack_cfg=crack_cfg,
        misalignment_cfg=mis_cfg,
        variant_params=vp_for_builder,
        seed_offset=seed_offset,
    )
    datum = build_info["datum"]
    anomalies_meta = build_anomalies(
        active_ids=active_ids,
        anomaly_cfg=anom_cfg,
        sim_cfg=sim_cfg,
        track_datum=datum,
        variant_params=vp_for_builder,
        seed_offset=seed_offset,
    )

    cam_obj = create_train_camera(cam_cfg, datum, start_pos_m=0.0)
    configure_world_and_lighting(ren_cfg, variant_params=vp)

    temp_dir = project_root / sim_cfg["paths"].get("temp_dir", "results/_tmp")
    render_summary = configure_render_settings(
        camera_cfg=cam_cfg,
        render_cfg=ren_cfg,
        mode=mode,
        temp_dir=temp_dir,
        exposure_override=vp.get("exposure_ev"),
    )
    fps = int(render_summary["fps"])

    geo_val = validate_railway_geometry(sim_cfg, misalignment_cfg=mis_cfg)
    cam_val = validate_camera(scene, cam_cfg, datum, probe_distance_m=15.0)
    anom_val = validate_anomalies_in_scene(active_ids, anomalies_meta, anom_cfg, datum)

    anomaly_objects: dict[str, tuple[bpy.types.Object, dict[str, Any]]] = {}
    for aid, meta in anomalies_meta.items():
        obj = bpy.data.objects.get(meta["object_name"])
        if obj is not None:
            anomaly_objects[aid] = (obj, meta)

    start_m = float(sim_cfg["simulation"]["start_position_m"])
    end_m = float(sim_cfg["simulation"]["end_position_m"])
    schedule = compute_speed_schedule(speed_kmh, start_m, end_m, fps)
    apply_speed_trajectory(scene, cam_obj, schedule)

    eval_res = evaluate_trajectory_and_visibility(
        scene=scene,
        cam_obj=cam_obj,
        schedule=schedule,
        anomaly_objects=anomaly_objects,
        gt_rules=anom_cfg["ground_truth_rules"],
    )

    video_out_path = Path(v_cfg["video_out_path"])
    t_ren0 = time.perf_counter()
    rendered_video = render_preview_video(
        scene=scene,
        render_cfg=ren_cfg,
        output_video_path=video_out_path,
        frame_start=schedule["frame_start"],
        frame_end=schedule["frame_end"],
    )
    render_time_s = round(time.perf_counter() - t_ren0, 3)

    seg_dir = Path(v_cfg["seg_out_dir"])
    a3_tasks: list[tuple[int, Path]] = []
    a4_tasks: list[tuple[int, Path]] = []
    for f_entry in eval_res["frames"]:
        f_num = int(f_entry["frame"])
        for anom in f_entry["anomalies"]:
            if anom["id"] == "A3" and anom.get("sufficiently_visible"):
                a3_tasks.append(
                    (f_num, seg_dir / "a3_masks" / f"{stem_prefix}_f{f_num:04d}_a3_mask.png")
                )
            elif anom["id"] == "A4" and anom.get("sufficiently_visible"):
                a4_tasks.append(
                    (f_num, seg_dir / "a4_masks" / f"{stem_prefix}_f{f_num:04d}_a4_mask.png")
                )

    a3_obj = bpy.data.objects.get("A3_Rail_Crack")
    a4_obj = bpy.data.objects.get("A4_Rail_Misalignment_Mask")
    a3_masks = (
        render_segmentation_mask_sequence(scene, a3_obj, a3_tasks)
        if a3_obj is not None
        else []
    )
    a4_masks = (
        render_segmentation_mask_sequence(scene, a4_obj, a4_tasks)
        if a4_obj is not None
        else []
    )

    total_elapsed_s = round(time.perf_counter() - t0, 3)
    payload = {
        "synthetic": True,
        "data_label": "SYNTHETIC",
        "milestone": "M5_DATASET_GENERATION",
        "variant_id": variant_id,
        "split": str(v_cfg["split"]),
        "dataset_group": str(v_cfg["dataset_group"]),
        "stem_prefix": stem_prefix,
        "random_seed": int(sim_cfg["project"]["random_seed"]) + seed_offset,
        "seed_offset": seed_offset,
        "variant_params": vp,
        "video_filename": video_out_path.name,
        "video_path": rendered_video,
        "video_size_bytes": int(video_out_path.stat().st_size) if video_out_path.is_file() else 0,
        "speed_kmh": schedule["speed_kmh"],
        "speed_mps": schedule["speed_mps"],
        "track_length_m": schedule["track_length_m"],
        "expected_duration_s": schedule["expected_duration_s"],
        "actual_duration_s": schedule["actual_travel_duration_s"],
        "fps": schedule["fps"],
        "frame_count": schedule["frame_count"],
        "camera_start_position_m": eval_res["trajectory_validation"]["first_frame_y_m"],
        "camera_end_position_m": eval_res["trajectory_validation"]["last_frame_y_m"],
        "blender_render_configuration": {
            "blender_version": bpy.app.version_string,
            **render_summary,
            "render_time_seconds": render_time_s,
            "total_elapsed_seconds": total_elapsed_s,
        },
        "anomalies_metadata": anomalies_meta,
        "visibility_summary": eval_res["visibility_summary"],
        "masks_generated": {
            "a3_mask_count": len(a3_masks),
            "a4_mask_count": len(a4_masks),
        },
        "validation": {
            "geometry_passed": geo_val["all_passed"],
            "camera_passed": cam_val["all_passed"],
            "anomalies_passed": anom_val["all_passed"],
            "trajectory_validation": eval_res["trajectory_validation"],
            "all_passed": bool(
                geo_val["all_passed"]
                and cam_val["all_passed"]
                and anom_val["all_passed"]
                and eval_res["trajectory_validation"]["passed"]
            ),
        },
        "frames": eval_res["frames"],
    }

    meta_out_path = Path(v_cfg["meta_out_path"])
    meta_out_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return payload

