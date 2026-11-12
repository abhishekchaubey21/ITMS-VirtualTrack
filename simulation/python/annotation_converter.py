"""Ground-truth annotation converter for frame JSON, YOLO labels, A3/A4 masks & profiles."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _sample_a4_displacement_profile(
    a4_cfg: dict[str, Any], datum: dict[str, float]
) -> list[dict[str, float]]:
    """Compute exact 0.25 m sampled lateral displacement profile for A4."""
    start_m = float(a4_cfg["start_m"])
    end_m = float(a4_cfg["end_m"])
    span = end_m - start_m
    max_dev = float(a4_cfg["max_lateral_deviation_m"])
    affected_rails = str(a4_cfg.get("affected_rails", "both"))
    left_cx = float(datum["left_rail_center_x"])
    right_cx = float(datum["right_rail_center_x"])

    samples: list[dict[str, float]] = []
    step = 0.25
    n_samples = int(round(span / step)) + 1
    for i in range(n_samples):
        y = start_m + i * step
        u = (y - start_m) / span if span > 0 else 0.0
        dx = max_dev * (math.sin(math.pi * u) ** 2)
        slope = max_dev * (math.pi / span) * math.sin(2.0 * math.pi * u) if span > 0 else 0.0
        samples.append(
            {
                "y_m": round(y, 3),
                "lateral_shift_m": round(dx, 6),
                "left_rail_center_x_m": round(
                    left_cx + (dx if affected_rails in ("left", "both") else 0.0), 6
                ),
                "right_rail_center_x_m": round(
                    right_cx + (dx if affected_rails in ("right", "both") else 0.0), 6
                ),
                "tangent_angle_deg": round(math.degrees(math.atan(slope)), 4),
            }
        )
    return samples


def _mask_stats(mask_path: Path) -> tuple[int, list[float] | None]:
    """Ensure strict binary {0, 255} mask on disk and return foreground area + tight bbox."""
    if not mask_path.is_file():
        return 0, None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return 0, None
    uniq = set(int(v) for v in np.unique(mask))
    if not uniq.issubset({0, 255}):
        mask = np.where(mask >= 128, 255, 0).astype(np.uint8)
        cv2.imwrite(str(mask_path), mask, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    ys, xs = np.where(mask > 127)
    fg_count = int(len(xs))
    if fg_count == 0:
        return 0, None
    return fg_count, [
        float(np.min(xs)),
        float(np.min(ys)),
        float(np.max(xs) + 1),
        float(np.max(ys) + 1),
    ]


def convert_video_annotations(
    video_id: str,
    source_video_path: Path,
    source_scene: str,
    split: str,
    dataset_group: str,
    stem_prefix: str,
    raw_meta: dict[str, Any],
    extracted_frames: list[dict[str, Any]],
    anomalies_world_meta: dict[str, Any],
    a4_cfg: dict[str, Any],
    track_datum: dict[str, float],
    dataset_root: Path,
) -> dict[str, Any]:
    """Generate frame-level GT JSON, YOLO labels (A1/A2), and A3/A4 metadata records."""
    labels_dir = dataset_root / "labels" / split
    seg_dir = dataset_root / "segmentation" / split
    gt_root = dataset_root / "ground_truth"
    labels_dir.mkdir(parents=True, exist_ok=True)
    (gt_root / "frame_annotations").mkdir(parents=True, exist_ok=True)
    (gt_root / "anomaly_metadata").mkdir(parents=True, exist_ok=True)
    (gt_root / "geometry_profiles").mkdir(parents=True, exist_ok=True)

    img_w = int(raw_meta["blender_render_configuration"]["width"])
    img_h = int(raw_meta["blender_render_configuration"]["height"])

    ext_by_fnum = {int(f["frame_number"]): f for f in extracted_frames}
    frame_records: list[dict[str, Any]] = []
    a4_frame_visibility: list[dict[str, Any]] = []

    yolo_box_count = 0
    a3_mask_count = 0
    a4_mask_count = 0

    for raw_frame in raw_meta["frames"]:
        f_num = int(raw_frame["frame"])
        ext_info = ext_by_fnum.get(f_num)
        if ext_info is None:
            continue

        img_filename = ext_info["filename"]
        img_rel_path = f"images/{split}/{img_filename}"
        stem = Path(img_filename).stem
        yolo_lines: list[str] = []
        frame_anoms: list[dict[str, Any]] = []

        for anom in raw_frame["anomalies"]:
            aid = str(anom["id"])
            w_meta = anomalies_world_meta.get(aid, {})
            world_coords = w_meta.get("world_center_m")

            is_visible = bool(anom.get("sufficiently_visible", anom.get("visible", False)))
            in_frustum = bool(anom.get("in_frustum", False))
            occluded = bool(anom.get("occluded", False))
            vis_frac = float(anom.get("visible_fraction", 1.0))
            fully_occluded = in_frustum and (vis_frac < 0.15)
            partially_occluded = occluded and not fully_occluded

            bbox = anom.get("bbox") if is_visible else None
            yolo_bbox = None
            mask_rel_path = None
            visible_pixel_area = 0

            if is_visible and bbox is not None:
                x1, y1, x2, y2 = [float(v) for v in bbox]
                visible_pixel_area = int(round(max(0.0, x2 - x1) * max(0.0, y2 - y1)))

            # Only YOLO-enabled object detection classes (0: animal_carcass, 1: grain_spillage)
            if is_visible and anom.get("yolo_enabled", False) and not fully_occluded:
                yb = anom.get("yolo_bbox")
                if yb is not None:
                    cx, cy, bw, bh = [float(v) for v in yb]
                    # Strictly clamp & validate normalized YOLO coordinates
                    cx = min(1.0, max(0.0, cx))
                    cy = min(1.0, max(0.0, cy))
                    bw = min(1.0, max(0.0, bw))
                    bh = min(1.0, max(0.0, bh))
                    if bw > 0.0 and bh > 0.0:
                        yolo_bbox = [round(cx, 6), round(cy, 6), round(bw, 6), round(bh, 6)]
                        yolo_lines.append(
                            f"{int(anom['class_id'])} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} "
                            f"{yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}"
                        )
                        yolo_box_count += 1

            # Segmentation masks for A3 (rail_crack) and A4 (rail_misalignment)
            if aid == "A3" and is_visible:
                m_file = seg_dir / "a3_masks" / f"{stem}_a3_mask.png"
                if m_file.is_file():
                    fg_px, mask_bbox = _mask_stats(m_file)
                    mask_rel_path = f"segmentation/{split}/a3_masks/{m_file.name}"
                    visible_pixel_area = fg_px
                    if mask_bbox is not None:
                        bbox = mask_bbox
                    a3_mask_count += 1
            elif aid == "A4" and is_visible:
                m_file = seg_dir / "a4_masks" / f"{stem}_a4_mask.png"
                if m_file.is_file():
                    fg_px, _ = _mask_stats(m_file)
                    mask_rel_path = f"segmentation/{split}/a4_masks/{m_file.name}"
                    visible_pixel_area = fg_px
                    a4_mask_count += 1

            if is_visible:
                vis_status = "partially_visible" if (anom.get("partially_clipped") or partially_occluded) else "fully_visible"
            elif fully_occluded:
                vis_status = "fully_occluded"
            elif anom.get("reason") == "behind_camera":
                vis_status = "behind_camera"
            elif not in_frustum:
                vis_status = "outside_frame"
            else:
                vis_status = "too_small_or_out_of_range"

            anom_record = {
                "anomaly_id": aid,
                "anomaly_category": anom["class"],
                "class_id": int(anom["class_id"]),
                "detection_method": anom["detection_method"],
                "yolo_enabled": bool(anom.get("yolo_enabled", False)),
                "position_m": float(anom["position_m"]),
                "world_coordinates_m": world_coords,
                "camera_to_anomaly_distance_m": float(anom["camera_distance_m"]),
                "visible": bool(is_visible),
                "visibility_status": vis_status,
                "in_frustum": in_frustum,
                "occluded": occluded,
                "partially_occluded": partially_occluded,
                "fully_occluded": fully_occluded,
                "visible_fraction": round(vis_frac, 4),
                "bbox": [round(float(v), 2) for v in bbox] if bbox is not None else None,
                "yolo_bbox": yolo_bbox,
                "visible_pixel_area": int(visible_pixel_area),
                "segmentation_mask": mask_rel_path,
            }
            frame_anoms.append(anom_record)

            if aid == "A4":
                a4_frame_visibility.append(
                    {
                        "frame_number": f_num,
                        "timestamp_s": float(raw_frame["timestamp_s"]),
                        "camera_position_m": float(raw_frame["camera_position_m"]),
                        "camera_to_a4_center_m": float(anom["camera_distance_m"]),
                        "visible": bool(is_visible),
                        "visibility_status": vis_status,
                        "bbox": anom_record["bbox"],
                        "spatial_mask": mask_rel_path,
                    }
                )

        # Write YOLO .txt label file (empty for negative/background frames)
        label_path = labels_dir / f"{stem}.txt"
        label_path.write_text(
            "\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8"
        )

        frame_records.append(
            {
                "dataset_type": "SYNTHETIC",
                "video_id": video_id,
                "source_video": source_video_path.name,
                "source_scene": source_scene,
                "split": split,
                "dataset_group": dataset_group,
                "frame_number": f_num,
                "frame_index": int(raw_frame["frame_index"]),
                "timestamp_s": float(raw_frame["timestamp_s"]),
                "train_speed_kmh": raw_meta["speed_kmh"],
                "camera_position_m": float(raw_frame["camera_position_m"]),
                "image_width": img_w,
                "image_height": img_h,
                "image_filename": img_filename,
                "image_path": img_rel_path,
                "label_path": f"labels/{split}/{stem}.txt",
                "anomalies": frame_anoms,
            }
        )

    # Save frame-level ground truth JSON
    frame_gt_path = gt_root / "frame_annotations" / f"{video_id}_frames.json"
    with frame_gt_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset_type": "SYNTHETIC",
                "video_id": video_id,
                "source_scene": source_scene,
                "split": split,
                "dataset_group": dataset_group,
                "train_speed_kmh": raw_meta["speed_kmh"],
                "fps": raw_meta["fps"],
                "frame_count": len(frame_records),
                "frames": frame_records,
            },
            handle,
            indent=2,
        )

    # Save anomaly world metadata JSON
    anom_meta_path = gt_root / "anomaly_metadata" / f"{video_id}_anomalies.json"
    with anom_meta_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset_type": "SYNTHETIC",
                "video_id": video_id,
                "source_scene": source_scene,
                "anomalies": anomalies_world_meta,
            },
            handle,
            indent=2,
        )

    # Save A4 geometry profile JSON
    a4_meta = anomalies_world_meta.get("A4", {})
    profile = a4_meta.get("displacement_profile") or _sample_a4_displacement_profile(
        a4_cfg, track_datum
    )
    geom_path = gt_root / "geometry_profiles" / f"{video_id}_a4_geometry.json"
    with geom_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset_type": "SYNTHETIC",
                "video_id": video_id,
                "source_scene": source_scene,
                "anomaly_id": "A4",
                "anomaly_category": "rail_misalignment",
                "detection_method": "geometric_analysis",
                "center_position_m": float(a4_cfg["position_m"]),
                "affected_span_m": float(a4_cfg["affected_span_m"]),
                "affected_region_m": [float(a4_cfg["start_m"]), float(a4_cfg["end_m"])],
                "max_lateral_deviation_m": float(a4_cfg["max_lateral_deviation_m"]),
                "affected_rails": str(a4_cfg.get("affected_rails", "both")),
                "sampling_interval_m": 0.25,
                "displacement_profile": profile,
                "frame_visibility": a4_frame_visibility,
            },
            handle,
            indent=2,
        )

    return {
        "video_id": video_id,
        "frame_gt_path": str(frame_gt_path),
        "anomaly_metadata_path": str(anom_meta_path),
        "geometry_profile_path": str(geom_path),
        "frames_written": len(frame_records),
        "yolo_boxes_written": yolo_box_count,
        "a3_masks_linked": a3_mask_count,
        "a4_masks_linked": a4_mask_count,
        "frame_records": frame_records,
    }


def write_dataset_configs(dataset_root: Path, anomaly_cfg: dict[str, Any]) -> dict[str, str]:
    """Write Ultralytics data.yaml (classes 0, 1 only) and full class_mapping.json."""
    cfg_dir = dataset_root / "configs"
    meta_dir = dataset_root / "metadata"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    yaml_text = "\n".join(
        [
            "# ITMS-VirtualTrack SYNTHETIC Railway Anomaly Detection Dataset",
            "# Controlled YOLO object-detection classes only (A1 & A2).",
            "# A3 (rail_crack) uses segmentation masks; A4 (rail_misalignment) uses geometry profiles.",
            f"path: {dataset_root.as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "",
            "nc: 2",
            "names:",
            "  0: animal_carcass",
            "  1: grain_spillage",
            "",
        ]
    )
    yaml_path = cfg_dir / "data.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")

    class_map = {
        "dataset_type": "SYNTHETIC",
        "yolo_object_detection_classes": {
            "0": "animal_carcass",
            "1": "grain_spillage",
        },
        "all_anomaly_categories": anomaly_cfg["classes"],
    }
    class_map_path = meta_dir / "class_mapping.json"
    with class_map_path.open("w", encoding="utf-8") as handle:
        json.dump(class_map, handle, indent=2)

    return {"data_yaml": str(yaml_path), "class_mapping": str(class_map_path)}

