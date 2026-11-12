"""Dataset manifest, statistics, leak-free validation, and visual QC for Milestone 5."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest_and_statistics(
    dataset_root: Path,
    video_entries: list[dict[str, Any]],
    all_frame_records: list[dict[str, Any]],
    split_plan: dict[str, Any],
) -> dict[str, Any]:
    """Write dataset_manifest.json and dataset_statistics.json."""
    meta_dir = dataset_root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate statistics
    frames_per_split: dict[str, int] = defaultdict(int)
    frames_per_group: dict[str, int] = defaultdict(int)
    frames_per_speed: dict[str, int] = defaultdict(int)

    anom_stats: dict[str, dict[str, int]] = {
        aid: {
            "total_frame_records": 0,
            "visible_frames": 0,
            "invisible_frames": 0,
            "partially_occluded_frames": 0,
            "fully_occluded_frames": 0,
        }
        for aid in ("A1", "A2", "A3", "A4")
    }
    yolo_bbox_count = 0
    a3_mask_count = 0
    a4_mask_count = 0
    geom_gt_count = len(video_entries)

    for rec in all_frame_records:
        frames_per_split[rec["split"]] += 1
        frames_per_group[rec["dataset_group"]] += 1
        frames_per_speed[f"{rec['train_speed_kmh']}kmh"] += 1

        for anom in rec["anomalies"]:
            aid = anom["anomaly_id"]
            st = anom_stats[aid]
            st["total_frame_records"] += 1
            if anom["visible"]:
                st["visible_frames"] += 1
            else:
                st["invisible_frames"] += 1
            if anom.get("partially_occluded"):
                st["partially_occluded_frames"] += 1
            if anom.get("fully_occluded"):
                st["fully_occluded_frames"] += 1

            if anom.get("yolo_bbox") is not None:
                yolo_bbox_count += 1
            if aid == "A3" and anom.get("segmentation_mask"):
                a3_mask_count += 1
            if aid == "A4" and anom.get("segmentation_mask"):
                a4_mask_count += 1

    storage_breakdown = {
        "videos_mb": round(_dir_size_bytes(dataset_root / "videos") / (1024 * 1024), 3),
        "images_mb": round(_dir_size_bytes(dataset_root / "images") / (1024 * 1024), 3),
        "labels_mb": round(_dir_size_bytes(dataset_root / "labels") / (1024 * 1024), 3),
        "segmentation_mb": round(
            _dir_size_bytes(dataset_root / "segmentation") / (1024 * 1024), 3
        ),
        "ground_truth_mb": round(
            _dir_size_bytes(dataset_root / "ground_truth") / (1024 * 1024), 3
        ),
        "total_dataset_mb": round(_dir_size_bytes(dataset_root) / (1024 * 1024), 3),
    }

    manifest = {
        "dataset_type": "SYNTHETIC",
        "milestone": "M5_DATASET_GENERATION",
        "dataset_root": str(dataset_root),
        "split_strategy": split_plan,
        "videos": video_entries,
        "total_frames": len(all_frame_records),
    }
    manifest_path = meta_dir / "dataset_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    statistics = {
        "dataset_type": "SYNTHETIC",
        "milestone": "M5_DATASET_GENERATION",
        "total_video_count": len(video_entries),
        "total_extracted_frame_count": len(all_frame_records),
        "frames_per_speed": dict(sorted(frames_per_speed.items(), key=lambda kv: int(kv[0].replace("kmh", "")))),
        "frames_per_split": dict(frames_per_split),
        "frames_per_dataset_group": dict(frames_per_group),
        "anomaly_frame_counts": anom_stats,
        "total_yolo_bounding_box_annotations": yolo_bbox_count,
        "total_segmentation_masks": {
            "A3_rail_crack": a3_mask_count,
            "A4_rail_misalignment": a4_mask_count,
            "total": a3_mask_count + a4_mask_count,
        },
        "total_geometry_ground_truth_profiles": geom_gt_count,
        "split_scene_distribution": split_plan,
        "storage_consumption_mb": storage_breakdown,
    }
    stats_path = meta_dir / "dataset_statistics.json"
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(statistics, handle, indent=2)

    return {
        "manifest_path": str(manifest_path),
        "statistics_path": str(stats_path),
        "statistics": statistics,
    }


def validate_dataset_quality(
    dataset_root: Path,
    video_entries: list[dict[str, Any]],
    all_frame_records: list[dict[str, Any]],
    expected_width: int = 960,
    expected_height: int = 540,
) -> dict[str, Any]:
    """Run exhaustive Section 9 automated dataset quality and zero-leakage validation."""
    errors: list[str] = []
    seen_filenames: set[str] = set()
    seen_hashes: dict[str, str] = {}
    scenes_by_split: dict[str, set[str]] = defaultdict(set)
    anom_categories_seen: set[str] = set()

    # Check image <-> label pairing per split
    for split in ("train", "val", "test"):
        img_files = {p.stem: p for p in (dataset_root / "images" / split).glob("*.png")}
        lbl_files = {p.stem: p for p in (dataset_root / "labels" / split).glob("*.txt")}
        if set(img_files.keys()) != set(lbl_files.keys()):
            missing_lbl = len(set(img_files) - set(lbl_files))
            orphan_lbl = len(set(lbl_files) - set(img_files))
            errors.append(
                f"Split {split}: image/label mismatch (missing_lbl={missing_lbl}, orphan_lbl={orphan_lbl})"
            )

    masks_checked = 0
    yolo_boxes_checked = 0
    images_checked = 0
    duplicate_images = 0

    for rec in all_frame_records:
        split = rec["split"]
        scenes_by_split[split].add(rec["source_scene"])

        img_path = dataset_root / rec["image_path"]
        lbl_path = dataset_root / rec["label_path"]
        fname = rec["image_filename"]
        if fname in seen_filenames:
            duplicate_images += 1
        seen_filenames.add(fname)

        if not img_path.is_file():
            errors.append(f"Missing image: {img_path}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            errors.append(f"Unreadable image: {img_path}")
            continue
        h, w = img.shape[:2]
        if w != expected_width or h != expected_height:
            errors.append(f"Bad image dimensions {w}x{h} on {img_path}")
        images_checked += 1

        # Exact binary file duplicate check via sha256
        fhash = _file_sha256(img_path)
        if fhash in seen_hashes:
            duplicate_images += 1
            errors.append(f"Duplicate image content: {img_path} == {seen_hashes[fhash]}")
        else:
            seen_hashes[fhash] = str(img_path)

        # Validate YOLO .txt lines
        if not lbl_path.is_file():
            errors.append(f"Missing label file: {lbl_path}")
        else:
            txt_lines = [
                ln.strip()
                for ln in lbl_path.read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]
            expected_yolo = [
                a for a in rec["anomalies"] if a.get("yolo_bbox") is not None
            ]
            if len(txt_lines) != len(expected_yolo):
                errors.append(
                    f"Label count mismatch in {lbl_path}: {len(txt_lines)} vs {len(expected_yolo)}"
                )
            for ln in txt_lines:
                parts = ln.split()
                if len(parts) != 5:
                    errors.append(f"Malformed YOLO line in {lbl_path}: {ln}")
                    continue
                cid = int(parts[0])
                cx, cy, bw, bh = [float(x) for x in parts[1:]]
                if cid not in (0, 1):
                    errors.append(f"Non-YOLO class {cid} in {lbl_path}")
                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                    errors.append(f"Invalid YOLO normalized coords in {lbl_path}: {ln}")
                yolo_boxes_checked += 1

        # Validate frame anomalies & segmentation masks
        for anom in rec["anomalies"]:
            anom_categories_seen.add(anom["anomaly_category"])
            if anom["fully_occluded"] and anom.get("yolo_bbox") is not None:
                errors.append(f"Fully occluded anomaly has YOLO bbox in {fname}")

            if anom["visible"] and anom.get("bbox"):
                x1, y1, x2, y2 = anom["bbox"]
                if not (0.0 <= x1 < x2 <= expected_width and 0.0 <= y1 < y2 <= expected_height):
                    errors.append(f"BBox out of range in {fname}: {anom['bbox']}")

            m_rel = anom.get("segmentation_mask")
            if m_rel:
                m_path = dataset_root / m_rel
                if not m_path.is_file():
                    errors.append(f"Missing mask: {m_path}")
                else:
                    m_img = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE)
                    if m_img is None or m_img.shape != (expected_height, expected_width):
                        errors.append(f"Invalid mask dimensions on {m_path}")
                    else:
                        uniq = set(int(v) for v in np.unique(m_img))
                        if not uniq.issubset({0, 255}):
                            errors.append(f"Non-binary mask values {uniq} on {m_path}")
                        if int(np.count_nonzero(m_img > 127)) == 0:
                            errors.append(f"Empty foreground mask on {m_path}")
                        masks_checked += 1

    # Zero leakage check across train / val / test
    train_scenes = scenes_by_split["train"]
    val_scenes = scenes_by_split["val"]
    test_scenes = scenes_by_split["test"]
    overlap_tv = train_scenes.intersection(val_scenes)
    overlap_tt = train_scenes.intersection(test_scenes)
    overlap_vt = val_scenes.intersection(test_scenes)
    leakage_free = len(overlap_tv) == 0 and len(overlap_tt) == 0 and len(overlap_vt) == 0
    if not leakage_free:
        errors.append(
            f"Scene leakage detected across splits: tv={overlap_tv}, tt={overlap_tt}, vt={overlap_vt}"
        )

    # Verify video frame counts match metadata
    for ve in video_entries:
        if ve["extracted_frames"] != ve["expected_frames"]:
            errors.append(
                f"Video {ve['video_id']} extracted {ve['extracted_frames']} != expected {ve['expected_frames']}"
            )

    all_four_present = anom_categories_seen == {
        "animal_carcass",
        "grain_spillage",
        "rail_crack",
        "rail_misalignment",
    }
    if not all_four_present:
        errors.append(f"Missing anomaly categories: {anom_categories_seen}")

    passed = len(errors) == 0 and duplicate_images == 0 and leakage_free and all_four_present
    return {
        "dataset_type": "SYNTHETIC",
        "milestone": "M5_DATASET_GENERATION",
        "passed": passed,
        "checks": {
            "images_checked": images_checked,
            "yolo_boxes_checked": yolo_boxes_checked,
            "segmentation_masks_checked": masks_checked,
            "duplicate_images": duplicate_images,
            "leakage_free": leakage_free,
            "train_scenes_count": len(train_scenes),
            "val_scenes_count": len(val_scenes),
            "test_scenes_count": len(test_scenes),
            "all_four_categories_present": all_four_present,
        },
        "errors_count": len(errors),
        "errors": errors[:50],
    }


def generate_visual_samples(
    dataset_root: Path,
    all_frame_records: list[dict[str, Any]],
    output_dir: Path,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate visual sample overlays for A1, A2, A3, A4, and multi-variant diversity."""
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[str, list[str]] = {
        "A1_samples": [],
        "A2_samples": [],
        "A3_overlays": [],
        "A4_overlays": [],
        "variant_diversity_grid": [],
        "combined_four_anomalies": [],
    }

    def find_candidates(aid: str, min_area: int = 300) -> list[dict[str, Any]]:
        cands = []
        for rec in all_frame_records:
            for a in rec["anomalies"]:
                if (
                    a["anomaly_id"] == aid
                    and a["visible"]
                    and a["visible_pixel_area"] >= min_area
                ):
                    cands.append((rec, a))
        return cands

    # A1 & A2 annotated samples (3 each across train/test)
    for aid, key, color in (
        ("A1", "A1_samples", (0, 230, 80)),
        ("A2", "A2_samples", (0, 200, 255)),
    ):
        cands = find_candidates(aid, min_area=400)
        chosen = rng.sample(cands, min(3, len(cands))) if cands else []
        for idx, (rec, a) in enumerate(chosen):
            img = cv2.imread(str(dataset_root / rec["image_path"]))
            if img is None:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in a["bbox"]]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            txt = (
                f"{aid} {a['anomaly_category']} | {rec['split']} | "
                f"{rec['train_speed_kmh']}kmh | d={a['camera_to_anomaly_distance_m']:.1f}m"
            )
            cv2.putText(img, txt, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)
            out_p = output_dir / f"{aid}_sample_{idx + 1}_{Path(rec['image_filename']).stem}.png"
            cv2.imwrite(str(out_p), img)
            generated[key].append(str(out_p))

    # A3 crack image/mask overlays (3 samples)
    a3_cands = [
        (rec, a)
        for rec in all_frame_records
        for a in rec["anomalies"]
        if a["anomaly_id"] == "A3" and a["visible"] and a.get("segmentation_mask")
    ]
    for idx, (rec, a) in enumerate(rng.sample(a3_cands, min(3, len(a3_cands)))):
        img = cv2.imread(str(dataset_root / rec["image_path"]))
        mask = cv2.imread(str(dataset_root / a["segmentation_mask"]), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue
        overlay = img.copy()
        fg = mask > 127
        overlay[fg] = (overlay[fg] * 0.3 + np.array([0, 0, 255]) * 0.7).astype(np.uint8)
        if a.get("bbox"):
            x1, y1, x2, y2 = [int(round(v)) for v in a["bbox"]]
            cv2.rectangle(overlay, (max(0, x1 - 4), max(0, y1 - 4)), (x2 + 4, y2 + 4), (0, 255, 255), 2)
        txt = (
            f"A3 rail_crack mask | {a['visible_pixel_area']}px | "
            f"{rec['split']} | d={a['camera_to_anomaly_distance_m']:.1f}m"
        )
        cv2.putText(overlay, txt, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2, cv2.LINE_AA)
        out_p = output_dir / f"A3_mask_overlay_{idx + 1}_{Path(rec['image_filename']).stem}.png"
        cv2.imwrite(str(out_p), overlay)
        generated["A3_overlays"].append(str(out_p))

    # A4 geometry + mask overlays (3 samples)
    a4_cands = [
        (rec, a)
        for rec in all_frame_records
        for a in rec["anomalies"]
        if a["anomaly_id"] == "A4"
        and a["visible"]
        and a.get("segmentation_mask")
        and 8.0 <= a["camera_to_anomaly_distance_m"] <= 20.0
    ]
    for idx, (rec, a) in enumerate(rng.sample(a4_cands, min(3, len(a4_cands)))):
        img = cv2.imread(str(dataset_root / rec["image_path"]))
        mask = cv2.imread(str(dataset_root / a["segmentation_mask"]), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue
        overlay = img.copy()
        fg = mask > 127
        overlay[fg] = (overlay[fg] * 0.45 + np.array([0, 220, 255]) * 0.55).astype(np.uint8)
        if a.get("bbox"):
            x1, y1, x2, y2 = [int(round(v)) for v in a["bbox"]]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 140, 255), 2)
        txt = (
            f"A4 rail_misalignment | {rec['split']} | span 196-204m | "
            f"d={a['camera_to_anomaly_distance_m']:.1f}m"
        )
        cv2.putText(overlay, txt, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 180, 255), 2, cv2.LINE_AA)
        out_p = output_dir / f"A4_geometry_overlay_{idx + 1}_{Path(rec['image_filename']).stem}.png"
        cv2.imwrite(str(out_p), overlay)
        generated["A4_overlays"].append(str(out_p))

    # Multi-variant training diversity montage (4 different training variants side-by-side at A1)
    train_a1_by_scene: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for rec in all_frame_records:
        if rec["split"] == "train":
            for a in rec["anomalies"]:
                if (
                    a["anomaly_id"] == "A1"
                    and a["visible"]
                    and 6.0 <= a["camera_to_anomaly_distance_m"] <= 11.0
                    and rec["source_scene"] not in train_a1_by_scene
                ):
                    train_a1_by_scene[rec["source_scene"]] = (rec, a)

    sample_scenes = list(train_a1_by_scene.values())[:4]
    if sample_scenes:
        tile_w, tile_h = 480, 270
        grid = np.zeros((2 * tile_h, 2 * tile_w, 3), dtype=np.uint8)
        for i, (rec, a) in enumerate(sample_scenes):
            img = cv2.imread(str(dataset_root / rec["image_path"]))
            if img is None:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in a["bbox"]]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 230, 80), 3)
            tile = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            cv2.putText(
                tile,
                f"{rec['source_scene']} ({rec['train_speed_kmh']}kmh)",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            r, c = i // 2, i % 2
            grid[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = tile
        grid_path = output_dir / "train_variant_diversity_grid.png"
        cv2.imwrite(str(grid_path), grid)
        generated["variant_diversity_grid"].append(str(grid_path))

    # Combined 4-anomaly showcase tiling one A1, A2, A3 overlay, and A4 overlay
    showcase_paths = [
        generated["A1_samples"][0] if generated["A1_samples"] else None,
        generated["A2_samples"][0] if generated["A2_samples"] else None,
        generated["A3_overlays"][0] if generated["A3_overlays"] else None,
        generated["A4_overlays"][0] if generated["A4_overlays"] else None,
    ]
    if all(p is not None for p in showcase_paths):
        tile_w, tile_h = 480, 270
        showcase = np.zeros((2 * tile_h, 2 * tile_w, 3), dtype=np.uint8)
        for i, p_str in enumerate(showcase_paths):
            img = cv2.imread(str(p_str))
            if img is None:
                continue
            tile = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            r, c = i // 2, i % 2
            showcase[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = tile
        sc_path = output_dir / "combined_four_anomalies_showcase.png"
        cv2.imwrite(str(sc_path), showcase)
        generated["combined_four_anomalies"].append(str(sc_path))

    return generated

