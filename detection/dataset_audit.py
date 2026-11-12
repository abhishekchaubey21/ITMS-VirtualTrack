"""Phase 1 Dataset Audit for ITMS-VirtualTrack Milestone 6.

Audits dataset_manifest.json, dataset_statistics.json, data.yaml, split counts,
per-class YOLO label distributions, A3 crack mask pixel distributions, A4 geometry
profiles, path integrity, and train/val/test isolation before any training begins.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_dataset_audit(
    dataset_root: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Execute complete Phase 1 audit and write m6_dataset_audit.json."""
    ds_root = dataset_root or (PROJECT_ROOT / "dataset" / "synthetic")
    out_json = output_path or (
        PROJECT_ROOT / "results" / "metrics" / "m6_dataset_audit.json"
    )

    manifest_path = ds_root / "metadata" / "dataset_manifest.json"
    stats_path = ds_root / "metadata" / "dataset_statistics.json"
    data_yaml_path = ds_root / "configs" / "data.yaml"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    data_yaml = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))

    # Inspect images & YOLO labels directly on disk per split
    split_image_counts: dict[str, int] = {}
    split_label_counts: dict[str, int] = {}
    yolo_per_split_class: dict[str, dict[str, int]] = {}
    positive_vs_negative_frames: dict[str, dict[str, int]] = {}
    test_breakdown: dict[str, dict[str, int]] = {
        "controlled_evaluation": {"images": 0, "class_0": 0, "class_1": 0, "positive_frames": 0},
        "randomized_variant_test": {"images": 0, "class_0": 0, "class_1": 0, "positive_frames": 0},
    }
    path_errors: list[str] = []

    for split in ("train", "val", "test"):
        img_dir = ds_root / data_yaml[split]
        lbl_dir = ds_root / "labels" / split
        imgs = sorted(img_dir.glob("*.png"))
        lbls = sorted(lbl_dir.glob("*.txt"))

        split_image_counts[split] = len(imgs)
        split_label_counts[split] = len(lbls)

        img_stems = {p.stem for p in imgs}
        lbl_stems = {p.stem for p in lbls}
        if img_stems != lbl_stems:
            path_errors.append(
                f"Split {split} stem mismatch: img-lbl={len(img_stems - lbl_stems)}, "
                f"lbl-img={len(lbl_stems - img_stems)}"
            )

        c0 = 0
        c1 = 0
        pos_frames = 0
        neg_frames = 0
        for lbl_p in lbls:
            lines = [
                ln.strip()
                for ln in lbl_p.read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]
            if lines:
                pos_frames += 1
            else:
                neg_frames += 1

            f_c0 = 0
            f_c1 = 0
            for ln in lines:
                parts = ln.split()
                cid = int(parts[0])
                if cid == 0:
                    c0 += 1
                    f_c0 += 1
                elif cid == 1:
                    c1 += 1
                    f_c1 += 1
                else:
                    path_errors.append(f"Unexpected YOLO class {cid} in {lbl_p}")

            if split == "test":
                grp = (
                    "controlled_evaluation"
                    if lbl_p.stem.startswith("eval_")
                    else "randomized_variant_test"
                )
                test_breakdown[grp]["images"] += 1
                test_breakdown[grp]["class_0"] += f_c0
                test_breakdown[grp]["class_1"] += f_c1
                if lines:
                    test_breakdown[grp]["positive_frames"] += 1

        yolo_per_split_class[split] = {
            "0_animal_carcass": c0,
            "1_grain_spillage": c1,
            "total_boxes": c0 + c1,
        }
        positive_vs_negative_frames[split] = {
            "positive_frames": pos_frames,
            "negative_background_frames": neg_frames,
            "positive_ratio": round(pos_frames / max(1, len(lbls)), 4),
        }

    # Check scene disjointness & controlled eval exclusion from train/val
    scenes_by_split: dict[str, set[str]] = defaultdict(set)
    eval_in_train_or_val: list[str] = []
    for v_entry in manifest["videos"]:
        sp = v_entry["split"]
        scenes_by_split[sp].add(v_entry["source_scene"])
        if sp in ("train", "val") and (
            v_entry["dataset_group"] == "controlled_evaluation"
            or v_entry["video_id"].startswith("eval_")
        ):
            eval_in_train_or_val.append(v_entry["video_id"])

    # Also check actual filenames inside train/ and val/ folders
    eval_files_in_train = [
        p.name for p in (ds_root / "images" / "train").glob("eval_*")
    ]
    eval_files_in_val = [
        p.name for p in (ds_root / "images" / "val").glob("eval_*")
    ]
    controlled_eval_excluded = (
        len(eval_in_train_or_val) == 0
        and len(eval_files_in_train) == 0
        and len(eval_files_in_val) == 0
    )

    overlap_tv = scenes_by_split["train"] & scenes_by_split["val"]
    overlap_tt = scenes_by_split["train"] & scenes_by_split["test"]
    overlap_vt = scenes_by_split["val"] & scenes_by_split["test"]
    no_variant_overlap = (len(overlap_tv) + len(overlap_tt) + len(overlap_vt)) == 0

    # Audit A3 crack segmentation mask counts & pixel statistics per split
    a3_mask_audit: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        m_dir = ds_root / "segmentation" / split / "a3_masks"
        masks = sorted(m_dir.glob("*.png"))
        fg_areas: list[int] = []
        for mp in masks:
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if m is None:
                path_errors.append(f"Unreadable A3 mask: {mp}")
                continue
            fg_areas.append(int(np.count_nonzero(m > 127)))
        a3_mask_audit[split] = {
            "mask_count": len(masks),
            "min_fg_pixels": min(fg_areas) if fg_areas else 0,
            "median_fg_pixels": float(np.median(fg_areas)) if fg_areas else 0.0,
            "max_fg_pixels": max(fg_areas) if fg_areas else 0,
            "mean_fg_fraction_of_960x540": round(
                (float(np.mean(fg_areas)) / (960.0 * 540.0)) if fg_areas else 0.0, 7
            ),
        }

    # Audit A4 geometry profiles & A4 masks
    geom_files = sorted((ds_root / "ground_truth" / "geometry_profiles").glob("*.json"))
    frame_gt_files = sorted((ds_root / "ground_truth" / "frame_annotations").glob("*.json"))
    a4_mask_counts = {
        sp: len(list((ds_root / "segmentation" / sp / "a4_masks").glob("*.png")))
        for sp in ("train", "val", "test")
    }

    # Verify paths referenced in data.yaml resolve cleanly
    data_yaml_root = Path(data_yaml["path"])
    yaml_valid = (
        data_yaml_root.is_dir()
        and (data_yaml_root / data_yaml["train"]).is_dir()
        and (data_yaml_root / data_yaml["val"]).is_dir()
        and (data_yaml_root / data_yaml["test"]).is_dir()
        and int(data_yaml["nc"]) == 2
    )

    # Class imbalance diagnostic
    train_c0 = yolo_per_split_class["train"]["0_animal_carcass"]
    train_c1 = yolo_per_split_class["train"]["1_grain_spillage"]
    c0_c1_ratio = round(max(train_c0, train_c1) / max(1, min(train_c0, train_c1)), 3)

    warnings_list: list[str] = []
    if c0_c1_ratio >= 1.4:
        warnings_list.append(
            f"Moderate YOLO class imbalance in train: animal_carcass={train_c0} vs grain_spillage={train_c1} (ratio {c0_c1_ratio}:1)."
        )
    if positive_vs_negative_frames["train"]["positive_ratio"] < 0.35:
        warnings_list.append(
            f"Majority negative background frames in train ({positive_vs_negative_frames['train']['negative_background_frames']}/{split_image_counts['train']})."
        )
    if a3_mask_audit["train"]["mean_fg_fraction_of_960x540"] < 0.001:
        warnings_list.append(
            f"Extreme foreground/background imbalance for A3 rail_crack in full 960x540 frames "
            f"(mean {a3_mask_audit['train']['median_fg_pixels']} px = {a3_mask_audit['train']['mean_fg_fraction_of_960x540']*100:.4f}%); "
            "a track-corridor ROI crop + focal/Dice loss is required for segmentation."
        )

    audit_passed = (
        yaml_valid
        and controlled_eval_excluded
        and no_variant_overlap
        and len(path_errors) == 0
        and len(geom_files) == len(manifest["videos"])
        and len(frame_gt_files) == len(manifest["videos"])
    )

    audit = {
        "dataset_type": "SYNTHETIC",
        "milestone": "M6_PHASE1_DATASET_AUDIT",
        "audit_passed": audit_passed,
        "data_yaml_path": str(data_yaml_path),
        "data_yaml_valid": yaml_valid,
        "data_yaml_contents": data_yaml,
        "split_image_counts": split_image_counts,
        "split_label_counts": split_label_counts,
        "yolo_annotations_per_split": yolo_per_split_class,
        "test_split_breakdown": test_breakdown,
        "positive_vs_negative_frames": positive_vs_negative_frames,
        "train_class_imbalance_ratio": c0_c1_ratio,
        "controlled_eval_excluded_from_train_val": controlled_eval_excluded,
        "no_scene_variant_overlap_across_splits": no_variant_overlap,
        "scenes_per_split": {k: sorted(v) for k, v in scenes_by_split.items()},
        "a3_segmentation_mask_audit": a3_mask_audit,
        "a4_mask_counts": a4_mask_counts,
        "geometry_profile_files_count": len(geom_files),
        "frame_annotation_files_count": len(frame_gt_files),
        "statistics_json_total_boxes_check": stats["total_yolo_bounding_box_annotations"],
        "actual_train_boxes_available": yolo_per_split_class["train"]["total_boxes"],
        "actual_val_boxes_available": yolo_per_split_class["val"]["total_boxes"],
        "warnings": warnings_list,
        "errors": path_errors,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


if __name__ == "__main__":
    report = run_dataset_audit()
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["audit_passed"] else 2)

