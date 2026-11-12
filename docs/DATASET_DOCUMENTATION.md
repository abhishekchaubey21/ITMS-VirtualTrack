# Dataset Documentation — ITMS-VirtualTrack (Milestone 5)

> **SYNTHETIC DATASET LABEL**: Every video, extracted frame, YOLO label, segmentation mask, and geometry profile under [`dataset/synthetic/`](file:///d:/ITMS-VirtualTrack/dataset/synthetic) is **SYNTHETIC** Blender-generated data.

## 1. Dataset Composition & Leak-Free Splits

Two strictly scene-disjoint groups compose the dataset (`27` videos, `6,102` lossless `960×540` PNG frames, `1,633.425 MB` total):

1. **Group A — Controlled Speed Evaluation (`dataset_group: "controlled_evaluation"`, `split: "test"`)**:
   - All `7` M4 speed runs (`30, 60, 90, 120, 150, 180, 200 km/h`) over the fixed combined anomaly scene (`2,347` frames).
   - Preserved strictly out of `train` and `val` to prevent adjacent-frame leakage.
2. **Group B — Independently Randomized Scene Variants (`dataset_group: "randomized_variants"`, `20` scenes, `3,755` frames)**:
   - `70%` **Train** (`14` scenes `var_00`–`var_13`, `2,639` frames)
   - `15%` **Val** (`3` scenes `var_14`–`var_16`, `558` frames)
   - `15%` **Test** (`3` scenes `var_17`–`var_19`, `558` frames)
   - Each variant independently varies (`base_seed = 20261111`): sun elevation (`16°–66°`), sun azimuth (`75°–285°`), sky turbidity (`1.8–6.8`), sun/sky energy, exposure EV (`-0.35..+0.20`), ballast/ground/concrete surface tints, anomaly dimensions, lateral jitter, and yaw orientation.

## 2. Directory Layout

- `dataset/synthetic/videos/`: `27` MP4 source videos (`7` M4 eval + `20` variant runs).
- `dataset/synthetic/images/{train,val,test}/`: `6,102` lossless PNG frames (`train: 2,639`, `val: 558`, `test: 2,905`).
- `dataset/synthetic/labels/{train,val,test}/`: `6,102` YOLO `.txt` label files (`1,847` bounding boxes across classes `0: animal_carcass` and `1: grain_spillage`; empty `.txt` for negative frames).
- `dataset/synthetic/segmentation/{train,val,test}/{a3_masks,a4_masks}/`: `1,128` strictly binary `{0, 255}` PNG masks (`153` `A3` crack masks, `975` `A4` misalignment masks).
- `dataset/synthetic/ground_truth/{frame_annotations,anomaly_metadata,geometry_profiles}/`: Per-video frame-level JSON annotations, world coordinates, and `0.25 m`-sampled `A4` lateral displacement profiles.
- `dataset/synthetic/metadata/`: `dataset_manifest.json`, `dataset_statistics.json`, `class_mapping.json`.
- `dataset/synthetic/configs/data.yaml`: Ultralytics YOLO dataset config (`nc: 2`).

## 3. Anomaly Representation Policy

- `A1` (`animal_carcass`, class `0`) & `A2` (`grain_spillage`, class `1`): Annotated as normalized YOLO bounding boxes when visible and not fully occluded.
- `A3` (`rail_crack`, class `2`): Annotated with pixel-aligned binary segmentation masks, crack pixel area, and frame visibility metadata (not converted to standard YOLO detection labels).
- `A4` (`rail_misalignment`, class `3`): Annotated with `0.25 m` lateral displacement profiles over `[196.0, 204.0] m`, frame visibility metadata, and spatial rail-head masks.
