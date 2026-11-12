# ITMS-VirtualTrack — Final Technical Report (Milestones 1–7)

> **MANDATORY SCIENTIFIC INTEGRITY & SYNTHETIC DATA NOTICE**:
> - **Entirely Synthetic**: All scenes, videos, labels, neural-network checkpoints, predictions, and evaluation metrics in this project are generated **synthetically** in **Blender 5.3.0 (EEVEE)** at `960x540`, `30 FPS`.
> - **Simplified Geometry & Physics**: The corridor models a straight, flat `250 m` Indian Broad Gauge (`1.676 m`) single track without curves, turnouts, switches, cant, overhead catenary, bogie vibration, aerodynamic buffeting, or CMOS sensor noise.
> - **Motion Blur Disabled**: `config/camera_config.json` specifies `motion_blur.enabled = false`; observed high-speed degradation therefore reflects temporal spatial sampling (`Δy = v / fps`) rather than optical shutter blur.
> - **Limited Variant Diversity**: Only `20` randomized lighting/environment variants exist (`14` train / `3` val / `3` held-out test).
> - **Benchmark Reuse**: The `7` controlled speeds (`30–200 km/h`) travel over **one fixed benchmark scene**, not seven independent real-world environments.
> - **No Real-World Validation or Safety Certification**: Zero physical railway footage has been used for training or evaluation. None of these findings establish field generalization or operational railway-safety certification.

---

## 1. Executive Summary & Research Answer

**Research Question**: *"How does train speed (30–200 km/h) affect the ability of a forward-facing computer vision system to detect, identify, and localize railway anomalies?"*

Using a deterministic synthetic Indian Broad Gauge (`1.676 m`) track with four fixed anomalies (`A1 @ 50 m`, `A2 @ 100 m`, `A3 @ 150 m`, `A4 @ 200 m`), a frozen hybrid pipeline combining **YOLOv8n** (`A1`/`A2`), **RailCrackUNet** (`512x256` track-corridor ROI segmentation for `A3`), and a pure-CPU **RailGeometryAnalyzer** (IPM + gauge-coupled rail correlation for `A4`) was benchmarked across `30, 60, 90, 120, 150, 180, 200 km/h` and separately on `3` held-out randomized test variants (`var_17`–`var_19`), under an authoritative read-only independent audit (`CONDITIONAL GO`):

1. **Event-Level Detection Holds (`100%`) Across All Speeds When Motion Blur Is Disabled**: Every unique anomaly (`A1`, `A2`, `A3`, `A4`) is detected (`2/2` YOLO events, `1/1` A3 event, `1/1` A4 event) across all `7` controlled speeds and `3` independent variants.
2. **Temporal Visibility & Sampling Stride Shrink Hyperbolically With Speed**: At fixed `30 FPS`, longitudinal travel per frame rises from **`0.278 m/frame` (`30 km/h`)** to **`1.852 m/frame` (`200 km/h`)**. Consequently, visible frames per anomaly shrink `6.7x`—reducing `A3` crack opportunities from **`22` frames (`1,339 px`) at `30 km/h`** to just **`3` frames (`157 px`, `0.10 s` window) at `200 km/h`**.
3. **200 km/h Recall Dip Is Explained by Exit-Edge Frustum Clipping + Denominator Shrinkage**: Authoritative YOLO (`conf >= 0.001` PR curve) holds `mAP50 = 0.9958–1.0000` from `30–180 km/h` (`mAP50-95 = 0.8021–0.8308`), dropping at `200 km/h` (`recall = 0.9535`, `mAP50 = 0.9514`) because Frame 26 (`A1`) and Frame 53 (`A2`) both land at `d = 3.704 m` where objects slice across the image bottom (`y2 = 540 px`). Those `2` exit misses represent `4.65%` of `43` GT boxes at `200 km/h` vs `0.72%` of `276` boxes at `30 km/h`.
4. **`A4` Geometry Reconstruction Sustains `0.42–0.67 cm` MAE, With `2.24–3.89 cm` Pointwise Slope Error & Asymmetric Bias**: Scalar peak-amplitude difference (`||est|_max - |gt|_max|`) stays within `0.59–2.49 mm`, while true pointwise worst-case error reaches `3.89 cm` at `200 km/h` on the `201.0–202.25 m` receding slope.
5. **End-to-End Throughput Fails Real-Time Camera Budgets**: Measured hybrid compute takes `84.7–98.1 ms/frame` (`10.19–11.80 FPS`), and full end-to-end execution (PNG read + compute + annotation + MP4 write on `eval_60kmh`) measures **`98.50 ms/frame` (`10.15 FPS`)**—missing `30 FPS` (`33.33 ms`) and `60 FPS` (`16.67 ms`) budgets primarily due to the `42.9–53.2 ms` CPU `RailGeometryAnalyzer` bottleneck.

---

## 2. System Architecture & Verification Summary

![System Architecture](file:///d:/ITMS-VirtualTrack/results/graphs/m7_system_architecture.png)

- **Leakage Proof ([`m6_leakage_and_split_audit.json`](file:///d:/ITMS-VirtualTrack/results/metrics/m6_audit/m6_leakage_and_split_audit.json))**: SHA-256 hashing across all `6,102` `.png` frames (`2,639` train `var_00`–`var_13` / `558` val `var_14`–`var_16` / `2,905` test) confirmed **`0` cross-split duplicates** and **`0` within-split duplicates**.
- **Training Isolation (`train` / `val` only)**:
  - **YOLOv8n** (`3.01M` params, COCO-initialized): `25` epochs (`imgsz=640`, `batch=16`, RTX 3050 6GB GPU, peak `1,880.22 MB` VRAM, `767.67 s`), best epoch `24` (`val P=1.0`, `R=0.9611`, `mAP50=0.9936`, `mAP50-95=0.8266`).
  - **RailCrackUNet** (`512x256` track-corridor ROI): `35` epochs (`297.08 s`), best epoch `32`, val-calibrated threshold `0.85` (`15+15` subsample; full-`558`-frame audit sweep confirmed `0.85` global val peak `Dice = 0.8238`, `1/543` FP negative frames).
  - **RailGeometryAnalyzer**: Analytical pinhole IPM (`h=1.85 m`, `pitch=-7.5°`, `f=35 mm`) + Indian Broad Gauge joint left/right rail peak correlation (`1.748 m ± 0.055 m`), tuned on `val` (`var_14`–`var_16`). Assumes exact frame `camera_position_m` telemetry.

---

## 3. Audit-Corrected Controlled 7-Speed Benchmark (`30–200 km/h`)

All metrics below match [`m6_independent_audit.json`](file:///d:/ITMS-VirtualTrack/results/metrics/m6_audit/m6_independent_audit.json) and CSV outputs in [`results/metrics/m7_evaluation/`](file:///d:/ITMS-VirtualTrack/results/metrics/m7_evaluation):

| Speed (`km/h`) | Frames | Stride (`m/f`) | YOLO `P / R / F1` (`conf>=0.25`) | YOLO `mAP50 / 50-95` (`conf>=0.001`) | A3 Crack Frames (Hits) | A3 `Dice` (Vis / All Frames) | A4 `Peak Diff` (`mm`) | A4 `MAE / RMSE / Max` (`cm`) | Compute (`ms/f` / `FPS`) |
|---|---|---|---|---|---|---|---|---|---|
| `30` | 901 | 0.278 | `0.9682 / 0.9928 / 0.9803` | `0.9958 / 0.8043` | 22 (22) | `0.7957 / 0.7801` | `1.03` | `0.59 / 0.91 / 2.34` | `98.09 / 10.19` |
| `60` | 451 | 0.556 | `0.9855 / 0.9855 / 0.9855` | `0.9977 / 0.8021` | 11 (11) | `0.8068 / 0.7967` | `1.05` | `0.56 / 0.91 / 2.57` | `92.07 / 10.86` |
| `90` | 301 | 0.833 | `0.9892 / 1.0000 / 0.9946` | `1.0000 / 0.8050` | 8 (8) | `0.7927 / 0.7793` | `1.42` | `0.67 / 1.09 / 3.36` | `85.56 / 11.69` |
| `120` | 226 | 1.111 | `0.9857 / 1.0000 / 0.9928` | `1.0000 / 0.8221` | 6 (6) | `0.8203 / 0.8203` | `1.36` | `0.59 / 0.93 / 2.59` | `84.71 / 11.80` |
| `150` | 181 | 1.389 | `1.0000 / 1.0000 / 1.0000` | `1.0000 / 0.8059` | 5 (5) | `0.8033 / 0.8033` | `1.78` | `0.42 / 0.69 / 2.24` | `85.07 / 11.75` |
| `180` | 151 | 1.667 | `1.0000 / 1.0000 / 1.0000` | `1.0000 / 0.8308` | 4 (4) | `0.7788 / 0.7788` | `0.59` | `0.49 / 0.80 / 3.08` | `85.63 / 11.68` |
| `200` | 136 | 1.852 | `1.0000 / 0.9535 / 0.9762` | `0.9514 / 0.7928` | 3 (3) | `0.8148 / 0.8148` | `2.49` | `0.63 / 1.24 / 3.89` | `84.92 / 11.78` |

### Key Analytical Charts

- **YOLO Per-Class & PR Audit**: [`m7_speed_vs_yolo_prf1.png`](file:///d:/ITMS-VirtualTrack/results/graphs/m7_speed_vs_yolo_prf1.png) and [`m7_speed_vs_map.png`](file:///d:/ITMS-VirtualTrack/results/graphs/m7_speed_vs_map.png) show why M6's `conf >= 0.25` pre-filtering slightly under-reported `mAP50` at `30 km/h` (`0.9922` vs `0.9958`) and `60 km/h` (`0.9846` vs `0.9977`). All `11` controlled `fp_outside_positive_frames` were valid early `A2` (`grain_spillage`) detections at `31.67–36.67 m` prior to M5's `min_bbox_height_px=8` visibility threshold, with zero background hallucinations.
- **A3 Segmentation Behavior**: [`m7_speed_vs_a3_segmentation.png`](file:///d:/ITMS-VirtualTrack/results/graphs/m7_speed_vs_a3_segmentation.png) separates visible-only from all-frame Dice (`0` FPs across all negative frames outside the `10.28–11.11 m` crack-approach corridor; `6` pre-visibility approach hits at `30/60/90 km/h`). At `200 km/h`, total ground-truth foreground is `157 px` (`3` frames).
- **A4 Geometry Calibration & Units**: [`m7_speed_vs_a4_geometry_metrics.png`](file:///d:/ITMS-VirtualTrack/results/graphs/m7_speed_vs_a4_geometry_metrics.png) and [`m7_a4_reconstructed_profiles.png`](file:///d:/ITMS-VirtualTrack/results/graphs/m7_a4_reconstructed_profiles.png) clarify the unit difference (`mm` scalar peak amplitude difference vs `cm` 33-station MAE/RMSE/pointwise max error) and highlight the negative `-2.2` to `-3.89 cm` bias along `[201.0, 202.25] m` plus the post-evaluation `[188.0, 212.0] m` region gate (`geometry_analyzer.py:366`).
- **Event Distance & Visibility Windows**: [`m7_speed_vs_detection_distance_timeline.png`](file:///d:/ITMS-VirtualTrack/results/graphs/m7_speed_vs_detection_distance_timeline.png) documents `A1` (`50.0 m` left-censored at Frame 1), `A2` (`33.33–33.61 m`), `A4` frame alerts (`31.11–34.44 m`, precision `95.2%–100.0%`, recall `66.7%–71.0%`), and `A3` (`9.26–10.00 m`).

---

## 4. Held-Out Independent Test Variants (`var_17`, `var_18`, `var_19`)

Reported strictly separately ([`m7_independent_test_variants_summary.png`](file:///d:/ITMS-VirtualTrack/results/graphs/m7_independent_test_variants_summary.png)):

| Variant (`Speed`) | YOLO `P / R / F1` (`0.25`) | YOLO `mAP50 / 50-95` (`0.001`) | A3 `Dice` (Vis / All) | A4 `Peak Diff` (`mm`) | A4 `MAE / RMSE / Max` (`cm`) | A4 Frame `P / R (FP)` |
|---|---|---|---|---|---|---|
| `var_17` (`180 km/h`, overcast) | `0.9565 / 1.0000 / 0.9778` | `1.0000 / 0.8128` | `0.8403 / 0.8403` | `0.72` | `0.45 / 0.63 / 1.95` | `1.0000 / 0.7083 (0)` |
| `var_18` (`120 km/h`, clear) | `1.0000 / 0.9859 / 0.9929` | `1.0000 / 0.7885` | `0.8279 / 0.8279` | `1.32` | `0.99 / 1.38 / 3.64` | `0.9630 / 0.7222 (1*)` |
| `var_19` (`150 km/h`, warm sun) | `0.9825 / 0.9655 / 0.9739` | `0.9766 / 0.8547` | `0.8558 / 0.8335` | `1.60` | `0.65 / 1.01 / 3.20` | `0.8750 / 0.7241 (3**)` |

*\*`1` boundary frame @ `195.56 m` seeing `201.07 m` bend. \*\*`2` genuine pre-anomaly FPs @ `20.83 m` & `25.00 m` under warm lighting + `1` boundary frame @ `195.83 m`.*

---

## 5. Latency & Real-Time Feasibility ([`m7_latency_vs_fps_budgets.png`](file:///d:/ITMS-VirtualTrack/results/graphs/m7_latency_vs_fps_budgets.png))

- **Compute-Only (`predict_frame`, GPU+CPU)**: `84.71–98.09 ms/frame` (`10.19–11.80 FPS`) across all `10` test runs (`p95 = 93.9–117.4 ms`):
  - `YOLOv8n` (CUDA GPU): `17.12–20.71 ms`
  - `RailCrackUNet` (CUDA GPU, synchronous H2D/D2H): `22.74–24.11 ms`
  - `RailGeometryAnalyzer` (CPU scanline correlation): **`42.92–53.24 ms`** (**`51–54%` primary bottleneck**)
- **Measured End-to-End (`eval_60kmh` 100-frame post-warmup timed benchmark)**: PNG disk read (`8.11 ms`) + compute (`85.15 ms`) + annotation (`2.09 ms`) + `mp4v` encoding (`3.15 ms`) = **`98.50 ms/frame` (`10.15 FPS`)** (~`91.6 ms/f` when decoding `.mp4`).
- **Real-Time Target Deficit**: Misses `30 FPS` (`33.33 ms`) by `2.55x–2.96x` and `60 FPS` (`16.67 ms`) by `5.08x–5.91x`.

---

## 6. Scientific Limitations & Future Engineering Roadmap

- **Limitations**: Synthetic Blender renders only; `motion_blur.enabled = false`; exact noiseless `camera_position_m` odometry; straight flat single track; `20` environment variants; COCO-initialized YOLO weights; `A1` range left-censored at `50.0 m`; `[188, 212] m` evaluation gate on A4 span; non-real-time sequential latency (`10.15 FPS`).
- **Next Technical Priorities**:
  1. Port [`RailGeometryAnalyzer`](file:///d:/ITMS-VirtualTrack/detection/geometry_analyzer.py) horizontal prefix-sum & peak correlation onto CUDA GPU / TensorRT + batch GPU U-Net/YOLO streams to break the `33.33 ms` (`30 FPS`) and `16.67 ms` (`60 FPS`) thresholds.
  2. Enable Blender camera shutter blur (`motion_blur.enabled = true`) at `60 FPS` `1920x1080` to measure true motion-blur degradation.
  3. Add curved/graded multi-track geometry, extended `[-40, 250] m` approach buffer (uncensoring `A1`), visual odometry estimation for `A4`, and real-world railway validation before any field deployment.

---

## 7. Interactive Dashboard & Reproducible Reporting Deliverables (M7)

- **Streamlit + Plotly Application**: [`dashboard/app.py`](file:///d:/ITMS-VirtualTrack/dashboard/app.py) serves all `9` audit-verified interactive sections with persistent `SYNTHETIC` disclosures, browser-playable H.264 prediction videos (`results/videos/m7_dashboard_h264/`), interactive `A3` overlay inspector, interactive `A4` longitudinal error profile explorer, and strict separation of controlled vs. held-out `var_17`–`var_19` runs.
- **Verified Dashboard Screenshots**: `10` live rendered captures saved in [`results/graphs/dashboard_screenshots/`](file:///d:/ITMS-VirtualTrack/results/graphs/dashboard_screenshots) (`01_overview_and_architecture.png` through `09_limitations_and_audit.png`, plus `03b_seven_speed_video_inspector.png`).
- **Structured Evaluation Tables**: `10` CSVs + `m7_comprehensive_evaluation.json` stored in [`results/metrics/m7_evaluation/`](file:///d:/ITMS-VirtualTrack/results/metrics/m7_evaluation), verified against [`m6_independent_audit.json`](file:///d:/ITMS-VirtualTrack/results/metrics/m6_audit/m6_independent_audit.json).
