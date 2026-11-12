# ITMS-VirtualTrack — Milestone 6 Independent Scientific Evaluation Audit

> **SYNTHETIC EXPERIMENTAL PLATFORM NOTICE**: Every video, frame, label, model weight, prediction, and benchmark metric audited in this document originates from **SYNTHETIC** Blender 5.3.0 EEVEE simulations (`960x540`, `30 FPS`, `motion_blur.enabled = false`). None of the results represent real-world railway recordings, field calibration, or operational railway-safety certification.

- **Audit Scope**: Strictly read-only post-hoc verification of Milestone 6 training, frozen inference, evaluation code, ground-truth isolation, metric math, calibration bounds, and compute latency prior to Milestone 7 authorization.
- **Preservation Check**: No models were retrained, no checkpoints modified, no datasets regenerated, and no M2–M6 code/outputs altered. Only the 5 requested audit outputs were written under `D:\ITMS-VirtualTrack\results\metrics\m6_audit\`.
- **Verdict**: **`CONDITIONAL GO`** (No data leakage; core detections verified; requires mandatory methodological & reporting corrections inside Milestone 7 `evaluation/` and `dashboard/` before publishing comparative M7 conclusions—detailed in Section 7).

---

## 1. Executive Audit Classification: Verified vs. Corrected vs. Unverified Claims

| Category / Claim Audited | Audit Status | Independent Finding |
|---|---|---|
| Disjoint splits & controlled-eval exclusion (`train`: 2,639 / `val`: 558 / `test`: 2,905) | **VERIFIED** | Independent SHA-256 hashing of all `6,102` PNGs found **0** cross-split or within-split duplicate images. Prefixes, `data.yaml`, manifests, and `args.yaml` confirm `eval_30..200` and `var_17..19` were excluded from YOLO & U-Net training/validation. |
| Separate reporting of 7 controlled speeds vs. 3 independent variants | **VERIFIED** | `m6_evaluation_report.json` keeps `controlled_speed_benchmark` (`eval_30..200`) strictly separate from `independent_test_variants` (`var_17..19`). |
| YOLO `TP / FP / FN / P / R / F1` at `conf >= 0.25`, `IoU >= 0.50` | **VERIFIED** | Independent evaluator reproduces `tp_diff = 0`, `fp_diff = 0`, `fn_diff = 0` across all `10` test videos (`0` mismatch). |
| YOLO `mAP50` & `mAP50-95` methodology in `detection/evaluate.py` | **PARTIALLY CORRUPTED / TRUNCATED** | `predict.py` filtered boxes at `conf >= 0.25` *before* writing `pred_*.json`. `evaluate.py` integrated PR curves only down to `0.25`, slightly under-reporting `mAP50` on `eval_30kmh` (`0.9922` vs `0.9958`), `eval_60kmh` (`0.9846` vs `0.9977`), `var_18` (`0.9828` vs `1.0000`), and `var_19` (`0.9565` vs `0.9766`). Per-class metrics were omitted at test time. |
| YOLO `A1` first detection distance (`50.0 m`) | **CONDITIONALLY VERIFIED (LEFT-CENSORED)** | Technically exact from Frame 1 (`cam_y = 0.0 m`), but **left-censored** by track start at `0 m`: `A1` is already visible at `50.0 m` on Frame 1, so `50.0 m` measures track start bounds, not YOLO's maximum range limit. |
| YOLO `200 km/h` recall drop (`0.9535`) caused by sparse sampling | **VERIFIED & REFINED** | Forensics prove both `200 km/h` FNs (`A1` frame 26, `A2` frame 53) occur right at the `d = 3.704 m` camera bottom-edge crop (`y2 = 540.0 px`), identical in origin to `30/60 km/h` (`d = 3.61–3.89 m`, also `2` FNs each). Sparse sampling at `200 km/h` (`Δy = 1.852 m/f`) shrinks total GT boxes (`43` vs `276` at `30 km/h`), inflating `2` exit-edge misses from `0.72%` (`30 km/h`) to `4.65%` (`200 km/h`), while `90–180 km/h` step-over that `[3.6, 3.9] m` exit sliver cleanly (`FN = 0`). |
| `A3` U-Net threshold `0.85` selected on `val` only; `Dice = 0.779–0.820` | **VERIFIED WITH CAVEATS** | Selected strictly on `val` (`0` test leakage). However, training calibration sampled only `15` of `543` negative val frames and capped candidate grid at `0.85`. Independent full `558`-frame val sweep `(0.35..0.95)` confirms `0.85` actually maximizes full-val Dice (`0.8238`, FP frames `1/543`). Controlled-speed Dice matches (`0.7788–0.8203`), and all `7` non-GT-visible positive frames across test occurred exclusively `10.28–11.11 m` ahead of `150 m` (early crack detections just outside GT mask distance). |
| `A4` geometry analyzer uses no anomaly GT; unit & error definition discrepancy | **VERIFIED WITH BUG / PRIOR DISCLOSURE** | `analyze_frame()` uses **only** camera/track config (`cam_cfg`, `sim_cfg`) and frame odometry (`camera_position_m`). **HOWEVER**: (1) **`reconstruct_and_evaluate_track_profile` (`geometry_analyzer.py:366`) applies an unstated `[188.0, 212.0] m` filter** to `estimated_affected_region_m`, hiding single-bin spikes (`1.5–2.2 cm`) near `A3 @ 150 m` / `66 m` / `250 m`; (2) **"0.5–1.0 mm val" and "0.59–2.49 mm controlled" refer solely to scalar peak-height diff `||est|_max - |gt|_max|`**, whereas true pointwise worst-case profile error (`max_y |est(y) - gt(y)|`) is **`2.24–3.89 cm` (`22.4–38.9 mm`)** on controlled runs and **`1.89–4.06 cm` on val** (consistently peaking negatively at `-2.2` to `-3.9 cm` along `y = 201.0–202.25 m`); (3) **Exact GT odometry** (`camera_position_m`) is supplied per frame; (4) **Frame-level A4 FP/FN/distance metrics were omitted** from `m6_evaluation_report.json`. |
| Pipeline inference throughput vs `30 FPS` / `60 FPS` | **VERIFIED NON-REAL-TIME** | Reported latencies (`84.7–98.1 ms/f` = `10.2–11.8 FPS`) cover `predict_frame()` compute only (excluding disk PNG read `8.11 ms`, annotation `2.09 ms`, MP4 encode `3.15 ms`, totaling `98.50 ms/f` = `10.15 FPS`). **Fails both `30 FPS` (`33.33 ms`) and `60 FPS` (`16.67 ms`) budgets.** |

---

## 2. Evaluation Methodology & Leakage Audit (`m6_leakage_and_split_audit.json`)

Independent verification script [`run_m6_independent_audit.py`](file:///C:/Users/Abhishek/.gemini/antigravity/brain/93d20d1e-1c32-4030-b413-332212d6fce0/scratch/run_m6_independent_audit.py) checked disk directories, manifests, hashes, and code paths:

- **Exact SHA-256 Deduplication Check**: All `6,102` `.png` images across `train` (`2,639`), `val` (`558`), and `test` (`2,905`) were hashed. Result: **`0` cross-split duplicates** and **`0` within-split duplicates**.
- **Strict Scene-Variant Disjointness**:
  - `train` contains strictly `var_00`–`var_13` (`2,639` frames, `792` YOLO boxes: `468` carcass / `324` spillage; `64` A3 masks; `421` A4 masks).
  - `val` contains strictly `var_14`–`var_16` (`558` frames, `162` YOLO boxes: `97` carcass / `65` spillage; `15` A3 masks; `89` A4 masks).
  - `test` contains strictly `eval_30kmh`–`eval_200kmh` (`2,347` frames, `720` YOLO boxes: `438` carcass / `282` spillage; `59` A3 masks; `376` A4 masks) plus held-out randomized test variants `var_17`–`var_19` (`558` frames, `173` boxes: `104` carcass / `69` spillage; `15` A3 masks; `89` A4 masks).
- **Training & Calibration Code Isolation**:
  - YOLO (`baseline_v1/args.yaml`): trained strictly on `data.yaml` (`train` / `val`), `split: val`, `epochs=25`, `imgsz=640`, `batch=16`, `seed=42`. Controlled `eval_*` videos and `var_17..19` never entered training or early-stopping validation.
  - `RailCrackUNet` (`detection/crack_segmentation.py:240-366`): trained on `_collect_split_samples("train")` (`64` pos + `64` neg) and tuned threshold on `_collect_split_samples("val")` (`15` pos + `15` neg). Test ground truth was never touched.
  - `RailGeometryAnalyzer` (`detection/geometry_analyzer.py:23-301`): tuned exclusively against `val` (`var_14`–`var_16`) using only camera intrinsics/extrinsics (`config/camera_config.json`) and Indian Broad Gauge geometry (`1.676 m` inner gauge, `0.072 m` head width, `1.748 m` center-to-center in `config/simulation_config.json`).

---

## 3. YOLO Audit — A1 (`animal_carcass`) & A2 (`grain_spillage`)

All calculations were independently re-run from raw `gt_frames` and stored `pred_*.json` (`conf >= 0.25`, `IoU >= 0.50`), PLUS a read-only frozen-weight `conf >= 0.001` forward pass to quantify PR-curve truncation and compute per-class AP under both all-point envelope interpolation (used by `detection/evaluate.py`) and COCO 101-point interpolation.

### Controlled 7-Speed Benchmark — Overall & Per-Class Breakdown

| Speed | GT (A1 / A2) | TP / FP / FN | P / R / F1 (`conf>=0.25`) | mAP50 (trunc `0.25` vs `0.001`) | mAP50-95 (`0.001`) | A1 Carcass P / R / AP50 (`0.001`) | A2 Spillage P / R / AP50 (`0.001`) | First Detect Dist (A1 / A2) |
|---|---|---|---|---|---|---|---|---|
| `30 km/h` | 168 / 108 | 274 / 9 / 2 | 0.9682 / 0.9928 / 0.9803 | 0.9922 / **0.9958** | 0.8043 | 0.994 / 0.994 / 0.9940 | 0.930 / 0.991 / 0.9976 | `50.00 m*` / `33.61 m` |
| `60 km/h` | 84 / 54 | 136 / 2 / 2 | 0.9855 / 0.9855 / 0.9855 | 0.9846 / **0.9977** | 0.8021 | 0.988 / 0.988 / 0.9999 | 0.982 / 0.982 / 0.9956 | `50.00 m*` / `33.33 m` |
| `90 km/h` | 56 / 36 | 92 / 1 / 0 | 0.9892 / 1.0000 / 0.9946 | 1.0000 / 1.0000 | 0.8050 | 1.000 / 1.000 / 1.0000 | 0.973 / 1.000 / 1.0000 | `50.00 m*` / `33.33 m` |
| `120 km/h` | 42 / 27 | 69 / 1 / 0 | 0.9857 / 1.0000 / 0.9928 | 1.0000 / 1.0000 | 0.8221 | 1.000 / 1.000 / 1.0000 | 0.964 / 1.000 / 1.0000 | `50.00 m*` / `33.33 m` |
| `150 km/h` | 34 / 22 | 56 / 0 / 0 | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.8059 | 1.000 / 1.000 / 1.0000 | 1.000 / 1.000 / 1.0000 | `50.00 m*` / `33.33 m` |
| `180 km/h` | 28 / 18 | 46 / 0 / 0 | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.8308 | 1.000 / 1.000 / 1.0000 | 1.000 / 1.000 / 1.0000 | `50.00 m*` / `33.33 m` |
| `200 km/h` | 26 / 17 | 41 / 0 / 2 | 1.0000 / 0.9535 / 0.9762 | 0.9514 / 0.9514 | 0.7928 | 0.962 / 0.962 / 0.9615 | 1.000 / 0.941 / 0.9412 | `50.00 m*` / `33.33 m` |

*\*`A1` `50.00 m` is left-censored by track start at `y = 0 m` on Frame 1.*

### Independent Test Variants (`var_17`–`var_19`, Kept Separate)

| Variant (`Speed`) | GT (A1 / A2) | TP / FP / FN | P / R / F1 | mAP50 (trunc `0.25` / `0.001`) | mAP50-95 (`0.001`) | A1 AP50 / A2 AP50 (`0.001`) | Event Latency |
|---|---|---|---|---|---|---|---|
| `var_17` (`180 km/h`) | 28 / 16 | 44 / 2 / 0 | 0.9565 / 1.0000 / 0.9778 | 1.0000 / 1.0000 | 0.8128 | 1.0000 / 1.0000 | `0.0 s` |
| `var_18` (`120 km/h`) | 42 / 29 | 70 / 0 / 1 | 1.0000 / 0.9859 / 0.9929 | 0.9828 / **1.0000** | 0.7885 | 1.0000 / 1.0000 | `0.0167 s` (`A2` +1 frame) |
| `var_19` (`150 km/h`) | 34 / 24 | 56 / 1 / 2 | 0.9825 / 0.9655 / 0.9739 | 0.9565 / **0.9766** | 0.8547 | 1.0000 / 0.9531 | `0.0167 s` (`A2` +1 frame) |

### Forensic Findings on Event Grouping, FPs, FNs, and `200 km/h` Recall

- **Event Grouping**: `evaluate_yolo_video` correctly aggregates multi-frame detections by anomaly identity (`A1` and `A2`), reporting `2/2` events detected (`100%` event recall) at every speed with `0.0 s` mean detection delay across all 7 controlled speeds (Wait: note one code fragility: grouping hardcodes two IDs `A1`, `A2` assuming single instance per track; multi-anomaly scenes would require spatial track clustering).
- **Origin of Every False Positive**: **All `11` `fp_outside_positive_frames`** across controlled speeds (`8` at `30 km/h`, `1` at `60 km/h`, `1` at `90 km/h`, `1` at `120 km/h`) plus `2` in `var_17` are **valid early detections of `A2` grain spillage** at `31.67–36.67 m` before the GT box reached M5's `min_bbox_height_px = 8` rule (`visible=False`). Zero spurious background FPs on empty track, sleepers, ballast, horizon, or bushes occurred in all `2,905` test frames. The remaining `3` inside-positive FPs (`30 kmh` f167, `60 kmh` f84, `var_19` f70) were boundary IoU misses (`0.22–0.47 < 0.50`) as the object exited the bottom edge `y2 = 540`.
- **Exact Mechanism of `200 km/h` Recall Drop (`0.9535`)**: All `6` FNs across controlled runs (`30 kmh`: f168/f347; `60 kmh`: f84/f174; `200 kmh`: f26/f53) occurred solely at the **`d = 3.61–3.89 m` bottom-frame frustum exit** (`y2 = 540.0 px`, partially cropped). At `90–180 km/h`, frame steps (`0.833–1.667 m`) happen to step cleanly past that `[3.6, 3.9] m` sliver (e.g. `180 km/h` jumps from `d = 5.00 m` directly to `d = 3.33 m` off-screen). At `200 km/h` (`Δy = 1.852 m/frame`), Frame 26 lands squarely at `d = 3.704 m` (`A1`) and Frame 53 lands at `d = 3.704 m` (`A2`), yielding `2` misses—and because total GT boxes dropped from `276` (`30 km/h`) to `43` (`200 km/h`), those `2` bottom-edge misses lower recall by `2/43 = 4.65%` (`R = 0.9535`) vs `0.72%` (`R = 0.9928`) at `30 km/h`.

---

## 4. A3 Rail-Crack Segmentation Audit

- **Metric Formula & `0.85` Validation Calibration**: Dice `2TP / (2TP + FP + FN)` and IoU `TP / (TP + FP + FN)` are accumulated globally over pixels (correct micro-average across frames). Threshold `0.85` was selected **strictly on `val`**, though on a `1:1` subsample (`15` pos + `15` neg of `543` neg val frames) and clipped at the candidate grid ceiling `(0.35..0.85)`. Independently sweeping `(0.35..0.95)` across **all `558` `val` frames** confirms `0.85` remains the global Dice optimum (`0.8238`, FP frames `1/543`, vs `0.8150` at `0.75` and `0.8157` at `0.90`).
- **`65` Crack Overlays vs `59` GT Crack Frames Resolved**: All `7` non-GT-visible positive frames (`eval_30`: f501–504; `eval_60`: f252; `eval_90`: f168; `var_19`: f101) occurred strictly at **`cam_y = 138.89–139.72 m` (`10.28–11.11 m` ahead of the real `A3` crack at `150 m`)** where `A3` entered the `[4.2, 13] m` ROI prior to M5's `<= 10.0 m` `gt_visible` gate (`8–26 px`). Outside `[138.8, 140.0] m`, **`0` false-positive frames occurred across all `2,905` test frames** under the `min_area_px = 8` filter (raw threshold before `min_area_px=8` triggered `<= 5` frames per video with `1–7 px`).

| Dataset | GT Vis Frames / Hit | Event Det / Dist (`m`) / Delay | Dice (Vis-Only Frames) | Dice / IoU (All Frames, `min>=8px`) | Raw All-Frames Dice (no `min>=8px`) | Non-GT-Vis Frames |
|---|---|---|---|---|---|---|
| `eval_30kmh` | 22 / 22 (`100%`) | Yes / `10.00` / `0.0 s` | `0.7957` | `0.7801` / `0.6395` | `0.7799` | 4 (`10.3–11.1 m`) |
| `eval_60kmh` | 11 / 11 (`100%`) | Yes / `10.00` / `0.0 s` | `0.8068` | `0.7967` / `0.6622` | `0.7943` | 1 (`10.56 m`) |
| `eval_90kmh` | 8 / 8 (`100%`) | Yes / `10.00` / `0.0 s` | `0.7927` | `0.7793` / `0.6385` | `0.7793` | 1 (`10.83 m`) |
| `eval_120kmh` | 6 / 6 (`100%`) | Yes / `10.00` / `0.0 s` | `0.8203` | `0.8203` / `0.6954` | `0.8157` | 0 |
| `eval_150kmh` | 5 / 5 (`100%`) | Yes / `9.72` / `0.0 s` | `0.8033` | `0.8033` / `0.6712` | `0.7980` | 0 |
| `eval_180kmh` | 4 / 4 (`100%`) | Yes / `10.00` / `0.0 s` | `0.7788` | `0.7788` / `0.6377` | `0.7788` | 0 |
| `eval_200kmh` | 3 / 3 (`100%`) | Yes / `9.26` / `0.0 s` | `0.8148` | `0.8148` / `0.6875` | `0.8000` | 0 |
| `var_17` (180) | 4 / 4 (`100%`) | Yes / `10.00` / `0.0 s` | `0.8403` | `0.8403` / `0.7246` | `0.8403` | 0 |
| `var_18` (120) | 6 / 6 (`100%`) | Yes / `10.00` / `0.0 s` | `0.8279` | `0.8279` / `0.7063` | `0.8279` | 0 |
| `var_19` (150) | 5 / 5 (`100%`) | Yes / `9.72` / `0.0 s` | `0.8558` | `0.8335` / `0.7146` | `0.8335` | 1 (`11.11 m`) |

- **Small-Mask Sensitivity**: Test crack masks range `16–171 px` (median `47.0 px`), and total GT foreground pixels across the entire `200 km/h` video is only `157 px` (`3` frames). Small `~10 px` perimeter boundary shifts drive `~0.04` Dice swings, and at high speeds the `512x256` ROI captures only `3–4` frames (`~0.10–0.13 s` window).

---

## 5. A4 Geometry Calibration & Discrepancy Audit

- **Zero GT Leakage in Inference (`analyze_frame`)**: `RailGeometryAnalyzer` takes only `cam_cfg` (`h = 1.85 m`, `pitch = -7.5°`, `f = 35 mm`, `sensor_w = 36 mm`), `sim_cfg` (`1.676 m` gauge + `0.072 m` head width = `1.748 m` c2c), image pixels, and `camera_position_m`. No anomaly position (`200 m`), shape, or GT profile touches `analyze_frame()`. **Disclosure**: `predict.py` feeds exact GT `camera_position_m` (noise-free odometer/tachometer telemetry assumption).
- **Hidden Evaluation Prior in `reconstruct_and_evaluate_track_profile` (`geometry_analyzer.py:366`)**: Post-inference region estimation filters active bins (`>= 1.5 cm`) to **`188.0 <= y_m <= 212.0`**. Un-gated checks along the full `0–250 m` track prove straight-track MAE is tiny (`0.116–0.127 cm`), **yet small `1.54–2.25 cm` perturbations at `A3 @ 150 m` (`30/150 kmh`, `var_17`, `var_19`), `66.25 m` (`var_19`), and `250.25 m` (`200 kmh`) cross `1.5 cm`**—meaning un-gated `estimated_affected_region_m` would fail cleanly without line `366`'s `[188, 212] m` gate.
- **Resolving the Unit & Error-Definition Discrepancy (`0.5–1.0 mm` vs `0.59–2.49 mm` vs `cm`-scale MAE/RMSE vs `2.2–3.9 cm` pointwise max)**:

| Error Metric Definition (Over `33` Stations `[196.0, 204.0] m`) | Val (`var_14..16`, GT `8.77–10.51 cm`) | Controlled (`eval_30..200`, GT `8.50 cm`) | Test Variants (`var_17..19`, GT `8.28–10.22 cm`) | Physical Meaning |
|---|---|---|---|---|
| **Peak Amplitude Diff** `\|\|est\|_max - \|gt\|_max\|` | **`0.46–1.04 mm`** (`0.05–0.10 cm`) | **`0.59–2.49 mm`** (`0.06–0.25 cm`) | **`0.72–1.60 mm`** (`0.07–0.16 cm`) | Difference in peak height only; ignores `-0.25 m` longitudinal peak shift (`est_peak_y = 199.75 m` vs GT `200.00 m` in `8/10` test runs). |
| **Profile MAE** `mean_y \|est(y) - gt(y)\|` | **`0.37–0.64 cm`** (`3.71–6.43 mm`) | **`0.42–0.67 cm`** (`4.19–6.66 mm`) | **`0.45–0.99 cm`** (`4.46–9.86 mm`) | Average pointwise error across all `33` stations `[196, 204] m`. |
| **Profile RMSE** `sqrt(mean_y (est - gt)^2)` | **`0.66–1.18 cm`** (`6.60–11.77 mm`) | **`0.69–1.24 cm`** (`6.92–12.37 mm`) | **`0.63–1.38 cm`** (`6.30–13.82 mm`) | Penalizes localized slope errors across `[196, 204] m`. |
| **Pointwise Max Error** `max_y \|est(y) - gt(y)\|` | **`1.89–4.06 cm`** (`18.9–40.6 mm`) | **`2.24–3.89 cm`** (`22.4–38.9 mm`) | **`1.95–3.64 cm`** (`19.5–36.4 mm`) | **Largest single-station discrepancy**, occurring consistently on the receding slope (`201.0–202.25 m`, negative bias up to **`-3.89 cm`** at `200 km/h`, `201.25 m`) or `var_18` rising slope (`+3.64 cm` at `198.0 m`). |

- **Frame-Level A4 Performance (Previously Unreported)**:
  - Controlled speeds (`30..200 km/h`): frame Precision `95.2%–100.0%`, Recall `66.7%–71.0%` (`FNs` occur at `34–44 m` long range before close-up curvature exceeds `0.030 m`); `FPs` on controlled runs (`1–2` frames @ `30..150 km/h`) occurred exclusively at `cam_y = 195.56–195.83 m` (watching real `201.1–201.4 m` curvature just after A4 centroid passed `visible` cutoff). Event recall `7/7` (`100%`), first detection distance **`31.11–34.44 m`** (`0.23–1.23 s` after GT visibility).
  - Independent test variants: `var_17` `0` FPs (`31.67 m`); `var_18` `1` boundary FP at `195.56 m` (`33.33 m`); **`var_19` had `2` genuine false-positive frames ahead of anomalies** (`f16 @ 20.83 m`, `f19 @ 25.00 m`, `max_dev ~9.7 cm`, likely low-sun specular/shadow perturbation) plus `1` boundary frame at `195.83 m` (`precision = 0.875`).

---

## 6. Latency & Real-Time Throughput Audit

- **Measurement Scope**: `m6_evaluation_report.json` latency columns measure **`HybridRailwayDetector.predict_frame()` in-memory compute only** and **exclude** disk PNG reads, annotation drawing, crack overlay writes, JSON serialization, and `mp4v` encoding.
- **Measured Component Latency Breakdown (`eval_60kmh` first `100` frames post-warmup + full-run JSON stats)**:

| Stage | Device | Mean (`eval_60kmh` Bench / `10`-Run Range) | `p95` (`10`-Run Range) | Real-Time Implication |
|---|---|---|---|---|
| Image Load (`PNG` read vs `.mp4` decode) | CPU/Disk | `8.11 ms` (`.png`) / `1.23 ms` (`.mp4`) | — | Disk PNG loading adds ~`8 ms` overhead excluded from reported latency. |
| `YOLO` (`best.pt`, batch=1) | CUDA GPU | `18.21 ms` (`17.1–20.7 ms`) | `18.4–20.8 ms` | Fits `30 FPS` alone, exceeds `60 FPS` (`16.67 ms`) due to per-frame Python/H2D/D2H sync. |
| `RailCrackUNet` (`512x256` ROI, batch=1) | CUDA GPU | `24.11 ms` (`22.7–24.1 ms`) | `24.3–24.9 ms` | Dominated by synchronous H2D/D2H + CPU numpy post-processing per frame. |
| `RailGeometryAnalyzer` (`scanlines`) | CPU | `48.73 ms` (`42.9–53.2 ms`) | `50.7–72.8 ms` | **Primary bottleneck (~`51–54%` of total compute)**. |
| **Compute-Only Total** (`predict_frame`) | CPU+GPU | **`85.15 ms`** bench (`84.7–98.1 ms` runs) | **`93.9–117.4 ms`** | **`10.19–11.80 FPS` — NOT real-time (`2.55x–2.94x` over `30 FPS`, `5.08x–5.88x` over `60 FPS`)**. |
| `annotate_frame` + `VideoWriter` (`mp4v`) | CPU | `2.09 ms` + `3.15 ms` | — | End-to-end with PNG read + encode = **`98.50 ms/f` (`10.15 FPS`)** (~`91.6 ms/f` from `.mp4`). |

---

## 7. Required Corrections Before / During Milestone 7 & Final Recommendation

### Required Corrections (To Implement in M7 `evaluation/` & `dashboard/`, Preserving M6 Artifacts)

1. **Compute Unclipped YOLO PR Curves (`conf >= 0.001`) + Per-Class Breakdown in M7**: Do not rely solely on `conf >= 0.25`-filtered predictions for `mAP50`/`mAP50-95`; report full per-class metrics (`animal_carcass` vs `grain_spillage`) alongside `conf=0.25` operating-point `P/R/F1`.
2. **Label `A1` Range Censoring & Edge-Crop FNs**: Explicitly annotate `A1`'s `50.0 m` distance as track-start left-censored and distinguish early-approach `A2` detections (`31.7–36.7 m`) from background hallucinations.
3. **Standardize `A4` Geometry Units & Remove/Flag `[188, 212] m` Evaluation Gate**: Separate `peak_amplitude_diff` (`mm`), `MAE` (`cm`), `RMSE` (`cm`), and `max_pointwise_error` (`cm`, noting the `2.2–3.9 cm` receding-slope error) in every table/chart; report frame-level A4 FP/FN and track-wide (`0–250 m`) un-gated residuals.
4. **Report Both `gt_visible` and Full-Stream Metrics for `A3` (Plus Calibration Caveat)**: Expand M7 validation diagnostics to cite `A3` performance on negative frames explicitly (`0` FPs outside `10.28–11.11 m` approach) and document the `15`-neg `0.85`-endpoint training calibration bound.
5. **Present Real Throughput Constraints Honestly**: Display `compute-only (~10–12 FPS)` vs `end-to-end (~10 FPS)` against `30/60 FPS` target lines in M7 latency dashboards, explicitly noting exact `camera_position_m` odometry reliance and **`motion_blur.enabled = false`** synthetic scope.

### Final Recommendation: **`CONDITIONAL GO` for Milestone 7**

Milestone 6 passes leakage, split-isolation, calibration-blindness, and core detection integrity checks with high detection performance (`100%` event recall across all anomalies/speeds/variants and zero duplicate-frame contamination). Proceeding to Milestone 7 (`evaluation/` & `dashboard/`) is **approved conditionally** without retraining M6 models, **provided that Milestone 7 implements the 5 reporting, unit-standardization, un-gated evaluation, and latency-disclosure requirements above** and maintains full separation between the 7 controlled speeds, the 3 independent test variants, and all **SYNTHETIC** disclaimers.

---

### Generated Audit Outputs (`D:\ITMS-VirtualTrack\results\metrics\m6_audit\`)
1. [`m6_independent_audit_report.md`](file:///d:/ITMS-VirtualTrack/results/metrics/m6_audit/m6_independent_audit_report.md) — Narrative scientific audit and `CONDITIONAL GO` decision.
2. [`m6_independent_audit.json`](file:///d:/ITMS-VirtualTrack/results/metrics/m6_audit/m6_independent_audit.json) — Complete recalculated YOLO (`0.25` & `0.001`), A3 (`gt_visible` / all-frames / raw), A4 (33-point error profile, frame FP/FN, full-track residuals), and latency/I/O data.
3. [`m6_metric_consistency_report.json`](file:///d:/ITMS-VirtualTrack/results/metrics/m6_audit/m6_metric_consistency_report.json) — Reported vs recalculated consistency diffs, A3 full `558`-frame val sweep, current-code val A4 metrics, and unit definitions.
4. [`m6_leakage_and_split_audit.json`](file:///d:/ITMS-VirtualTrack/results/metrics/m6_audit/m6_leakage_and_split_audit.json) — SHA-256 verification across all `6,102` frames, prefix breakdown, and split isolation audit.
5. [`m6_final_results_tables.csv`](file:///d:/ITMS-VirtualTrack/results/metrics/m6_audit/m6_final_results_tables.csv) — Consolidated CSV benchmark table for M7 ingestion.

