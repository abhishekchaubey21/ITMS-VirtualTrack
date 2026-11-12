# Simulation Methodology — ITMS-VirtualTrack (Milestone 1 Baseline)

> **SYNTHETIC DATA NOTICE**: All simulation runs produce **SYNTHETIC** imagery in Blender `5.3.0`.
> **Status**: Experimental protocol defined in Milestone 1; Blender simulation modules will be implemented in Milestones 2–4.

## 1. Kinematic Speed Model

For each speed condition $v_{\text{km/h}} \in \{30, 60, 90, 120, 150, 180, 200\}$:

$$v_{\text{m/s}} = \frac{v_{\text{km/h}}}{3.6}$$

At fixed frame rate $\text{FPS} = 60\text{ Hz}$ ($\Delta t = \frac{1}{60}\text{ s}$):

$$\Delta y = \frac{v_{\text{m/s}}}{60}, \qquad y(n) = v_{\text{m/s}} \cdot \frac{n}{\text{FPS}}$$

| Speed (km/h) | Speed (m/s) | Travel Time for 250 m (s) | Distance per Frame @ 60 FPS (m) | Approx. Total Frames |
|---:|---:|---:|---:|---:|
| 30 | 8.333 | 30.00 | 0.1389 | 1800 |
| 60 | 16.667 | 15.00 | 0.2778 | 900 |
| 90 | 25.000 | 10.00 | 0.4167 | 600 |
| 120 | 33.333 | 7.50 | 0.5556 | 450 |
| 150 | 41.667 | 6.00 | 0.6944 | 360 |
| 180 | 50.000 | 5.00 | 0.8333 | 300 |
| 200 | 55.556 | 4.50 | 0.9259 | 270 |

Every speed run traverses the exact same `250 m` environment (`Y = 0.0` to `Y = 250.0 m`) with identical camera intrinsics, mounting pose, and anomaly placements (`50 m`, `100 m`, `150 m`, `200 m`). Videos are rendered natively at each physical speed rather than retiming a single clip.

To include both the exact `0.0 m` origin and the `250.0 m` endpoint at exact step $\Delta y = v_{\text{m/s}} / \text{FPS}$, each run renders `num_intervals + 1` frames (`[0, num_intervals]`):
- Preview (`30 FPS`, `960×540`): `901, 451, 301, 226, 181, 151, 136` frames (`900..135` intervals).
- Production (`60 FPS`, `1920×1080`): `1801, 901, 601, 451, 361, 301, 271` frames (`1800..270` intervals).

## 2. Anomaly Taxonomy & Placement

1. **A1 — Animal carcass (`50 m`)**: Non-graphic procedural 3D obstruction near the track (`object_detection`).
2. **A2 — Grain spillage (`100 m`)**: Procedural grain pile and scattered kernels visually distinct from ballast (`object_detection`).
3. **A3 — Rail crack (`150 m`)**: Visual surface defect on the right rail head (`segmentation` + mask export).
4. **A4 — Rail misalignment (`200 m`)**: Controlled `8 m` sinusoidal lateral geometry displacement (`geometric_analysis` + mask export).

## 3. Constant-Speed Translation vs. Physically Calibrated Railway Camera

As a controlled Milestone 4 baseline experiment, camera motion is modeled as pure deterministic constant-speed translation along `+Y`. It explicitly does **not** yet model:
- Train hunting oscillation or rail-wheel suspension vibration.
- CMOS rolling-shutter scanline skew.
- Real camera auto-exposure / motion-adaptation dynamics.
- Physical sensor shot/read noise or lens distortion.
- Train acceleration and braking profiles.

Accordingly, Milestone 4 results isolate the pure geometric/temporal sampling effect of forward speed before introducing optional shutter motion blur or camera degradation models.

