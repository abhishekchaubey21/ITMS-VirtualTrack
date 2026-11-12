"""Interactive Milestone 7 Evaluation Dashboard for ITMS-VirtualTrack.

Presents the audit-corrected synthetic railway anomaly detection benchmark
using M6 Independent Audit reports and M7 evaluation CSV/JSON outputs.
Completely independent of Blender (uses only Streamlit, Plotly, Pandas, OpenCV/Pillow).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.detection_metrics import build_all_metric_tables  # noqa: E402
from evaluation.distance_analysis import build_anomaly_distance_timeline  # noqa: E402
from evaluation.speed_analysis import build_latency_table, build_speed_synthesis  # noqa: E402


st.set_page_config(
    page_title="ITMS-VirtualTrack | Synthetic Railway Anomaly Benchmark",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded",
)


SECTIONS = [
    "1. Overview & Architecture",
    "2. Synthetic Track & Anomaly Map",
    "3. Seven-Speed Benchmark (30-200 km/h)",
    "4. YOLO A1/A2 Object Detection",
    "5. RailCrackUNet A3 Segmentation",
    "6. A4 Rail Geometry Reconstruction",
    "7. Independent Test Variants (var_17-19)",
    "8. Pipeline Latency & Real-Time Audit",
    "9. Limitations, Audit & Future Work",
]


@st.cache_data(show_spinner=False)
def load_dashboard_data() -> dict[str, Any]:
    tables = build_all_metric_tables(PROJECT_ROOT)
    distance_df = build_anomaly_distance_timeline(tables)
    latency_df, io_summary = build_latency_table(tables)
    synthesis = build_speed_synthesis(tables, distance_df, latency_df, io_summary)
    return {
        "tables": tables,
        "distance_df": distance_df,
        "latency_df": latency_df,
        "io_summary": io_summary,
        "synthesis": synthesis,
    }


def render_scientific_disclaimer(compact: bool = False) -> None:
    """Render mandatory synthetic experimental disclosure on every page."""
    if compact:
        st.caption(
            "**SYNTHETIC DATA NOTICE**: Entirely synthetic Blender 5.3 EEVEE simulation "
            "| Straight 250 m Indian Broad Gauge (1.676 m) | Motion blur disabled (`motion_blur=False`) "
            "| 20 randomized variants | Controlled speeds share one benchmark environment "
            "| Not validated on real railway footage & not safety-certified."
        )
        return

    st.warning(
        "**SYNTHETIC EXPERIMENTAL PLATFORM — SCIENTIFIC INTEGRITY NOTICE**\n\n"
        "- **Entirely Synthetic**: All scenes, videos, labels, and metrics are Blender 5.3.0 EEVEE renders (`960x540`, `30 FPS`).\n"
        "- **Simplified Geometry & Physics**: Straight `250 m` single Indian Broad Gauge track (`1.676 m`); `motion_blur.enabled = false`, ideal pinhole camera, noiseless frame odometry (`camera_position_m`), and zero train suspension/vibration dynamics.\n"
        "- **Limited Environment Diversity**: Only `20` randomized environment variants exist (`14` train / `3` val / `3` held-out test). The `7` controlled-speed runs evaluate speed on **one fixed benchmark environment**, not seven independent real-world tracks.\n"
        "- **No Real-World Validation or Safety Certification**: Zero physical railway footage was used; results do **not** imply field generalization or railway-safety certification."
    )


def get_browser_video_path(pred_filename: str) -> Path:
    h264_p = PROJECT_ROOT / "results" / "videos" / "m7_dashboard_h264" / pred_filename
    if h264_p.is_file():
        return h264_p
    return PROJECT_ROOT / "results" / "videos" / "m6_predictions" / pred_filename


def load_pred_json(video_id: str) -> dict[str, Any]:
    p = PROJECT_ROOT / "results" / "metrics" / "m6_predictions" / f"pred_{video_id}.json"
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# PAGE RENDERERS (SECTIONS 1 TO 9)
# ---------------------------------------------------------------------------
def page_overview(data: dict[str, Any]) -> None:
    st.header("1. Project Overview & System Architecture")
    bundle = data["tables"]["bundle"]
    leak = bundle["leakage_audit"]
    train_sum = bundle["training_summary"]
    io_sum = data["io_summary"]

    st.info(
        '**Core Research Question**: *"How does train speed (30–200 km/h) affect the ability of a '
        'forward-facing computer vision system to detect, identify, and localize railway anomalies?"*'
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Virtual Track", "250 m", "1.676 m Broad Gauge", delta_color="off")
    c2.metric("Hashed PNG Frames", f"{leak['total_images_hashed_sha256']:,}", "0 cross-split dupes", delta_color="off")
    c3.metric("Event Recall (A1–A4)", "100%", "All 7 speeds & 3 variants", delta_color="off")
    c4.metric("YOLO val mAP50", f"{train_sum['yolo']['val_summary']['best_metrics']['metrics/mAP50(B)']:.3f}", "Epoch 24 (train/val only)", delta_color="off")
    c5.metric("E2E Throughput", f"{io_sum['end_to_end_fps_measured']:.2f} FPS", "Non-Real-Time (30/60 FPS fail)", delta_color="inverse")

    arch_img = PROJECT_ROOT / "results" / "graphs" / "m7_system_architecture.png"
    if arch_img.is_file():
        st.image(str(arch_img), caption="Verified M1–M7 System Architecture (Synthetic Experimental Pipeline)", width="stretch")

    st.subheader("M6 Independent Audit Verdict: CONDITIONAL GO")
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown(
            "**Verified Integrity Controls**:\n"
            "- **Zero Leakage (`m6_leakage_and_split_audit.json`)**: All `6,102` images SHA-256 hashed (`0` duplicates). `train` (`var_00`–`var_13`, `2,639` frames), `val` (`var_14`–`var_16`, `558` frames), and `test` (`2,905` frames) are strictly disjoint.\n"
            "- **Separated Test Reporting**: Seven controlled speeds (`eval_30..200kmh`, `2,347` frames) and three held-out randomized test variants (`var_17`–`var_19`, `558` frames) are never mixed.\n"
            "- **Authoritative Corrections Applied Here**: This dashboard uses `m6_independent_audit.json` (`conf >= 0.001` PR sweep for mAP, per-class metrics, unit-explicit A4 definitions, and measured E2E latency)."
        )
    with col_r:
        split_df = pd.DataFrame(
            [
                {"Split": "train (var_00–var_13)", **leak["recalculated_counts_per_split"]["train"], "A3 Masks": leak["a3_mask_counts_on_disk"]["train"]},
                {"Split": "val (var_14–var_16)", **leak["recalculated_counts_per_split"]["val"], "A3 Masks": leak["a3_mask_counts_on_disk"]["val"]},
                {"Split": "test (eval_30–200 + var_17–19)", **leak["recalculated_counts_per_split"]["test"], "A3 Masks": leak["a3_mask_counts_on_disk"]["test"]},
            ]
        )
        st.dataframe(split_df, width="stretch", hide_index=True)


def page_environment(data: dict[str, Any]) -> None:
    st.header("2. Synthetic Railway Environment & Four Anomaly Locations")
    bundle = data["tables"]["bundle"]
    sim = bundle["simulation_config"]
    cam = bundle["camera_config"]
    anom = bundle["anomaly_config"]["anomalies"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Track Gauge", f"{sim['track']['gauge_m']} m", sim["track"]["gauge_standard"], delta_color="off")
    col2.metric("Camera Height", f"{cam['mounting']['height_above_rail_head_m']} m", "Above rail head (+Z=2.452m)", delta_color="off")
    col3.metric("Camera Optics", f"{cam['optics']['focal_length_mm']} mm", f"Pitch {cam['orientation_deg']['pitch_deg']}°", delta_color="off")
    col4.metric("Preview / Motion Blur", f"{cam['resolution']['preview']['width']}x{cam['resolution']['preview']['height']}", f"30 FPS | blur={cam['motion_blur']['enabled']}", delta_color="off")

    # Interactive longitudinal schematic of 250 m track + 4 fixed anomalies
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 250], y=[0, 0], mode="lines",
            line=dict(color="#475569", width=6),
            name="250 m Straight Track (1.676 m Gauge)",
        )
    )
    a_specs = [
        ("A1", 50.0, "Animal Carcass (Non-graphic)<br>YOLOv8n (Class 0)<br>Left-censored @ 50m", "#16A34A", 0.35),
        ("A2", 100.0, "Grain Spillage<br>YOLOv8n (Class 1)<br>1.8m x 1.1m mound", "#CA8A04", 0.35),
        ("A3", 150.0, "Rail Crack (Right Rail)<br>RailCrackUNet ROI (512x256)<br>0.65m x 0.024m defect", "#DC2626", 0.35),
        ("A4", 200.0, "Rail Misalignment<br>IPM RailGeometryAnalyzer<br>[196-204 m], GT max 8.50 cm", "#EA580C", 0.35),
    ]
    for aid, pos, desc, col, yv in a_specs:
        fig.add_trace(
            go.Scatter(
                x=[pos], y=[yv], mode="markers+text",
                marker=dict(size=18, color=col, symbol="diamond"),
                text=[f"{aid} @ {int(pos)} m"],
                textposition="top center",
                hovertext=desc,
                name=f"{aid}: {anom[aid]['class']}",
            )
        )
    fig.add_vrect(x0=196, x1=204, fillcolor="#FED7AA", opacity=0.35, line_width=0, annotation_text="A4 [196-204m]", annotation_position="bottom right")
    fig.update_layout(
        title="Fixed Anomaly Placement Along 250 m Virtual Indian Broad Gauge Track (SYNTHETIC)",
        xaxis_title="Longitudinal Position Y (metres)",
        yaxis=dict(range=[-0.4, 0.85], showticklabels=False, title=""),
        height=320,
        margin=dict(l=30, r=20, t=55, b=40),
    )
    st.plotly_chart(fig, width="stretch")

    anom_table = pd.DataFrame(
        [
            {"ID": "A1", "Anomaly": "animal_carcass", "Position (m)": 50.0, "Method": "YOLOv8n (Class 0)", "Key Geometry": "1.15m x 0.45m x 0.42m @ +0.22m lateral"},
            {"ID": "A2", "Anomaly": "grain_spillage", "Position (m)": 100.0, "Method": "YOLOv8n (Class 1)", "Key Geometry": "1.80m x 1.10m x 0.14m mound + 450 grains"},
            {"ID": "A3", "Anomaly": "rail_crack", "Position (m)": 150.0, "Method": "RailCrackUNet ROI U-Net (Class 2)", "Key Geometry": "0.65m x 0.024m surface crack on right rail"},
            {"ID": "A4", "Anomaly": "rail_misalignment", "Position (m)": 200.0, "Method": "CPU IPM RailGeometryAnalyzer (Class 3)", "Key Geometry": "8.0m span [196-204m], 8.50cm peak shift (controlled)"},
        ]
    )
    st.dataframe(anom_table, width="stretch", hide_index=True)

    sc1, sc2 = st.columns(2)
    p_show = PROJECT_ROOT / "results" / "videos" / "m5_gt_samples" / "combined_four_anomalies_showcase.png"
    p_grid = PROJECT_ROOT / "results" / "videos" / "m5_gt_samples" / "train_variant_diversity_grid.png"
    if p_show.is_file():
        sc1.image(str(p_show), caption="M5 Ground-Truth Projection Showcase: A1–A4 Combined View (SYNTHETIC)", width="stretch")
    if p_grid.is_file():
        sc2.image(str(p_grid), caption="Randomized Lighting & Environmental Variants Subset (var_00–var_13 Train Split)", width="stretch")


def page_seven_speed_benchmark(data: dict[str, Any]) -> None:
    st.header("3. Seven-Speed Controlled Benchmark (30–200 km/h)")
    st.caption(
        "Evaluates the exact same track & anomaly layout (`eval_30kmh`–`eval_200kmh`, 2,347 total frames) "
        "using frozen models. Independent variants (`var_17`–`var_19`) are isolated on Section 7."
    )

    y_c = data["tables"]["yolo_summary"].query("dataset_group == 'controlled_evaluation'").sort_values("speed_kmh")
    a3_c = data["tables"]["a3_summary"].query("dataset_group == 'controlled_evaluation'").sort_values("speed_kmh")
    a4_c = data["tables"]["a4_summary"].query("dataset_group == 'controlled_evaluation'").sort_values("speed_kmh")
    d_c = data["distance_df"].query("dataset_group == 'controlled_evaluation'")
    lat_c = data["latency_df"].query("dataset_group == 'controlled_evaluation'").sort_values("speed_kmh")

    merged = y_c[["speed_kmh", "frame_count", "precision_conf025", "recall_conf025", "f1_conf025", "mAP50_conf001", "mAP50_95_conf001", "events_detected"]].merge(
        a3_c[["speed_kmh", "gt_visible_frames", "all_frames_dice", "all_frames_iou", "first_detection_distance_m"]],
        on="speed_kmh",
    ).merge(
        a4_c[["speed_kmh", "peak_amplitude_diff_mm", "profile_mae_cm", "profile_rmse_cm", "max_pointwise_error_cm"]],
        on="speed_kmh",
    ).merge(
        lat_c[["speed_kmh", "compute_only_ms_mean", "compute_only_fps"]],
        on="speed_kmh",
    )

    st.dataframe(merged, width="stretch", hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_d = px.line(
            d_c, x="speed_kmh", y="first_detect_distance_m", color="anomaly_id", markers=True,
            title="Earliest Event Detection Distance vs Speed (Note: A1 50m Left-Censored)",
            labels={"speed_kmh": "Train Speed (km/h)", "first_detect_distance_m": "Distance to Anomaly (m)"},
            hover_data=["anomaly_name", "gt_visible_frames", "detection_delay_s", "range_censoring_note"],
        )
        fig_d.update_xaxes(tickvals=merged["speed_kmh"].tolist())
        fig_d.update_yaxes(range=[0, 56])
        st.plotly_chart(fig_d, width="stretch")

    with c2:
        fig_w = px.line(
            d_c, x="speed_kmh", y="gt_visible_frames", color="anomaly_id", markers=True, log_y=True,
            title="Visible GT Frames vs Speed (30 FPS, Δy = v/30)",
            labels={"speed_kmh": "Train Speed (km/h)", "gt_visible_frames": "GT Visible Frames (log scale)"},
            hover_data=["stride_m_per_frame", "visible_window_duration_s"],
        )
        fig_w.update_xaxes(tickvals=merged["speed_kmh"].tolist())
        st.plotly_chart(fig_w, width="stretch")

    st.subheader("Controlled Run Video Inspector (Browser H.264 Playback & Per-Frame Telemetry)")
    sel_speed = st.select_slider("Select Controlled Train Speed (km/h)", options=merged["speed_kmh"].tolist(), value=60)
    vid_id = f"eval_{sel_speed}kmh"
    pred_vid = get_browser_video_path(f"pred_railway_{sel_speed}kmh.mp4")

    vc1, vc2 = st.columns([1.45, 1.0])
    with vc1:
        if pred_vid.is_file():
            st.video(str(pred_vid))
            st.caption(f"Annotated prediction video: `{pred_vid.relative_to(PROJECT_ROOT)}` (SYNTHETIC, H.264 browser copy)")
    with vc2:
        pj = load_pred_json(vid_id)
        frames_df = pd.DataFrame(
            [
                {
                    "Frame": f["frame_number"],
                    "Y (m)": round(f["camera_position_m"], 2),
                    "YOLO Hits": len(f["yolo_detections"]),
                    "A3 Det": f["crack_prediction"]["detected"],
                    "A3 Px": f["crack_prediction"]["pixel_area"],
                    "A4 Det": f["geometry_prediction"]["detected"],
                    "A4 Dev (cm)": round(f["geometry_prediction"]["max_estimated_deviation_m"] * 100.0, 2),
                    "Compute (ms)": round(f["latencies_ms"]["total_ms"], 1),
                }
                for f in pj["frames"]
            ]
        )
        pos_only = frames_df[(frames_df["YOLO Hits"] > 0) | (frames_df["A3 Det"]) | (frames_df["A4 Det"])]
        st.markdown(f"**Detected Anomaly Frames in `{vid_id}`** ({len(pos_only)} / {len(frames_df)} frames):")
        st.dataframe(pos_only, width="stretch", height=340, hide_index=True)


def page_yolo_detection(data: dict[str, Any]) -> None:
    st.header("4. YOLO A1 & A2 Object Detection Performance (Audit-Corrected)")
    st.markdown(
        "**Audit Corrections Enforced**: PR curves and `mAP50`/`mAP50-95` use predictions down to **`conf >= 0.001`** "
        "(fixing M6's `conf >= 0.25` truncation bias on `30/60 km/h` and `var_18/19`), while selected operating-point `Precision/Recall/F1` "
        "is evaluated at **`conf >= 0.25` (`IoU >= 0.50`)**."
    )

    y_c = data["tables"]["yolo_summary"].query("dataset_group == 'controlled_evaluation'").sort_values("speed_kmh")
    ypc_c = data["tables"]["yolo_per_class"].query("dataset_group == 'controlled_evaluation'").sort_values(["class_key", "speed_kmh"])
    fp_df = data["tables"]["yolo_fp_forensics"]
    fn_df = data["tables"]["yolo_fn_forensics"]

    c1, c2 = st.columns(2)
    with c1:
        fig_prf = go.Figure()
        fig_prf.add_trace(go.Scatter(x=y_c["speed_kmh"], y=y_c["precision_conf025"], mode="lines+markers", name="Precision (conf>=0.25)"))
        fig_prf.add_trace(go.Scatter(x=y_c["speed_kmh"], y=y_c["recall_conf025"], mode="lines+markers", name="Recall (conf>=0.25)"))
        fig_prf.add_trace(go.Scatter(x=y_c["speed_kmh"], y=y_c["f1_conf025"], mode="lines+markers", name="F1-score (conf>=0.25)"))
        fig_prf.update_layout(title="Operating-Point P / R / F1 vs Speed (conf>=0.25, IoU>=0.50)", xaxis_title="Speed (km/h)", yaxis_title="Score", yaxis_range=[0.88, 1.02])
        st.plotly_chart(fig_prf, width="stretch")

    with c2:
        fig_map = go.Figure()
        fig_map.add_trace(go.Scatter(x=y_c["speed_kmh"], y=y_c["mAP50_conf001"], mode="lines+markers", name="mAP50 (Authoritative conf>=0.001)"))
        fig_map.add_trace(go.Scatter(x=y_c["speed_kmh"], y=y_c["mAP50_trunc_conf025"], mode="lines+markers", line=dict(dash="dash"), name="mAP50 (M6 conf>=0.25 truncated)"))
        fig_map.add_trace(go.Scatter(x=y_c["speed_kmh"], y=y_c["mAP50_95_conf001"], mode="lines+markers", name="mAP50-95 (conf>=0.001)"))
        fig_map.update_layout(title="Unclipped vs Truncated mAP50 & mAP50-95 vs Speed", xaxis_title="Speed (km/h)", yaxis_title="AP", yaxis_range=[0.74, 1.02])
        st.plotly_chart(fig_map, width="stretch")

    st.subheader("Per-Class Metrics (A1 animal_carcass vs A2 grain_spillage) & Event-Level Isolation")
    st.dataframe(ypc_c.drop(columns=["dataset_group"]), width="stretch", hide_index=True)

    fc1, fc2 = st.columns(2)
    with fc1:
        st.info(
            "**A1 Left-Censoring & 200 km/h Recall Drop Explanation**:\n"
            "- **`A1` (`50.0 m`) Left-Censoring**: At Frame 1 (`cam_y = 0.0 m`), `A1` is already `50.0 m` ahead inside the frustum, so YOLO fires immediately (`latency = 0.0 s`). `50.0 m` reflects the track-start bound, not YOLO's maximum range limit.\n"
            "- **Why `200 km/h` Recall Dropped (`0.9535`)**: Both `200 km/h` misses (Frame 26 `A1` & Frame 53 `A2`) occur right at **`d = 3.704 m`** where the anomaly touches the image bottom (`y2 = 540 px`). `90–180 km/h` step-over that `[3.6, 3.9] m` exit sliver cleanly (`FN=0`), whereas at `200 km/h` (`43` GT boxes vs `276` at `30 km/h`), `2` exit-edge clips produce `-4.65%` recall."
        )
        st.markdown("**Forensic FN Log (All Runs)**")
        st.dataframe(fn_df, width="stretch", hide_index=True)

    with fc2:
        st.success(
            "**Zero Spurious Background Hallucinations**:\n"
            f"- Across all `{len(fp_df)}` false positives in all `10` test runs (`2,905` frames), **`13` FPs outside positive GT frames** were valid early detections of `A2` (`grain_spillage`) at **`31.67–36.67 m`** prior to M5's `min_bbox_height_px=8` visibility rule, and **`3` FPs** inside positive frames were bottom-edge crop boxes with `IoU = 0.22–0.47 < 0.50`."
        )
        st.markdown("**Forensic FP Log (All Runs)**")
        st.dataframe(fp_df, width="stretch", hide_index=True)


def page_a3_segmentation(data: dict[str, Any]) -> None:
    st.header("5. RailCrackUNet A3 Segmentation Performance")
    a3_c = data["tables"]["a3_summary"].query("dataset_group == 'controlled_evaluation'").sort_values("speed_kmh")
    pre_df = data["tables"]["a3_pre_visibility_detections"]
    sweep_df = data["tables"]["a3_val_threshold_sweep"]

    c1, c2 = st.columns(2)
    with c1:
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(x=a3_c["speed_kmh"], y=a3_c["vis_only_dice"], mode="lines+markers", name="Dice (GT Visible Frames Only)"))
        fig_s.add_trace(go.Scatter(x=a3_c["speed_kmh"], y=a3_c["all_frames_dice"], mode="lines+markers", line=dict(dash="dash"), name="Dice (All Frames incl. Pre-Vis 10–11m)"))
        fig_s.add_trace(go.Scatter(x=a3_c["speed_kmh"], y=a3_c["all_frames_iou"], mode="lines+markers", name="IoU (All Frames)"))
        fig_s.update_layout(title="A3 Dice & IoU vs Speed (Visible-Only vs All-Frame)", xaxis_title="Speed (km/h)", yaxis_title="Score", yaxis_range=[0.55, 0.90])
        st.plotly_chart(fig_s, width="stretch")

    with c2:
        fig_cal = make_subplots(specs=[[{"secondary_y": True}]])
        fig_cal.add_trace(go.Scatter(x=sweep_df["threshold"], y=sweep_df["dice_all558"], mode="lines+markers", name="Full Val Dice (558 frames)"), secondary_y=False)
        fig_cal.add_trace(go.Scatter(x=sweep_df["threshold"], y=sweep_df["fp_on_negative_frames"], mode="lines+markers", line=dict(dash="dot", color="#DC2626"), name="FP Negative Val Frames"), secondary_y=True)
        fig_cal.add_vline(x=0.85, line_dash="dash", line_color="#16A34A", annotation_text="Selected Val Threshold (0.85)")
        fig_cal.update_layout(title="Audit Sweep Across ALL 558 Validation Frames (Threshold Calibration Check)", xaxis_title="Probability Threshold")
        fig_cal.update_yaxes(title_text="Global Val Dice", secondary_y=False)
        fig_cal.update_yaxes(title_text="FP Negative Frames", secondary_y=True)
        st.plotly_chart(fig_cal, width="stretch")

    st.dataframe(a3_c.drop(columns=["dataset_group"]), width="stretch", hide_index=True)

    col_a, col_b = st.columns(2)
    col_a.info(
        "**Early Pre-Visibility Detections vs Negative-Frame FPs**:\n"
        "- **`0` false positives** occurred across all negative frames outside `[138.8, 140.0] m` (`2,281` controlled negative frames + `542` variant negative frames) using `min_area >= 8 px`.\n"
        "- All `7` non-GT-visible positive frames across controlled & variant runs (`4` @ `30`, `1` @ `60`, `1` @ `90`, `1` @ `var_19`) occurred strictly at **`10.28–11.11 m` ahead of `150 m`**, detecting the real crack right before M5's `<= 10.0 m` GT gate.\n"
        "- **Extreme Small-Mask Sensitivity (`200 km/h`)**: Total GT crack foreground drops to just **`157 px` across `3` frames (`~0.10 s` window)** at `200 km/h`, vs `1,339 px` across `22` frames at `30 km/h`."
    )
    col_b.dataframe(pre_df, width="stretch", hide_index=True)

    st.subheader("Verified GT (Green) vs RailCrackUNet Prediction (Red) Overlays")
    ov_dir = PROJECT_ROOT / "results" / "videos" / "m6_predictions" / "crack_overlays"
    ov_files = sorted(ov_dir.glob("*.png"))
    if ov_files:
        sel_ov_speed = st.selectbox("Filter Crack Overlays by Speed", [30, 60, 90, 120, 150, 180, 200], index=1)
        matching = [p for p in ov_files if p.name.startswith(f"eval_{sel_ov_speed}kmh_")]
        if matching:
            sel_img_name = st.selectbox("Select Overlay Frame", [p.name for p in matching], index=min(len(matching) // 2, len(matching) - 1))
            st.image(str(ov_dir / sel_img_name), caption=f"{sel_img_name} — Green: M5 GT Mask | Red: RailCrackUNet Prediction (SYNTHETIC)", width="stretch")


def page_a4_geometry(data: dict[str, Any]) -> None:
    st.header("6. A4 Rail Geometry Reconstruction & Anomaly Detection")
    st.warning(
        "**Unit & Error-Definition Rule**: Scalar **Peak-Amplitude Difference** (`0.59–2.49 mm`, `||est|_max - |gt|_max|`) measures ONLY the maximum lateral amplitude height while ignoring `-0.25 m` peak longitudinal shifts (`199.75 m` vs `200.0 m` GT in `8/10` runs). It does **NOT** represent full reconstructed profile accuracy—true **Profile MAE is `0.42–0.67 cm`**, **RMSE is `0.69–1.24 cm`**, and **Pointwise Max Error reaches `2.24–3.89 cm`** along `[196, 204] m`!"
    )

    a4_c = data["tables"]["a4_summary"].query("dataset_group == 'controlled_evaluation'").sort_values("speed_kmh")
    a4_p = data["tables"]["a4_error_profiles"].query("dataset_group == 'controlled_evaluation'")

    c1, c2 = st.columns(2)
    with c1:
        fig_u = go.Figure()
        fig_u.add_trace(go.Scatter(x=a4_c["speed_kmh"], y=a4_c["profile_mae_cm"], mode="lines+markers", name="Profile MAE (cm) — [196,204]m"))
        fig_u.add_trace(go.Scatter(x=a4_c["speed_kmh"], y=a4_c["profile_rmse_cm"], mode="lines+markers", name="Profile RMSE (cm) — [196,204]m"))
        fig_u.add_trace(go.Scatter(x=a4_c["speed_kmh"], y=a4_c["max_pointwise_error_cm"], mode="lines+markers", line=dict(dash="dash", color="#DC2626"), name="Pointwise Max |Err| (cm)"))
        fig_u.add_trace(go.Scatter(x=a4_c["speed_kmh"], y=a4_c["outside_196_204_mae_cm"], mode="lines+markers", line=dict(dash="dot", color="#059669"), name="Outside [196,204]m Straight MAE (cm)"))
        fig_u.update_layout(title="Profile-Wide Errors (cm) vs Speed", xaxis_title="Speed (km/h)", yaxis_title="Error (cm)", yaxis_range=[0, 4.4])
        st.plotly_chart(fig_u, width="stretch")

    with c2:
        sel_speeds_prof = st.multiselect("Compare Controlled Speeds on A4 Profile", [30, 60, 90, 120, 150, 180, 200], default=[30, 120, 200])
        fig_pr = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("Reconstructed Displacement vs GT (cm)", "Pointwise Signed Error (est - GT cm)"))
        gt_sub = a4_p[a4_p["video_id"] == "eval_30kmh"].sort_values("y_m")
        fig_pr.add_trace(go.Scatter(x=gt_sub["y_m"], y=gt_sub["gt_cm"], mode="lines", line=dict(color="#111827", width=3), name="Ground Truth (GT max 8.50 cm)"), row=1, col=1)
        for sp in sel_speeds_prof:
            sub = a4_p[a4_p["video_id"] == f"eval_{sp}kmh"].sort_values("y_m")
            fig_pr.add_trace(go.Scatter(x=sub["y_m"], y=sub["est_cm"], mode="lines+markers", marker=dict(size=4), name=f"Est @ {sp} km/h"), row=1, col=1)
            fig_pr.add_trace(go.Scatter(x=sub["y_m"], y=sub["signed_err_cm"], mode="lines+markers", marker=dict(size=4), name=f"Err @ {sp} km/h"), row=2, col=1)
        fig_pr.add_vrect(x0=200.75, x1=202.25, fillcolor="#FEE2E2", opacity=0.45, line_width=0, row=2, col=1)
        fig_pr.update_xaxes(title_text="Track Position Y (m)", row=2, col=1)
        fig_pr.update_yaxes(title_text="cm", row=1, col=1)
        fig_pr.update_yaxes(title_text="Error (cm)", row=2, col=1)
        fig_pr.update_layout(height=470, margin=dict(t=45, b=30))
        st.plotly_chart(fig_pr, width="stretch")

    st.subheader("Audit-Corrected A4 Controlled Table (Separating Peak mm vs Profile cm, Frame P/R & Post-Gate Disclosure)")
    st.dataframe(a4_c.drop(columns=["dataset_group"]), width="stretch", hide_index=True)
    st.info(
        "**`188.0–212.0 m` Post-Evaluation Filter Disclosure (`geometry_analyzer.py:366`)**:\n"
        "- `RailGeometryAnalyzer.analyze_frame()` runs **blindly** along `0–250 m` using only `cam_cfg`, `sim_cfg`, and `camera_position_m` odometry (zero anomaly GT leakage).\n"
        "- However, post-inference `reconstruct_and_evaluate_track_profile()` restricts `estimated_affected_region_m` to **`188.0 <= y_m <= 212.0 m`**, concealing single-bin `1.54–2.25 cm` spikes at `A3 @ 150.25 m` (`30/150 km/h`, `var_17`, `var_19`), `66.25 m` (`var_19`), and `250.25 m` (`200 km/h`). Outside `[196, 204] m`, full-track residual MAE is `0.120–0.127 cm`."
    )


def page_independent_variants(data: dict[str, Any]) -> None:
    st.header("7. Held-Out Independent Test Variants (`var_17`, `var_18`, `var_19`)")
    st.info(
        "These `3` randomized Blender weather, sun angle, surface tint, and anomaly-scale/jitter test scenes are kept "
        "**strictly separate** from both `train/val` (`var_00`–`var_16`) and the `7` controlled-speed benchmark runs (`eval_*`)."
    )

    y_v = data["tables"]["yolo_summary"].query("dataset_group != 'controlled_evaluation'")
    a3_v = data["tables"]["a3_summary"].query("dataset_group != 'controlled_evaluation'")
    a4_v = data["tables"]["a4_summary"].query("dataset_group != 'controlled_evaluation'")
    a4_vp = data["tables"]["a4_error_profiles"].query("dataset_group != 'controlled_evaluation'")

    comb_v = y_v[["video_id", "speed_kmh", "frame_count", "precision_conf025", "recall_conf025", "f1_conf025", "mAP50_conf001", "mAP50_95_conf001"]].merge(
        a3_v[["video_id", "vis_only_dice", "all_frames_dice", "all_frames_iou", "first_detection_distance_m"]],
        on="video_id",
    ).merge(
        a4_v[["video_id", "gt_max_dev_cm", "est_max_dev_cm", "peak_amplitude_diff_mm", "profile_mae_cm", "profile_rmse_cm", "max_pointwise_error_cm", "frame_fp", "frame_precision", "frame_recall"]],
        on="video_id",
    )
    st.dataframe(comb_v, width="stretch", hide_index=True)

    c1, c2 = st.columns([1.2, 1.0])
    manifest_v = {v["video_id"]: v for v in data["tables"]["bundle"]["dataset_manifest"]["videos"] if v["video_id"] in ("var_17", "var_18", "var_19")}
    sel_var = c1.selectbox("Inspect Test Variant", ["var_17", "var_18", "var_19"])
    with c1:
        vp_vid = get_browser_video_path(f"pred_{sel_var}.mp4")
        if vp_vid.is_file():
            st.video(str(vp_vid))
        st.json(manifest_v[sel_var]["variant_params"])

    with c2:
        sub_p = a4_vp[a4_vp["video_id"] == sel_var].sort_values("y_m")
        fig_vp = go.Figure()
        fig_vp.add_trace(go.Scatter(x=sub_p["y_m"], y=sub_p["gt_cm"], mode="lines", line=dict(color="#111827", width=3), name="Variant GT A4 Profile"))
        fig_vp.add_trace(go.Scatter(x=sub_p["y_m"], y=sub_p["est_cm"], mode="lines+markers", line=dict(color="#EA580C"), name=f"Reconstructed {sel_var}"))
        fig_vp.update_layout(title=f"{sel_var} ({manifest_v[sel_var]['speed_kmh']} km/h) A4 Reconstruction", xaxis_title="Track Y (m)", yaxis_title="Lateral Shift (cm)")
        st.plotly_chart(fig_vp, width="stretch")
        if sel_var == "var_19":
            st.warning(
                "`var_19` Note: Low-sun/warm lighting produced `2` genuine early false-positive A4 frames (`f16 @ 20.83 m`, "
                "`f19 @ 25.00 m`, `frame precision = 0.875`) and `1` pre-visibility crack hit (`f101 @ 11.11 m`), while `mAP50 (0.9766)` remained high."
            )


def page_latency_realtime(data: dict[str, Any]) -> None:
    st.header("8. Pipeline Latency & Real-Time Throughput Audit")
    lat_c = data["latency_df"].query("dataset_group == 'controlled_evaluation'").sort_values("speed_kmh")
    io_sum = data["io_summary"]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Compute-Only Mean (Controlled)", f"{lat_c['compute_only_ms_mean'].mean():.2f} ms/f", f"{lat_c['compute_only_fps'].mean():.2f} FPS (Non-RT)", delta_color="off")
    k2.metric("E2E Measured (60kmh Bench)", f"{io_sum['end_to_end_png_compute_annotate_encode_ms_mean']:.2f} ms/f", f"{io_sum['end_to_end_fps_measured']:.2f} FPS (Authoritative)", delta_color="off")
    k3.metric("30 FPS Frame Budget", "33.33 ms/f", "FAILS (2.55x–2.96x too slow)", delta_color="inverse")
    k4.metric("60 FPS Frame Budget", "16.67 ms/f", "FAILS (5.08x–5.91x too slow)", delta_color="inverse")

    c1, c2 = st.columns(2)
    with c1:
        fig_l = go.Figure()
        x_s = [f"{s} km/h" for s in lat_c["speed_kmh"]]
        fig_l.add_trace(go.Bar(x=x_s, y=lat_c["yolo_ms_mean"], name="YOLO GPU (17.2–20.7 ms)", marker_color="#16A34A"))
        fig_l.add_trace(go.Bar(x=x_s, y=lat_c["crack_unet_ms_mean"], name="RailCrackUNet GPU (22.7–24.1 ms)", marker_color="#DC2626"))
        fig_l.add_trace(go.Bar(x=x_s, y=lat_c["geometry_ms_mean"], name="Geometry IPM CPU (42.9–53.2 ms, Bottleneck)", marker_color="#EA580C"))
        fig_l.add_hline(
            y=33.333,
            line_dash="dash",
            line_color="#2563EB",
            annotation_text="30 FPS Budget (33.33 ms)",
            annotation_position="top right",
            annotation=dict(bgcolor="rgba(255,255,255,0.92)", font=dict(size=11, color="#1D4ED8")),
        )
        fig_l.add_hline(
            y=16.667,
            line_dash="dot",
            line_color="#7C3AED",
            annotation_text="60 FPS Budget (16.67 ms)",
            annotation_position="bottom right",
            annotation=dict(bgcolor="rgba(255,255,255,0.92)", font=dict(size=11, color="#6D28D9")),
        )
        fig_l.update_layout(barmode="stack", title="Compute-Only Latency Stack vs Real-Time Budgets", yaxis_title="ms / frame", yaxis_range=[0, 118])
        st.plotly_chart(fig_l, width="stretch")

    with c2:
        stage_df = pd.DataFrame(
            [
                {"Stage": "PNG Disk Read (cv2.imread)", "Device": "CPU/Disk", "Mean (ms/f)": io_sum["png_disk_read_ms_mean"], "Note": "Excluded from reported predict_frame latency"},
                {"Stage": "MP4 Decode (VideoCapture)", "Device": "CPU", "Mean (ms/f)": io_sum["mp4_video_decode_ms_mean"], "Note": "Alternative to PNG disk read"},
                {"Stage": "Hybrid Compute (predict_frame)", "Device": "GPU+CPU", "Mean (ms/f)": io_sum["compute_only_predict_frame_ms_mean"], "Note": "YOLO + U-Net + Geometry IPM (11.74 FPS)"},
                {"Stage": "Frame Overlay (annotate_frame)", "Device": "CPU", "Mean (ms/f)": io_sum["annotate_frame_ms_mean"], "Note": "Excluded from compute-only total"},
                {"Stage": "MP4 Encode (VideoWriter)", "Device": "CPU", "Mean (ms/f)": io_sum["cv2_videowriter_encode_ms_mean"], "Note": "Excluded from compute-only total"},
                {"Stage": "End-to-End (PNG + Compute + Annot + Enc)", "Device": "Full", "Mean (ms/f)": io_sum["end_to_end_png_compute_annotate_encode_ms_mean"], "Note": "10.15 FPS Measured End-to-End Throughput"},
            ]
        )
        st.dataframe(stage_df, width="stretch", hide_index=True)
        st.error(
            "**Real-Time Verdict**: Neither compute-only (`10.19–11.80 FPS`) nor end-to-end (`10.15 FPS`) meets `30 FPS` or `60 FPS`. "
            "`RailGeometryAnalyzer` CPU scanline loops consume **`51–54%`** of compute latency, while synchronous GPU H2D/D2H transfers "
            "inflate `RailCrackUNet` to `~23–24 ms/f`."
        )

    st.dataframe(data["latency_df"].drop(columns=["budget_30fps_ms", "budget_60fps_ms"]), width="stretch", hide_index=True)


def page_limitations_future(data: dict[str, Any]) -> None:
    st.header("9. Limitations, Audit Findings & Future Work")
    render_scientific_disclaimer(compact=False)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Methodological & Experimental Limitations")
        st.markdown(
            "1. **Motion Blur & Physical Optics Disabled**: `camera_config.json` sets `motion_blur.enabled = false`. High-speed (`180–200 km/h`) degradation observed here reflects longitudinal sampling sparsity (`Δy = 1.67–1.85 m/f`) rather than physical shutter smear or rolling-shutter skew.\n"
            "2. **Exact Odometry Assumption (`camera_position_m`)**: `RailGeometryAnalyzer` uses exact noiseless frame longitudinal telemetry (`camera_position_m`) from GT JSONs.\n"
            "3. **Post-Evaluation `[188, 212] m` Gate**: Track-span reporting in `reconstruct_and_evaluate_track_profile` filters out bins outside `188–212 m`.\n"
            "4. **Left-Censored `A1` Detection Range**: Track start (`0 m`) is `50.0 m` from `A1`, preventing measurement of true max YOLO detection distance beyond `50 m`.\n"
            "5. **Small `A3` Calibration Pool & Tiny Pixel Counts**: `0.85` U-Net threshold was chosen from a `15+15` val subsample (`0.85` grid ceiling), though an independent `558`-frame sweep confirmed `0.85` peak Dice. Test `200 km/h` spans only `157 px` across `3` frames.\n"
            "6. **COCO Initialization & Straight Single Track**: Generic COCO YOLO weights fine-tuned on only `20` simplified synthetic variants (`14/3/3`); excludes curves, turnouts, switches, rain/fog particles, or real railway footage."
        )
    with col2:
        st.subheader("Future Work & Optimization Roadmap")
        st.markdown(
            "1. **Real-Time Architecture Upgrade (Target >= 60 FPS)**: Port `RailGeometryAnalyzer` 1D row cumsum & peak correlation from CPU loop (`~49 ms`) to batched CUDA/PyTorch tensor ops or Numba/C++, fuse GPU preprocessing for YOLO + U-Net with pinned memory / TensorRT FP16.\n"
            "2. **Enable Blender Shutter Motion Blur (`motion_blur.enabled=True`) & 60 FPS Production**: Evaluate motion-blur degradation vs frame rate directly at `1920x1080` `60 FPS`.\n"
            "3. **Visual Odometry / IMU Fusion for A4 Geometry**: Replace ground-truth `camera_position_m` with optical-flow / sleeper-counting odometry + noise stress testing.\n"
            "4. **Extend Track Pre-Buffer (`-40 m` Start) & Spatial Event Clustering**: Remove `A1`'s `50 m` left-censoring and generalize `A1`/`A2`/`A4` grouping to arbitrary multi-anomaly densities without fixed `[188, 212] m` gates.\n"
            "5. **Sim-to-Real Domain Adaptation**: Validate frozen representations against real annotated railway inspection datasets before any field trial."
        )


def main() -> None:
    data = load_dashboard_data()

    st.sidebar.title("ITMS-VirtualTrack (M7)")
    st.sidebar.caption("SYNTHETIC Railway Anomaly Evaluation")

    default_idx = 0
    qp_sec = st.query_params.get("section")
    if qp_sec is not None:
        try:
            idx_cand = int(qp_sec)
            if 0 <= idx_cand < len(SECTIONS):
                default_idx = idx_cand
        except ValueError:
            pass

    selected_page = st.sidebar.radio("Dashboard Sections", SECTIONS, index=default_idx)

    st.sidebar.divider()
    st.sidebar.markdown(
        "**Verified Benchmark Scope**\n"
        "- Track: `250 m` straight (`1.676 m`)\n"
        "- Speeds: `30–200 km/h` (`eval_*`)\n"
        "- Held-out: `var_17`, `var_18`, `var_19`\n"
        "- Audit Status: **CONDITIONAL GO**\n"
        "- Throughput: **`10.15 FPS` E2E** (Non-RT)"
    )

    st.title("ITMS-VirtualTrack — Synthetic Railway Anomaly Detection Dashboard")
    render_scientific_disclaimer(compact=True)
    st.divider()

    page_map = {
        SECTIONS[0]: page_overview,
        SECTIONS[1]: page_environment,
        SECTIONS[2]: page_seven_speed_benchmark,
        SECTIONS[3]: page_yolo_detection,
        SECTIONS[4]: page_a3_segmentation,
        SECTIONS[5]: page_a4_geometry,
        SECTIONS[6]: page_independent_variants,
        SECTIONS[7]: page_latency_realtime,
        SECTIONS[8]: page_limitations_future,
    }
    page_map[selected_page](data)

    st.divider()
    render_scientific_disclaimer(compact=False)


if __name__ == "__main__":
    main()
