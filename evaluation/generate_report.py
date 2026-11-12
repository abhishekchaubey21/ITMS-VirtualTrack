"""Milestone 7 evaluation orchestrator, comparative charts, CSV/JSON reports.

Strictly uses the verified M6 independent audit outputs as authoritative data,
applies all M6 conditional audit corrections, avoids interpolating unmeasured
speeds, and separates controlled-speed runs from var_17-var_19 test variants.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.detection_metrics import build_all_metric_tables  # noqa: E402
from evaluation.distance_analysis import build_anomaly_distance_timeline  # noqa: E402
from evaluation.speed_analysis import build_latency_table, build_speed_synthesis  # noqa: E402


SYNTHETIC_FOOTER = (
    "SYNTHETIC BLENDER 5.3 DATA | Simplified straight 250 m Indian Broad Gauge (1.676 m) | "
    "motion_blur=False | Not real-world or safety validated"
)


def _apply_footer(fig: plt.Figure) -> None:
    fig.text(
        0.5,
        0.01,
        SYNTHETIC_FOOTER,
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
        style="italic",
    )


def generate_system_architecture_diagram(out_png: Path) -> None:
    """Render updated project architecture diagram reflecting verified M1-M7 flow."""
    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")

    ax.text(
        50,
        59.2,
        "ITMS-VirtualTrack — Verified End-to-End Architecture (M1–M7)",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        50,
        56.6,
        "SYNTHETIC SIMULATION PLATFORM | Straight 250 m Indian Broad Gauge (1.676 m) | No Real-World Safety Certification",
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color="#991B1B",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FEE2E2", edgecolor="#DC2626", lw=1.0),
    )

    def draw_box(x: float, y: float, w: float, h: float, title: str, body: str, fc: str, ec: str) -> None:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.2", facecolor=fc, edgecolor=ec, linewidth=1.6
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h - 2.1, title, ha="center", va="top", fontsize=9.5, fontweight="bold", color="#111827")
        ax.text(x + w / 2, y + h / 2 - 1.1, body, ha="center", va="center", fontsize=8.0, color="#1F2937")

    # Layer 1: Config + Blender
    draw_box(
        3, 40, 21, 13.5,
        "1. Declarative Config (M1)",
        "config/*.json\n250 m track | 1.676 m gauge\nCam: 35mm, 1.85m, -7.5 deg\nSpeeds: 30-200 km/h (30 FPS)\nmotion_blur = False",
        "#EEF2FF", "#4F46E5",
    )
    draw_box(
        28, 40, 23, 13.5,
        "2. Blender 5.3 Sim (M2-M4)",
        "Dual-Python (bpy / Python 3.13)\nA1 Carcass @ 50m | A2 Grain @ 100m\nA3 Crack @ 150m | A4 Curve @ 200m\n7 controlled speeds + 20 variants\nSHA-256 preserved scenes & MP4s",
        "#E0F2FE", "#0284C7",
    )
    draw_box(
        55, 40, 20, 13.5,
        "3. Synthetic Dataset (M5)",
        "Host Python 3.10\n6,102 PNG frames (0 dupes)\ntrain: var_00-13 (2,639)\nval: var_14-16 (558)\ntest: eval_* (2,347) + var_17-19",
        "#ECFDF5", "#059669",
    )
    draw_box(
        79, 40, 18, 13.5,
        "4. Frozen M6 Weights",
        "RTX 3050 6GB (CUDA)\nYOLOv8n (25 ep, nc=2)\nRailCrackUNet (512x256)\nthr=0.85 calibrated on val\nTrain/Val isolated only",
        "#FEF3C7", "#D97706",
    )

    # Layer 2: Hybrid Detection
    draw_box(
        6, 20, 27, 13.5,
        "5a. A1 & A2: YOLOv8n",
        "640px inference (CUDA ~18ms)\nEval: PR down to conf >= 0.001\nOperating point: conf >= 0.25\nA1 left-censored @ 50.0 m\nA2 early hits from 36.7 m",
        "#F0FDF4", "#16A34A",
    )
    draw_box(
        36.5, 20, 27, 13.5,
        "5b. A3: RailCrackUNet",
        "512x256 rail ROI (~24ms CUDA)\nmin_area >= 8 px filter\nVis-only & All-frame metrics\n0 spurious neg FPs (<138.8m)\n3 frames / 157 px @ 200 km/h",
        "#FEF2F2", "#DC2626",
    )
    draw_box(
        67, 20, 27, 13.5,
        "5c. A4: RailGeometryAnalyzer",
        "Pure CPU IPM + gauge-coupled (~49ms)\nPrimary bottleneck (51-54% CPU)\nPeak diff (mm) != MAE/RMSE (cm)\nPointwise max up to -3.89 cm\nDiscloses [188,212]m post-gate",
        "#FFF7ED", "#EA580C",
    )

    # Layer 3: Audit + M7 Evaluation + Dashboard
    draw_box(
        3.5, 3.8, 43.5, 11.2,
        "6. M6 Independent Audit & M7 Evaluation Suite",
        "Authoritative audit (SHA-256, PR>=0.001, A3/A4 unit split, latency)\nevaluation/detection_metrics.py, distance_analysis.py, speed_analysis.py\nMeasured E2E throughput: 98.50 ms/f = 10.15 FPS (Non-Real-Time)",
        "#F3E8FF", "#9333EA",
    )
    draw_box(
        52.0, 3.8, 44.5, 11.2,
        "7. Interactive Dashboard & Technical Reporting (M7)",
        "dashboard/app.py (Streamlit + Plotly, H.264 browser videos, zero bpy)\n9 dedicated audit-aligned sections | Separate eval_* vs var_17-19\nFINAL_TECHNICAL_REPORT.md + m7_evaluation/*.csv + screenshots",
        "#E0E7FF", "#4338CA",
    )

    # Arrows
    arrow_kw = dict(arrowstyle="-|>", color="#374151", lw=1.6, mutation_scale=13)
    for (x0, y0, x1, y1) in [
        (24, 46.7, 28, 46.7),
        (51, 46.7, 55, 46.7),
        (75, 46.7, 79, 46.7),
        (88, 40, 19.5, 33.8),
        (88, 40, 50.0, 33.8),
        (88, 40, 80.5, 33.8),
        (19.5, 20, 23, 15.3),
        (50.0, 20, 27, 15.3),
        (80.5, 20, 31, 15.3),
        (47.2, 9.4, 51.8, 9.4),
    ]:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=arrow_kw)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def generate_m7_charts(
    tables: dict[str, Any],
    distance_df: pd.DataFrame,
    latency_df: pd.DataFrame,
    io_summary: dict[str, Any],
    graphs_dir: Path,
) -> dict[str, str]:
    """Generate all audit-corrected Milestone 7 static comparative charts."""
    graphs_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    y_df = tables["yolo_summary"]
    y_ctrl = y_df[y_df["dataset_group"] == "controlled_evaluation"].sort_values("speed_kmh")
    y_var = y_df[y_df["dataset_group"] != "controlled_evaluation"]
    ypc_ctrl = tables["yolo_per_class"][tables["yolo_per_class"]["dataset_group"] == "controlled_evaluation"]
    a3_ctrl = tables["a3_summary"][tables["a3_summary"]["dataset_group"] == "controlled_evaluation"].sort_values("speed_kmh")
    a4_ctrl = tables["a4_summary"][tables["a4_summary"]["dataset_group"] == "controlled_evaluation"].sort_values("speed_kmh")
    a4_prof = tables["a4_error_profiles"]
    lat_ctrl = latency_df[latency_df["dataset_group"] == "controlled_evaluation"].sort_values("speed_kmh")
    speeds = y_ctrl["speed_kmh"].tolist()

    # 0. Architecture diagram
    p_arch = graphs_dir / "m7_system_architecture.png"
    generate_system_architecture_diagram(p_arch)
    paths["system_architecture"] = str(p_arch)

    # 1. Speed vs YOLO Precision / Recall / F1 (@ conf=0.25 + per-class recall)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].plot(speeds, y_ctrl["precision_conf025"], "o-", lw=2, label="Precision (conf>=0.25)")
    axes[0].plot(speeds, y_ctrl["recall_conf025"], "s-", lw=2, label="Recall (conf>=0.25)")
    axes[0].plot(speeds, y_ctrl["f1_conf025"], "^-", lw=2, label="F1-score (conf>=0.25)")
    axes[0].annotate(
        "200 km/h R=0.9535\n(2 exit-edge clips / 43 GT)",
        xy=(200, 0.9535),
        xytext=(152, 0.905),
        arrowprops=dict(arrowstyle="->", color="#B91C1C"),
        fontsize=8.5,
        color="#991B1B",
    )
    axes[0].set_title("Overall YOLO Precision, Recall & F1 vs Speed (conf>=0.25)")
    axes[0].set_xlabel("Train Speed (km/h)")
    axes[0].set_ylabel("Score")
    axes[0].set_xticks(speeds)
    axes[0].set_ylim(0.88, 1.02)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower left")

    a1_pc = ypc_ctrl[ypc_ctrl["class_key"] == "0_animal_carcass"].sort_values("speed_kmh")
    a2_pc = ypc_ctrl[ypc_ctrl["class_key"] == "1_grain_spillage"].sort_values("speed_kmh")
    axes[1].plot(a1_pc["speed_kmh"], a1_pc["recall_conf025"], "o-", label="A1 animal_carcass Recall")
    axes[1].plot(a2_pc["speed_kmh"], a2_pc["recall_conf025"], "s-", label="A2 grain_spillage Recall")
    axes[1].plot(a2_pc["speed_kmh"], a2_pc["precision_conf025"], "d--", label="A2 grain_spillage Precision")
    axes[1].set_title("Per-Class YOLO Operating Point (conf>=0.25)")
    axes[1].set_xlabel("Train Speed (km/h)")
    axes[1].set_ylabel("Score")
    axes[1].set_xticks(speeds)
    axes[1].set_ylim(0.88, 1.02)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="lower left")

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _apply_footer(fig)
    p1 = graphs_dir / "m7_speed_vs_yolo_prf1.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    paths["speed_vs_yolo_prf1"] = str(p1)

    # 2. Speed vs Unclipped mAP50 (conf>=0.001) vs Truncated (conf>=0.25)
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.plot(speeds, y_ctrl["mAP50_conf001"], "o-", lw=2.2, label="mAP50 (Authoritative conf >= 0.001)")
    ax.plot(speeds, y_ctrl["mAP50_trunc_conf025"], "x--", lw=1.6, label="mAP50 (M6 conf >= 0.25 truncated)")
    ax.plot(speeds, y_ctrl["mAP50_95_conf001"], "s-", lw=2.0, label="mAP50-95 (conf >= 0.001)")
    ax.plot(a1_pc["speed_kmh"], a1_pc["ap50_conf001"], "^:", alpha=0.8, label="A1 Carcass AP50 (conf >= 0.001)")
    ax.plot(a2_pc["speed_kmh"], a2_pc["ap50_conf001"], "v:", alpha=0.8, label="A2 Spillage AP50 (conf >= 0.001)")
    ax.set_title("Controlled Speed vs. Unclipped mAP50 & mAP50-95 (PR sweep conf >= 0.001)")
    ax.set_xlabel("Train Speed (km/h)")
    ax.set_ylabel("AP / mAP")
    ax.set_xticks(speeds)
    ax.set_ylim(0.75, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _apply_footer(fig)
    p2 = graphs_dir / "m7_speed_vs_map.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    paths["speed_vs_map"] = str(p2)

    # 3. Speed vs A3 Crack Segmentation (Visible-only vs All-frames + pixel/frame budget)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].plot(speeds, a3_ctrl["vis_only_dice"], "o-", lw=2, label="Dice (GT Visible Frames Only)")
    axes[0].plot(speeds, a3_ctrl["all_frames_dice"], "s--", lw=2, label="Dice (All Frames incl. Pre-Vis 10-11m)")
    axes[0].plot(speeds, a3_ctrl["all_frames_iou"], "^-", lw=1.8, label="IoU (All Frames)")
    axes[0].set_title("A3 RailCrackUNet: Visible-Only vs All-Frame Dice & IoU")
    axes[0].set_xlabel("Train Speed (km/h)")
    axes[0].set_ylabel("Score")
    axes[0].set_xticks(speeds)
    axes[0].set_ylim(0.55, 0.90)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].bar([str(s) for s in speeds], a3_ctrl["gt_total_fg_pixels"], color="#DC2626", alpha=0.75, label="Total GT Crack Pixels in Run")
    ax2_r = axes[1].twinx()
    ax2_r.plot([str(s) for s in speeds], a3_ctrl["gt_visible_frames"], "ko-", lw=2, label="GT Visible Crack Frames")
    axes[1].set_title("A3 Sample Scarcity at High Speed (thr=0.85, 0 FP <138.8 m)")
    axes[1].set_xlabel("Train Speed (km/h)")
    axes[1].set_ylabel("Total GT Crack Foreground Pixels", color="#991B1B")
    ax2_r.set_ylabel("Visible Frames Count", color="#111827")
    ax2_r.set_ylim(0, 26)
    for idx, (px, fr) in enumerate(zip(a3_ctrl["gt_total_fg_pixels"], a3_ctrl["gt_visible_frames"])):
        if px >= 350:
            axes[1].text(idx, px * 0.45, f"{px} px\n({fr}f)", ha="center", va="center", fontsize=8.0, color="white", fontweight="bold")
        else:
            axes[1].text(idx, px + 55, f"{px} px\n({fr}f)", ha="center", va="bottom", fontsize=7.8, color="#7F1D1D", fontweight="bold")
    axes[1].set_ylim(0, max(a3_ctrl["gt_total_fg_pixels"]) * 1.25)
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _apply_footer(fig)
    p3 = graphs_dir / "m7_speed_vs_a3_segmentation.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    paths["speed_vs_a3_segmentation"] = str(p3)

    # 4. Speed vs A4 Geometry Metrics (Strict Unit Separation: cm profile metrics vs mm peak diff)
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.0))
    axes[0].plot(speeds, a4_ctrl["profile_mae_cm"], "o-", lw=2, label="Profile MAE (cm) — [196, 204] m")
    axes[0].plot(speeds, a4_ctrl["profile_rmse_cm"], "s-", lw=2, label="Profile RMSE (cm) — [196, 204] m")
    axes[0].plot(speeds, a4_ctrl["max_pointwise_error_cm"], "^--", lw=1.8, color="#B91C1C", label="Pointwise Max |Error| (cm)")
    axes[0].plot(speeds, a4_ctrl["outside_196_204_mae_cm"], "d:", color="#047857", label="Outside [196,204]m Straight MAE (cm)")
    axes[0].set_title("A4 Track Profile Errors (cm) — MAE, RMSE & Worst Station")
    axes[0].set_xlabel("Train Speed (km/h)")
    axes[0].set_ylabel("Error (cm)")
    axes[0].set_xticks(speeds)
    axes[0].set_ylim(0, 4.4)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=8.0)

    axes[1].bar([str(s) for s in speeds], a4_ctrl["peak_amplitude_diff_mm"], color="#EA580C", alpha=0.82)
    axes[1].set_title("A4 Scalar Peak-Amplitude Diff ONLY (mm)\n(Ignores -0.25 m Peak Y Shift & Profile Slope Error)")
    axes[1].set_xlabel("Train Speed (km/h)")
    axes[1].set_ylabel("Peak Amplitude Diff ||est|_max - |gt|_max|| (mm)")
    for idx, val in enumerate(a4_ctrl["peak_amplitude_diff_mm"]):
        axes[1].text(idx, val + 0.07, f"{val:.2f} mm", ha="center", va="bottom", fontsize=8.5)
    axes[1].set_ylim(0, max(a4_ctrl["peak_amplitude_diff_mm"]) * 1.25)
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _apply_footer(fig)
    p4 = graphs_dir / "m7_speed_vs_a4_geometry_metrics.png"
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    paths["speed_vs_a4_geometry_metrics"] = str(p4)

    # 5. A4 GT vs Reconstructed Profiles + Signed Pointwise Error Curve
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.0))
    for spd_p, style in ((30, "o--"), (120, "s--"), (200, "^--")):
        sub = a4_prof[a4_prof["video_id"] == f"eval_{spd_p}kmh"].sort_values("y_m")
        if spd_p == 30:
            axes[0].plot(sub["y_m"], sub["gt_cm"], "k-", lw=2.3, label="Ground Truth (GT max 8.50 cm)")
        axes[0].plot(sub["y_m"], sub["est_cm"], style, markersize=3.5, label=f"Est @ {spd_p} km/h")
        axes[1].plot(sub["y_m"], sub["signed_err_cm"], style, markersize=3.5, label=f"Signed Err @ {spd_p} km/h")

    axes[0].set_title("A4 Reconstructed Lateral Profile vs Ground Truth [196-204 m]")
    axes[0].set_xlabel("Track Position Y (m)")
    axes[0].set_ylabel("Lateral Displacement (cm)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8.5)

    axes[1].axhline(0.0, color="#111827", lw=1.0)
    axes[1].axvspan(200.75, 202.25, color="#FEE2E2", alpha=0.5, label="Receding slope negative bias zone")
    axes[1].set_title("Pointwise Signed Error (est - GT cm)\n([188, 212] m Evaluation Gate Disclosed)")
    axes[1].set_xlabel("Track Position Y (m)")
    axes[1].set_ylabel("Signed Error (cm)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8.0)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _apply_footer(fig)
    p5 = graphs_dir / "m7_a4_reconstructed_profiles.png"
    fig.savefig(p5, dpi=150)
    plt.close(fig)
    paths["a4_reconstructed_profiles"] = str(p5)

    # 6. Speed vs Detection Distance (A1-A4) & Anomaly Timeline
    d_ctrl = distance_df[distance_df["dataset_group"] == "controlled_evaluation"]
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.9))
    for aid, lbl, mk in (
        ("A1", "A1 Carcass @ 50m (LEFT-CENSORED @ track start)", "o-"),
        ("A2", "A2 Spillage @ 100m (YOLOv8n)", "s-"),
        ("A4", "A4 Misalignment @ 200m (Frame IPM >=3cm)", "d-"),
        ("A3", "A3 Crack @ 150m (UNet ROI <=10m)", "^-"),
    ):
        sub = d_ctrl[d_ctrl["anomaly_id"] == aid].sort_values("speed_kmh")
        axes[0].plot(sub["speed_kmh"], sub["first_detect_distance_m"], mk, lw=2, label=lbl)
    axes[0].set_title("Earliest Reliable Event Detection Distance vs Speed")
    axes[0].set_xlabel("Train Speed (km/h)")
    axes[0].set_ylabel("Camera-to-Anomaly Distance (m)")
    axes[0].set_xticks(speeds)
    axes[0].set_ylim(0, 56)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc=(0.03, 0.25), fontsize=8.0)

    for aid, col, y_off in (("A1", "#16A34A", 0), ("A2", "#CA8A04", 1), ("A3", "#DC2626", 2), ("A4", "#EA580C", 3)):
        sub = d_ctrl[d_ctrl["anomaly_id"] == aid].sort_values("speed_kmh")
        axes[1].plot(sub["speed_kmh"], sub["visible_window_duration_s"], "o-", color=col, lw=2, label=f"{aid} {sub.iloc[0]['anomaly_name']}")
    axes[1].set_title("Anomaly Temporal Visibility Window vs Speed (30 FPS Preview)")
    axes[1].set_xlabel("Train Speed (km/h)")
    axes[1].set_ylabel("Visible Window Duration (s)")
    axes[1].set_xticks(speeds)
    axes[1].set_yscale("log")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=8.2)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _apply_footer(fig)
    p6 = graphs_dir / "m7_speed_vs_detection_distance_timeline.png"
    fig.savefig(p6, dpi=150)
    plt.close(fig)
    paths["speed_vs_detection_distance_timeline"] = str(p6)

    # 7. Pipeline Latency Breakdown & Real-Time FPS Gap
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9))
    x_lbls = [f"{s}" for s in lat_ctrl["speed_kmh"]]
    yolo_m = lat_ctrl["yolo_ms_mean"].to_numpy()
    unet_m = lat_ctrl["crack_unet_ms_mean"].to_numpy()
    geom_m = lat_ctrl["geometry_ms_mean"].to_numpy()
    axes[0].bar(x_lbls, yolo_m, label="YOLO GPU (17.2-20.7 ms)", color="#16A34A")
    axes[0].bar(x_lbls, unet_m, bottom=yolo_m, label="RailCrackUNet GPU (22.7-24.1 ms)", color="#DC2626")
    axes[0].bar(x_lbls, geom_m, bottom=yolo_m + unet_m, label="Geometry IPM CPU (42.9-53.2 ms, Bottleneck)", color="#EA580C")
    axes[0].axhline(33.333, color="#2563EB", ls="--", lw=1.8, label="30 FPS Real-Time Budget (33.33 ms)")
    axes[0].axhline(16.667, color="#7C3AED", ls=":", lw=1.8, label="60 FPS Real-Time Budget (16.67 ms)")
    axes[0].set_title("Compute-Only Pipeline Latency Stack vs Real-Time Frame Budgets")
    axes[0].set_xlabel("Controlled Train Speed (km/h)")
    axes[0].set_ylabel("Mean Latency (ms / frame)")
    axes[0].set_ylim(0, 118)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=7.6)

    stage_labels = ["Compute Only\n(Controlled Mean)", "E2E Measured\n(60kmh PNG+Enc)", "30 FPS\nTarget", "60 FPS\nTarget"]
    fps_vals = [float(lat_ctrl["compute_only_fps"].mean()), io_summary["end_to_end_fps_measured"], 30.0, 60.0]
    cols = ["#EA580C", "#991B1B", "#2563EB", "#7C3AED"]
    bars = axes[1].bar(stage_labels, fps_vals, color=cols, alpha=0.88)
    for b, fv in zip(bars, fps_vals):
        axes[1].text(b.get_x() + b.get_width() / 2, fv + 1.2, f"{fv:.2f} FPS", ha="center", va="bottom", fontweight="bold", fontsize=8.8)
    axes[1].set_title("Throughput Reality Check: FAILS 30 FPS & 60 FPS Targets")
    axes[1].set_ylabel("Throughput (Frames Per Second)")
    axes[1].set_ylim(0, 68)
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _apply_footer(fig)
    p7 = graphs_dir / "m7_latency_vs_fps_budgets.png"
    fig.savefig(p7, dpi=150)
    plt.close(fig)
    paths["latency_vs_fps_budgets"] = str(p7)

    # 8. Separate Independent Test Variants (var_17, var_18, var_19)
    a3_var = tables["a3_summary"][tables["a3_summary"]["dataset_group"] != "controlled_evaluation"]
    a4_var = tables["a4_summary"][tables["a4_summary"]["dataset_group"] != "controlled_evaluation"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    v_lbls = [f"{r['video_id']}\n({r['speed_kmh']}kmh)" for _, r in y_var.iterrows()]
    x_i = np.arange(len(v_lbls))
    w = 0.25
    axes[0].bar(x_i - w, y_var["f1_conf025"], width=w, label="YOLO F1 (conf>=0.25)")
    axes[0].bar(x_i, y_var["mAP50_conf001"], width=w, label="YOLO mAP50 (conf>=0.001)")
    axes[0].bar(x_i + w, a3_var["all_frames_dice"], width=w, label="A3 All-Frame Dice")
    axes[0].set_title("Held-Out Independent Variants (var_17-var_19) Detection & Seg")
    axes[0].set_xticks(x_i)
    axes[0].set_xticklabels(v_lbls)
    axes[0].set_ylim(0.7, 1.05)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend(fontsize=8.2)

    axes[1].bar(x_i - 0.18, a4_var["profile_mae_cm"], width=0.36, label="A4 Profile MAE (cm)")
    axes[1].bar(x_i + 0.18, a4_var["profile_rmse_cm"], width=0.36, label="A4 Profile RMSE (cm)")
    for idx, (fp_cnt, p_max, top_h) in enumerate(
        zip(a4_var["frame_fp"], a4_var["max_pointwise_error_cm"], a4_var["profile_rmse_cm"])
    ):
        axes[1].text(idx, float(top_h) + 0.06, f"pt max {p_max:.2f}cm\nframe FP={fp_cnt}", ha="center", va="bottom", fontsize=7.8)
    axes[1].set_title("Held-Out Variants A4 Profile Error & Frame FPs (Separate from Eval)")
    axes[1].set_xticks(x_i)
    axes[1].set_xticklabels(v_lbls)
    axes[1].set_ylabel("Error (cm)")
    axes[1].set_ylim(0, 1.95)
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend(fontsize=8.2)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _apply_footer(fig)
    p8 = graphs_dir / "m7_independent_test_variants_summary.png"
    fig.savefig(p8, dpi=150)
    plt.close(fig)
    paths["independent_test_variants_summary"] = str(p8)

    return paths


def run_m7_evaluation_generation(project_root: Path | None = None) -> dict[str, Any]:
    """Run full Milestone 7 audit-corrected report & chart generation."""
    root = project_root or PROJECT_ROOT
    tables = build_all_metric_tables(root)
    distance_df = build_anomaly_distance_timeline(tables)
    latency_df, io_summary = build_latency_table(tables)
    synthesis = build_speed_synthesis(tables, distance_df, latency_df, io_summary)

    m7_dir = root / "results" / "metrics" / "m7_evaluation"
    m7_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir = root / "results" / "graphs"

    # Write clean CSV deliverables
    csv_outputs = {
        "yolo_summary_csv": m7_dir / "m7_yolo_summary.csv",
        "yolo_per_class_csv": m7_dir / "m7_yolo_per_class.csv",
        "yolo_fp_forensics_csv": m7_dir / "m7_yolo_fp_forensics.csv",
        "yolo_fn_forensics_csv": m7_dir / "m7_yolo_fn_forensics.csv",
        "a3_summary_csv": m7_dir / "m7_a3_segmentation_summary.csv",
        "a3_val_threshold_sweep_csv": m7_dir / "m7_a3_val_threshold_sweep.csv",
        "a4_summary_csv": m7_dir / "m7_a4_geometry_summary.csv",
        "a4_error_profiles_csv": m7_dir / "m7_a4_geometry_error_profiles.csv",
        "anomaly_distances_csv": m7_dir / "m7_anomaly_events_and_distances.csv",
        "latency_breakdown_csv": m7_dir / "m7_latency_breakdown.csv",
    }

    tables["yolo_summary"].to_csv(csv_outputs["yolo_summary_csv"], index=False)
    tables["yolo_per_class"].to_csv(csv_outputs["yolo_per_class_csv"], index=False)
    tables["yolo_fp_forensics"].to_csv(csv_outputs["yolo_fp_forensics_csv"], index=False)
    tables["yolo_fn_forensics"].to_csv(csv_outputs["yolo_fn_forensics_csv"], index=False)
    tables["a3_summary"].to_csv(csv_outputs["a3_summary_csv"], index=False)
    tables["a3_val_threshold_sweep"].to_csv(csv_outputs["a3_val_threshold_sweep_csv"], index=False)
    tables["a4_summary"].to_csv(csv_outputs["a4_summary_csv"], index=False)
    tables["a4_error_profiles"].to_csv(csv_outputs["a4_error_profiles_csv"], index=False)
    distance_df.to_csv(csv_outputs["anomaly_distances_csv"], index=False)
    latency_df.to_csv(csv_outputs["latency_breakdown_csv"], index=False)

    graph_paths = generate_m7_charts(tables, distance_df, latency_df, io_summary, graphs_dir)

    comprehensive = {
        "dataset_type": "SYNTHETIC",
        "milestone": "M7_EVALUATION_AND_REPORTING",
        "authoritative_source": "results/metrics/m6_audit/m6_independent_audit.json",
        "synthesis": synthesis,
        "csv_outputs": {k: str(v) for k, v in csv_outputs.items()},
        "graphs": graph_paths,
    }
    comp_path = m7_dir / "m7_comprehensive_evaluation.json"
    comp_path.write_text(json.dumps(comprehensive, indent=2), encoding="utf-8")
    comprehensive["comprehensive_json"] = str(comp_path)
    return comprehensive


if __name__ == "__main__":
    res = run_m7_evaluation_generation()
    print("[M7] Evaluation CSVs & charts generated:")
    print(json.dumps({"comprehensive_json": res["comprehensive_json"], "graphs": res["graphs"]}, indent=2))
