"""Milestone 6 model training pipeline (YOLO baseline + RailCrackUNet).

Strictly trains and validates ONLY on independent scene variants (train / val).
Controlled evaluation speed runs are excluded and audited before training begins.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure Ultralytics & PyTorch store configs, weights, and caches strictly on D:
RESULTS_DIR = PROJECT_ROOT / "results"
ULTRALYTICS_CFG_DIR = RESULTS_DIR / "_ultralytics_cfg"
TORCH_CACHE_DIR = RESULTS_DIR / "_torch_cache"
TMP_DIR = RESULTS_DIR / "_tmp"
for _d in (ULTRALYTICS_CFG_DIR, TORCH_CACHE_DIR, TMP_DIR):
    _d.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CFG_DIR))
os.environ.setdefault("TORCH_HOME", str(TORCH_CACHE_DIR))
os.environ.setdefault("TEMP", str(TMP_DIR))
os.environ.setdefault("TMP", str(TMP_DIR))

from detection.crack_segmentation import train_crack_unet  # noqa: E402
from detection.dataset_audit import run_dataset_audit  # noqa: E402


def _plot_yolo_curves(results_csv: Path, out_png: Path) -> None:
    if not results_csv.is_file():
        return
    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]
    if df.empty or "epoch" not in df.columns:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    epochs = df["epoch"]

    for col, lbl in (
        ("train/box_loss", "Train Box"),
        ("val/box_loss", "Val Box"),
        ("train/cls_loss", "Train Cls"),
        ("val/cls_loss", "Val Cls"),
    ):
        if col in df.columns:
            axes[0].plot(epochs, df[col], label=lbl)
    axes[0].set_title("YOLO Box & Cls Loss (SYNTHETIC)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    for col, lbl in (
        ("metrics/precision(B)", "Precision"),
        ("metrics/recall(B)", "Recall"),
    ):
        if col in df.columns:
            axes[1].plot(epochs, df[col], label=lbl)
    axes[1].set_title("Validation Precision & Recall")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    for col, lbl in (
        ("metrics/mAP50(B)", "mAP50"),
        ("metrics/mAP50-95(B)", "mAP50-95"),
    ):
        if col in df.columns:
            axes[2].plot(epochs, df[col], label=lbl)
    axes[2].set_title("Validation mAP50 / mAP50-95")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _plot_unet_curves(history: list[dict[str, float]], out_png: Path) -> None:
    if not history:
        return
    ep = [h["epoch"] for h in history]
    tr_l = [h["train_loss"] for h in history]
    va_l = [h["val_loss"] for h in history]
    va_d = [h["val_dice_05"] for h in history]
    va_i = [h["val_iou_05"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].plot(ep, tr_l, label="Train Loss")
    axes[0].plot(ep, va_l, label="Val Loss")
    axes[0].set_title("RailCrackUNet Loss (SYNTHETIC)")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(ep, va_d, label="Val Dice @0.5")
    axes[1].plot(ep, va_i, label="Val IoU @0.5")
    axes[1].set_title("Validation Segmentation Score")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def train_yolo_baseline(
    data_yaml: Path,
    model_weights: str = "yolov8n.pt",
    epochs: int = 30,
    imgsz: int = 640,
    batch: int = 16,
    patience: int = 8,
    seed: int = 42,
) -> dict[str, Any]:
    """Train baseline YOLO detector on train/val splits only."""
    from ultralytics import YOLO

    weights_cache_dir = RESULTS_DIR / "models" / "pretrained_base"
    weights_cache_dir.mkdir(parents=True, exist_ok=True)
    local_base_weight = weights_cache_dir / Path(model_weights).name

    # Download or load pretrained generic weights inside D:\...\results\models\pretrained_base
    prev_cwd = Path.cwd()
    os.chdir(str(weights_cache_dir))
    try:
        model = YOLO(str(local_base_weight if local_base_weight.is_file() else model_weights))
    finally:
        os.chdir(str(prev_cwd))

    yolo_project_dir = RESULTS_DIR / "models" / "yolo"
    run_name = "baseline_v1"

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)

    t0 = time.perf_counter()
    train_res = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=0 if torch.cuda.is_available() else "cpu",
        patience=patience,
        seed=seed,
        deterministic=True,
        project=str(yolo_project_dir),
        name=run_name,
        exist_ok=True,
        workers=2,
        plots=True,
        verbose=True,
    )
    total_duration_s = round(time.perf_counter() - t0, 2)

    peak_vram_mb = (
        round(torch.cuda.max_memory_allocated(0) / (1024**2), 2)
        if torch.cuda.is_available()
        else 0.0
    )

    run_dir = yolo_project_dir / run_name
    results_csv = run_dir / "results.csv"
    _plot_yolo_curves(results_csv, run_dir / "loss_and_metrics_curves.png")

    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"

    val_summary: dict[str, Any] = {}
    if results_csv.is_file():
        df = pd.read_csv(results_csv)
        df.columns = [c.strip() for c in df.columns]
        if not df.empty:
            best_idx = int(df["metrics/mAP50-95(B)"].idxmax()) if "metrics/mAP50-95(B)" in df.columns else len(df) - 1
            best_row = df.iloc[best_idx].to_dict()
            last_row = df.iloc[-1].to_dict()
            epochs_completed = int(len(df))
            val_summary = {
                "epochs_completed": epochs_completed,
                "mean_epoch_duration_s": round(total_duration_s / max(1, epochs_completed), 2),
                "best_epoch": int(best_row.get("epoch", best_idx + 1)),
                "best_metrics": {k: round(float(v), 5) for k, v in best_row.items() if isinstance(v, (int, float))},
                "final_metrics": {k: round(float(v), 5) for k, v in last_row.items() if isinstance(v, (int, float))},
            }

    return {
        "model_architecture": model_weights,
        "pretrained_disclaimer": (
            f"Initialized from generic COCO-pretrained {model_weights}; "
            "fine-tuned exclusively on synthetic train/val railway variants (never controlled eval)."
        ),
        "training_config": {
            "data_yaml": str(data_yaml),
            "epochs_max": epochs,
            "patience": patience,
            "imgsz": imgsz,
            "batch": batch,
            "seed": seed,
            "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        },
        "total_training_duration_s": total_duration_s,
        "peak_gpu_memory_mb": peak_vram_mb,
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_pt),
        "last_checkpoint": str(last_pt),
        "results_csv": str(results_csv),
        "val_summary": val_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train Milestone 6 YOLO object detector and RailCrackUNet on train/val only."
    )
    parser.add_argument("--yolo-weights", default="yolov8n.pt", help="Base Ultralytics weights.")
    parser.add_argument("--yolo-epochs", type=int, default=30, help="Max YOLO epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO input size.")
    parser.add_argument("--batch", type=int, default=16, help="YOLO batch size.")
    parser.add_argument("--patience", type=int, default=8, help="YOLO early stopping patience.")
    parser.add_argument("--unet-epochs", type=int, default=35, help="RailCrackUNet epochs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--skip-yolo", action="store_true", help="Skip YOLO training.")
    parser.add_argument("--skip-unet", action="store_true", help="Skip U-Net training.")
    args = parser.parse_args()

    # Phase 1 safety gate: verify dataset audit & controlled eval exclusion before any training
    audit = run_dataset_audit()
    if not audit["audit_passed"]:
        print("ERROR: Dataset audit failed; aborting training.", file=sys.stderr)
        return 2
    if not audit["controlled_eval_excluded_from_train_val"]:
        print("ERROR: Controlled evaluation videos found in train/val!", file=sys.stderr)
        return 2

    dataset_root = PROJECT_ROOT / "dataset" / "synthetic"
    data_yaml = dataset_root / "configs" / "data.yaml"

    summary: dict[str, Any] = {
        "dataset_type": "SYNTHETIC",
        "milestone": "M6_TRAINING_SUMMARY",
        "audit_verified": True,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "pytorch_version": torch.__version__,
    }

    if not args.skip_yolo:
        summary["yolo"] = train_yolo_baseline(
            data_yaml=data_yaml,
            model_weights=args.yolo_weights,
            epochs=args.yolo_epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=args.patience,
            seed=args.seed,
        )

    if not args.skip_unet:
        unet_dir = RESULTS_DIR / "models" / "crack_unet"
        unet_res = train_crack_unet(
            dataset_root=dataset_root,
            output_dir=unet_dir,
            epochs=args.unet_epochs,
            batch_size=8,
            lr=1e-3,
            seed=args.seed,
        )
        if unet_res.get("trained"):
            _plot_unet_curves(unet_res["history"], unet_dir / "crack_unet_curves.png")
        summary["crack_unet"] = unet_res

    out_summary = RESULTS_DIR / "metrics" / "m6_training_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[M6] Training summary saved to: {out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

