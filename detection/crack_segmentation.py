"""Independent PyTorch U-Net rail crack segmentation model for A3 (Phase 5).

Operates on a native-resolution track-corridor ROI crop (x=224..736, y=284..540 -> 512x256)
to counteract extreme foreground/background pixel imbalance (median ~50 px in 960x540 frames)
while preserving full pixel detail on both rails.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# Native-resolution ROI crop covering both left & right rails in 4.2m - 13m range
CRACK_ROI_X0 = 224
CRACK_ROI_Y0 = 284
CRACK_ROI_W = 512
CRACK_ROI_H = 256


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RailCrackUNet(nn.Module):
    """Lightweight 4-level residual-skip U-Net (~485K params) for tiny crack segmentation."""

    def __init__(self, base_ch: int = 16) -> None:
        super().__init__()
        self.enc1 = ConvBlock(3, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 8)

        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)

        self.head = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


class FocalTverskyBCELoss(nn.Module):
    """Loss tailored for extreme small-object foreground/background imbalance."""

    def __init__(self, pos_weight: float = 25.0, alpha: float = 0.3, beta: float = 0.7) -> None:
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.alpha = float(alpha)
        self.beta = float(beta)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pw = torch.tensor([self.pos_weight], device=logits.device, dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw, reduction="none")
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_bce = ((1.0 - pt) ** 1.5 * bce).mean()

        # Soft Tversky (penalizes false negatives higher with beta=0.7)
        tp = (probs * targets).sum(dim=(1, 2, 3))
        fp = (probs * (1.0 - targets)).sum(dim=(1, 2, 3))
        fn = ((1.0 - probs) * targets).sum(dim=(1, 2, 3))
        tversky = (tp + 1.0) / (tp + self.alpha * fp + self.beta * fn + 1.0)
        tversky_loss = (1.0 - tversky).mean()

        return 0.45 * focal_bce + 0.55 * tversky_loss


def crop_rail_roi(img: np.ndarray) -> np.ndarray:
    """Extract the 512x256 native-resolution rail-corridor ROI."""
    return img[
        CRACK_ROI_Y0 : CRACK_ROI_Y0 + CRACK_ROI_H,
        CRACK_ROI_X0 : CRACK_ROI_X0 + CRACK_ROI_W,
    ]


def uncrop_roi_mask(roi_mask: np.ndarray, full_w: int = 960, full_h: int = 540) -> np.ndarray:
    """Embed a 512x256 ROI mask back into full frame coordinates."""
    full = np.zeros((full_h, full_w), dtype=roi_mask.dtype)
    full[
        CRACK_ROI_Y0 : CRACK_ROI_Y0 + CRACK_ROI_H,
        CRACK_ROI_X0 : CRACK_ROI_X0 + CRACK_ROI_W,
    ] = roi_mask
    return full


class CrackROIDataset(Dataset):
    """Balanced positive/hard-negative ROI dataset for RailCrackUNet."""

    def __init__(
        self,
        samples: list[tuple[Path, Path | None]],
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        self.samples = samples
        self.augment = augment
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.samples[idx]
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        roi_img = crop_rail_roi(img_rgb)

        if mask_path is not None and mask_path.is_file():
            full_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            roi_mask = (crop_rail_roi(full_mask) > 127).astype(np.float32)
        else:
            roi_mask = np.zeros((CRACK_ROI_H, CRACK_ROI_W), dtype=np.float32)

        if self.augment:
            if self.rng.random() < 0.5:
                roi_img = np.ascontiguousarray(roi_img[:, ::-1, :])
                roi_mask = np.ascontiguousarray(roi_mask[:, ::-1])
            alpha = self.rng.uniform(0.88, 1.12)
            beta = self.rng.uniform(-12.0, 12.0)
            roi_img = np.clip(roi_img.astype(np.float32) * alpha + beta, 0.0, 255.0).astype(
                np.uint8
            )

        img_t = torch.from_numpy(roi_img.transpose(2, 0, 1)).float() / 255.0
        mask_t = torch.from_numpy(roi_mask[None, ...]).float()
        return img_t, mask_t


def compute_segmentation_metrics(
    pred_masks: list[np.ndarray],
    gt_masks: list[np.ndarray],
) -> dict[str, float]:
    """Compute global pixel Dice, IoU, precision, and recall over a list of binary masks."""
    tp = 0
    fp = 0
    fn = 0
    for pred, gt in zip(pred_masks, gt_masks):
        p = pred > 127
        g = gt > 127
        tp += int(np.logical_and(p, g).sum())
        fp += int(np.logical_and(p, np.logical_not(g)).sum())
        fn += int(np.logical_and(np.logical_not(p), g).sum())

    dice = (2.0 * tp) / max(1, 2 * tp + fp + fn)
    iou = tp / max(1, tp + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return {
        "tp_pixels": tp,
        "fp_pixels": fp,
        "fn_pixels": fn,
        "dice": round(float(dice), 4),
        "iou": round(float(iou), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
    }


def _collect_split_samples(
    dataset_root: Path, split: str, neg_ratio: float = 1.0, seed: int = 42
) -> tuple[list[tuple[Path, Path | None]], int, int]:
    """Collect all positive A3 masks plus hard-negative frames strictly from split."""
    img_dir = dataset_root / "images" / split
    mask_dir = dataset_root / "segmentation" / split / "a3_masks"
    all_imgs = sorted(img_dir.glob("*.png"))
    pos_samples: list[tuple[Path, Path | None]] = []
    neg_candidates: list[tuple[Path, Path | None]] = []

    for img_p in all_imgs:
        m_p = mask_dir / f"{img_p.stem}_a3_mask.png"
        if m_p.is_file():
            # Verify mask has non-empty crack inside ROI
            m_arr = cv2.imread(str(m_p), cv2.IMREAD_GRAYSCALE)
            if m_arr is not None and int((crop_rail_roi(m_arr) > 127).sum()) > 0:
                pos_samples.append((img_p, m_p))
        else:
            neg_candidates.append((img_p, None))

    rng = random.Random(seed)
    n_neg = min(len(neg_candidates), int(round(len(pos_samples) * neg_ratio)))
    neg_samples = rng.sample(neg_candidates, n_neg) if n_neg > 0 else []
    combined = pos_samples + neg_samples
    rng.shuffle(combined)
    return combined, len(pos_samples), len(neg_samples)


def train_crack_unet(
    dataset_root: Path,
    output_dir: Path,
    epochs: int = 35,
    batch_size: int = 8,
    lr: float = 1e-3,
    seed: int = 42,
) -> dict[str, Any]:
    """Train RailCrackUNet on train/val A3 masks (never touching test) and select threshold on val."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_samples, n_tr_pos, n_tr_neg = _collect_split_samples(
        dataset_root, "train", neg_ratio=1.0, seed=seed
    )
    val_samples, n_va_pos, n_va_neg = _collect_split_samples(
        dataset_root, "val", neg_ratio=1.0, seed=seed + 1
    )

    if n_tr_pos < 20:
        return {
            "trained": False,
            "reason": f"Insufficient positive A3 training masks ({n_tr_pos} < 20).",
        }

    # Oversample positive+hard-neg train set 2x for smoother gradient steps across 64 positives
    train_ds = CrackROIDataset(train_samples * 2, augment=True, seed=seed)
    val_ds = CrackROIDataset(val_samples, augment=False, seed=seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = RailCrackUNet(base_ch=16).to(device)
    criterion = FocalTverskyBCELoss(pos_weight=25.0, alpha=0.3, beta=0.7)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = output_dir / "best_crack_unet.pt"
    last_ckpt = output_dir / "last_crack_unet.pt"

    best_val_dice = -1.0
    best_epoch = 0
    history: list[dict[str, float]] = []
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss_sum = 0.0
        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            tr_loss_sum += float(loss.item()) * imgs.size(0)
        scheduler.step()
        tr_loss = tr_loss_sum / max(1, len(train_ds))

        # Validation evaluation at 0.5 threshold
        model.eval()
        val_loss_sum = 0.0
        val_preds: list[np.ndarray] = []
        val_gts: list[np.ndarray] = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)
                logits = model(imgs)
                loss = criterion(logits, masks)
                val_loss_sum += float(loss.item()) * imgs.size(0)
                probs = torch.sigmoid(logits).cpu().numpy()[:, 0]
                gts = masks.cpu().numpy()[:, 0]
                for p, g in zip(probs, gts):
                    val_preds.append((p >= 0.5).astype(np.uint8) * 255)
                    val_gts.append((g > 0.5).astype(np.uint8) * 255)

        val_loss = val_loss_sum / max(1, len(val_ds))
        val_metrics = compute_segmentation_metrics(val_preds, val_gts)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(tr_loss, 5),
                "val_loss": round(val_loss, 5),
                "val_dice_05": val_metrics["dice"],
                "val_iou_05": val_metrics["iou"],
            }
        )

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "roi": [CRACK_ROI_X0, CRACK_ROI_Y0, CRACK_ROI_W, CRACK_ROI_H],
                    "val_dice_05": best_val_dice,
                },
                best_ckpt,
            )

    torch.save({"model_state": model.state_dict(), "epoch": epochs}, last_ckpt)
    train_duration_s = round(time.perf_counter() - t0, 2)

    # Load best checkpoint and calibrate probability threshold on VAL split ONLY
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    val_prob_list: list[np.ndarray] = []
    val_gt_list: list[np.ndarray] = []
    with torch.no_grad():
        for imgs, masks in val_loader:
            probs = torch.sigmoid(model(imgs.to(device))).cpu().numpy()[:, 0]
            gts = masks.cpu().numpy()[:, 0]
            for p, g in zip(probs, gts):
                val_prob_list.append(p)
                val_gt_list.append((g > 0.5).astype(np.uint8) * 255)

    best_thresh = 0.5
    best_thresh_metrics: dict[str, float] = {}
    best_thresh_dice = -1.0
    for thr in (0.35, 0.45, 0.50, 0.55, 0.65, 0.75, 0.85):
        bin_preds = [(p >= thr).astype(np.uint8) * 255 for p in val_prob_list]
        m = compute_segmentation_metrics(bin_preds, val_gt_list)
        if m["dice"] > best_thresh_dice:
            best_thresh_dice = m["dice"]
            best_thresh = thr
            best_thresh_metrics = m

    # Re-save best checkpoint with calibrated val threshold
    ckpt["threshold"] = best_thresh
    ckpt["val_metrics"] = best_thresh_metrics
    torch.save(ckpt, best_ckpt)

    summary = {
        "trained": True,
        "architecture": "RailCrackUNet (base_ch=16, ROI 512x256)",
        "train_positives": n_tr_pos,
        "train_negatives": n_tr_neg,
        "val_positives": n_va_pos,
        "val_negatives": n_va_neg,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "val_calibrated_threshold": best_thresh,
        "val_metrics": best_thresh_metrics,
        "train_duration_seconds": train_duration_s,
        "best_checkpoint": str(best_ckpt),
        "last_checkpoint": str(last_ckpt),
        "history": history,
    }
    (output_dir / "crack_unet_training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


class RailCrackSegmentor:
    """Inference wrapper for frozen RailCrackUNet."""

    def __init__(self, checkpoint_path: Path, min_area_px: int = 8) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model = RailCrackUNet(base_ch=16).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.threshold = float(ckpt.get("threshold", 0.5))
        self.min_area_px = int(min_area_px)

    @torch.no_grad()
    def predict_frame(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        t0 = time.perf_counter()
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        roi = crop_rail_roi(rgb)
        inp = torch.from_numpy(roi.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device) / 255.0
        prob_roi = torch.sigmoid(self.model(inp))[0, 0].cpu().numpy()
        bin_roi = (prob_roi >= self.threshold).astype(np.uint8) * 255
        full_mask = uncrop_roi_mask(bin_roi, full_w=w, full_h=h)

        ys, xs = np.where(full_mask > 127)
        fg_area = int(len(xs))
        detected = fg_area >= self.min_area_px
        max_conf = float(np.max(prob_roi)) if fg_area > 0 else float(np.max(prob_roi))
        mean_fg_conf = (
            float(np.mean(prob_roi[bin_roi > 127])) if fg_area > 0 else 0.0
        )
        bbox = None
        if detected:
            bbox = [
                float(np.min(xs)),
                float(np.min(ys)),
                float(np.max(xs) + 1),
                float(np.max(ys) + 1),
            ]
        else:
            full_mask[:] = 0
            fg_area = 0

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        return {
            "detected": detected,
            "class_id": 2,
            "class_name": "rail_crack",
            "confidence": round(mean_fg_conf if detected else max_conf, 4),
            "pixel_area": fg_area,
            "bbox": bbox,
            "mask": full_mask,
            "inference_time_ms": latency_ms,
        }

