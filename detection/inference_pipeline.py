"""Unified hybrid inference pipeline combining YOLO, RailCrackUNet, and RailGeometryAnalyzer."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from detection.crack_segmentation import RailCrackSegmentor
from detection.geometry_analyzer import RailGeometryAnalyzer


class HybridRailwayDetector:
    """Extensible detector routing A1/A2 -> YOLO, A3 -> U-Net, A4 -> IPM Geometry."""

    def __init__(
        self,
        yolo_weights_path: Path,
        crack_unet_path: Path,
        cam_cfg: dict[str, Any],
        sim_cfg: dict[str, Any],
        yolo_conf: float = 0.25,
        yolo_iou: float = 0.45,
        imgsz: int = 640,
        img_width: int = 960,
        img_height: int = 540,
    ) -> None:
        p = Path(yolo_weights_path).resolve()
        results_dir = next((parent for parent in p.parents if parent.name == "results"), p.parents[min(4, len(p.parents) - 1)])
        os.environ.setdefault("YOLO_CONFIG_DIR", str(results_dir / "_ultralytics_cfg"))
        from ultralytics import YOLO

        self.yolo = YOLO(str(yolo_weights_path))
        self.yolo_conf = float(yolo_conf)
        self.yolo_iou = float(yolo_iou)
        self.imgsz = int(imgsz)
        self.device = 0 if torch.cuda.is_available() else "cpu"

        self.crack_segmentor = (
            RailCrackSegmentor(crack_unet_path) if crack_unet_path.is_file() else None
        )
        self.geometry_analyzer = RailGeometryAnalyzer(
            cam_cfg=cam_cfg,
            sim_cfg=sim_cfg,
            img_width=img_width,
            img_height=img_height,
        )

    def predict_frame(
        self, frame_bgr: np.ndarray, camera_position_m: float
    ) -> dict[str, Any]:
        """Run all three frozen anomaly detectors on one BGR frame."""
        t_start = time.perf_counter()

        # 1. YOLO object detection (Class 0: animal_carcass, Class 1: grain_spillage)
        t_y0 = time.perf_counter()
        yolo_res = self.yolo.predict(
            source=frame_bgr,
            imgsz=self.imgsz,
            conf=self.yolo_conf,
            iou=self.yolo_iou,
            device=self.device,
            verbose=False,
        )[0]
        yolo_ms = round((time.perf_counter() - t_y0) * 1000.0, 3)

        yolo_detections: list[dict[str, Any]] = []
        if yolo_res.boxes is not None and len(yolo_res.boxes) > 0:
            xyxy = yolo_res.boxes.xyxy.cpu().numpy()
            confs = yolo_res.boxes.conf.cpu().numpy()
            clss = yolo_res.boxes.cls.cpu().numpy()
            names = yolo_res.names
            for box, cf, cid in zip(xyxy, confs, clss):
                cid_i = int(cid)
                yolo_detections.append(
                    {
                        "class_id": cid_i,
                        "class_name": str(names.get(cid_i, cid_i)),
                        "confidence": round(float(cf), 4),
                        "bbox": [round(float(v), 2) for v in box.tolist()],
                    }
                )

        # 2. A3 Rail Crack Segmentation (PyTorch U-Net)
        if self.crack_segmentor is not None:
            crack_res = self.crack_segmentor.predict_frame(frame_bgr)
        else:
            crack_res = {
                "detected": False,
                "class_id": 2,
                "class_name": "rail_crack",
                "confidence": 0.0,
                "pixel_area": 0,
                "bbox": None,
                "mask": None,
                "inference_time_ms": 0.0,
            }

        # 3. A4 Rail Misalignment Geometry Analyzer (IPM)
        geom_res = self.geometry_analyzer.analyze_frame(frame_bgr, camera_position_m)

        total_ms = round((time.perf_counter() - t_start) * 1000.0, 3)
        return {
            "yolo_detections": yolo_detections,
            "crack_prediction": crack_res,
            "geometry_prediction": geom_res,
            "latencies_ms": {
                "yolo_ms": yolo_ms,
                "crack_unet_ms": crack_res["inference_time_ms"],
                "geometry_ms": geom_res["inference_time_ms"],
                "total_ms": total_ms,
            },
        }


def annotate_frame(
    frame_bgr: np.ndarray,
    pred: dict[str, Any],
    frame_num: int,
    speed_kmh: float | int,
    camera_position_m: float,
) -> np.ndarray:
    """Render readable prediction overlays (YOLO boxes, A3 mask, A4 rail lines/telemetry)."""
    canvas = frame_bgr.copy()

    # 1. A3 crack segmentation overlay (red)
    crack = pred["crack_prediction"]
    if crack.get("mask") is not None and crack.get("detected"):
        fg = crack["mask"] > 127
        canvas[fg] = (canvas[fg] * 0.30 + np.array([0, 0, 255]) * 0.70).astype(np.uint8)
        if crack.get("bbox"):
            x1, y1, x2, y2 = [int(round(v)) for v in crack["bbox"]]
            cv2.rectangle(canvas, (max(0, x1 - 4), max(0, y1 - 4)), (x2 + 4, y2 + 4), (0, 0, 255), 2)
            cv2.putText(
                canvas,
                f"rail_crack {crack['confidence']:.2f} ({crack['pixel_area']}px)",
                (max(4, x1 - 4), max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 80, 255),
                2,
                cv2.LINE_AA,
            )

    # 2. A4 detected rail samples + misalignment alert
    geom = pred["geometry_prediction"]
    for s in geom.get("samples", []):
        v = int(round(s["v"]))
        ul = int(round(s["u_l"]))
        ur = int(round(s["u_r"]))
        dev = abs(s.get("lateral_deviation_m", 0.0))
        pt_col = (0, 140, 255) if dev >= 0.025 else (200, 220, 40)
        cv2.circle(canvas, (ul, v), 1, pt_col, -1)
        cv2.circle(canvas, (ur, v), 1, pt_col, -1)

    if geom.get("detected") and geom.get("bbox"):
        gx1, gy1, gx2, gy2 = [int(round(v)) for v in geom["bbox"]]
        cv2.rectangle(canvas, (gx1, gy1), (gx2, gy2), (0, 165, 255), 2)
        cv2.putText(
            canvas,
            f"rail_misalignment dev={geom['max_estimated_deviation_m']*100:.1f}cm ({geom['confidence']:.2f})",
            (max(4, gx1), max(22, gy1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 180, 255),
            2,
            cv2.LINE_AA,
        )

    # 3. YOLO object detection boxes (green=animal_carcass, gold=grain_spillage)
    for det in pred["yolo_detections"]:
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
        color = (0, 230, 80) if det["class_id"] == 0 else (0, 215, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            canvas,
            f"{det['class_name']} {det['confidence']:.2f}",
            (max(4, x1), max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
            cv2.LINE_AA,
        )

    # 4. Telemetry HUD bar
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 32), (18, 20, 24), -1)
    hud = (
        f"ITMS-VirtualTrack [SYNTHETIC] | {speed_kmh} km/h | F{frame_num:04d} | "
        f"y={camera_position_m:.2f}m | lat={pred['latencies_ms']['total_ms']:.1f}ms | "
        f"dev={geom['max_estimated_deviation_m']*100:.1f}cm"
    )
    cv2.putText(
        canvas,
        hud,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (235, 240, 245),
        1,
        cv2.LINE_AA,
    )
    return canvas
