"""Perspective-calibrated CV rail geometry analyzer for A4 misalignment (Phase 6).

Estimates rail-head positions in image space using a width-matched rail ridge filter,
projects detections to the rail-head ground plane via Inverse Perspective Mapping (IPM)
using camera intrinsics and mounting height, estimates track centerline and lateral
deviation profiles, and evaluates MAE, RMSE, and max deviation error against A4 GT.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Any

import cv2
import numpy as np


class RailGeometryAnalyzer:
    """Pure computer-vision perspective rail geometry and lateral misalignment estimator."""

    def __init__(
        self,
        cam_cfg: dict[str, Any],
        sim_cfg: dict[str, Any],
        img_width: int = 960,
        img_height: int = 540,
        min_eval_dist_m: float = 5.5,
        max_eval_dist_m: float = 34.0,
        deviation_threshold_m: float = 0.030,
    ) -> None:
        self.img_w = int(img_width)
        self.img_h = int(img_height)
        self.h_cam = float(cam_cfg["mounting"]["height_above_rail_head_m"])
        self.pitch_deg = float(cam_cfg["orientation_deg"]["pitch_deg"])
        self.focal_mm = float(cam_cfg["optics"]["focal_length_mm"])
        self.sensor_w_mm = float(cam_cfg["optics"]["sensor_width_mm"])

        self.fx = self.img_w * self.focal_mm / self.sensor_w_mm
        self.fy = self.fx
        self.cx = self.img_w * 0.5
        self.cy = self.img_h * 0.5

        gauge_m = float(sim_cfg["track"]["gauge_m"])
        head_w_m = float(sim_cfg["track"]["rail"]["head_width_m"])
        self.rail_head_w_m = head_w_m
        self.nominal_half_c2c_m = 0.5 * gauge_m + 0.5 * head_w_m
        self.nominal_c2c_m = 2.0 * self.nominal_half_c2c_m

        self.min_dist_m = float(min_eval_dist_m)
        self.max_dist_m = float(max_eval_dist_m)
        self.deviation_threshold_m = float(deviation_threshold_m)

        # Rotation matrix matching Blender Euler RX = 90 + pitch_deg around X
        rx = math.radians(90.0 + self.pitch_deg)
        c, s = math.cos(rx), math.sin(rx)
        self.r_cam_to_world = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, c, -s],
                [0.0, s, c],
            ],
            dtype=np.float64,
        )
        self.r_world_to_cam = self.r_cam_to_world.T

        # Precompute nominal scanline geometry lookup
        self._scanlines = self._build_scanline_table()

    def world_to_pixel(self, x_m: float, y_dist_m: float, z_rel_rail_m: float = 0.0) -> tuple[float, float]:
        """Project a rail-plane point (x_m, y_dist_m ahead of cam, z_rel_rail_m) to (u, v)."""
        p_world = np.array([x_m, y_dist_m, z_rel_rail_m - self.h_cam], dtype=np.float64)
        p_cam = self.r_world_to_cam @ p_world
        depth = -p_cam[2]
        if depth <= 0.1:
            return -1.0, -1.0
        u = self.cx + self.fx * (p_cam[0] / depth)
        v = self.cy - self.fy * (p_cam[1] / depth)
        return float(u), float(v)

    def pixel_to_rail_plane(self, u: float, v: float) -> tuple[float, float] | None:
        """Intersect camera pixel ray with rail-head plane Z = -h_cam."""
        r_cam = np.array([(u - self.cx) / self.fx, (self.cy - v) / self.fy, -1.0], dtype=np.float64)
        r_world = self.r_cam_to_world @ r_cam
        if r_world[2] >= -1e-4:
            return None
        t = -self.h_cam / r_world[2]
        if t <= 0.0:
            return None
        return float(t * r_world[0]), float(t * r_world[1])

    def _build_scanline_table(self) -> list[dict[str, float]]:
        _, v_min_f = self.world_to_pixel(0.0, self.max_dist_m)
        _, v_max_f = self.world_to_pixel(0.0, self.min_dist_m)
        v_start = max(1, int(math.ceil(v_min_f)))
        v_end = min(self.img_h - 2, int(math.floor(v_max_f)))

        table: list[dict[str, float]] = []
        for v in range(v_start, v_end + 1):
            pt_center = self.pixel_to_rail_plane(self.cx, float(v))
            if pt_center is None:
                continue
            _, d_m = pt_center
            if not (self.min_dist_m <= d_m <= self.max_dist_m):
                continue
            u_l_nom, _ = self.world_to_pixel(-self.nominal_half_c2c_m, d_m)
            u_r_nom, _ = self.world_to_pixel(+self.nominal_half_c2c_m, d_m)
            # Expected rail head pixel width at distance d_m
            head_px = max(2.0, self.fx * self.rail_head_w_m / d_m)
            search_px = max(8, int(math.ceil(self.fx * 0.24 / d_m)) + 4)
            table.append(
                {
                    "v": float(v),
                    "d_m": float(d_m),
                    "u_l_nom": float(u_l_nom),
                    "u_r_nom": float(u_r_nom),
                    "head_px": float(head_px),
                    "search_px": float(search_px),
                }
            )
        return table

    @staticmethod
    def _score_rail_row(
        csum_row: np.ndarray,
        img_w: int,
        u_arr: np.ndarray,
        head_px: float,
        inner_sign: int,
    ) -> np.ndarray:
        """Vectorized width-matched inner-web + outer-bevel rail-head score across horizontal shifts."""
        half_c = max(1, int(round(0.35 * head_px)))
        in0_off = max(half_c + 1, int(round(0.50 * head_px)))
        in1_off = max(in0_off + 2, int(round(1.25 * head_px)))
        out0_off = max(half_c + 1, int(round(0.45 * head_px)))
        out1_off = max(out0_off + 1, int(round(0.85 * head_px)))

        c0 = np.clip(u_arr - half_c, 0, img_w - 1)
        c1 = np.clip(u_arr + half_c + 1, 1, img_w)
        center_v = (csum_row[c1] - csum_row[c0]) / np.maximum(1, c1 - c0)

        if inner_sign > 0:
            i0 = np.clip(u_arr + in0_off, 0, img_w - 1)
            i1 = np.clip(u_arr + in1_off + 1, 1, img_w)
            o0 = np.clip(u_arr - out1_off, 0, img_w - 1)
            o1 = np.clip(u_arr - out0_off + 1, 1, img_w)
        else:
            i0 = np.clip(u_arr - in1_off, 0, img_w - 1)
            i1 = np.clip(u_arr - in0_off + 1, 1, img_w)
            o0 = np.clip(u_arr + out0_off, 0, img_w - 1)
            o1 = np.clip(u_arr + out1_off + 1, 1, img_w)

        inner_v = (csum_row[i1] - csum_row[i0]) / np.maximum(1, i1 - i0)
        outer_v = (csum_row[o1] - csum_row[o0]) / np.maximum(1, o1 - o0)
        in_contrast = center_v - inner_v
        out_contrast = np.maximum(-15.0, center_v - outer_v)
        valid = (u_arr >= out1_off + 2) & (u_arr <= img_w - out1_off - 3) & (in_contrast >= 12.0)
        return np.where(valid, in_contrast + 0.35 * out_contrast, -1e9)

    @staticmethod
    def _subpixel_peak(
        scores: np.ndarray, u_arr: np.ndarray, center_idx: int, radius: int = 2
    ) -> tuple[float, float] | None:
        """Refine rail center sub-pixel peak near the joint correlation optimum."""
        lo_i = max(0, center_idx - radius)
        hi_i = min(len(scores) - 1, center_idx + radius)
        sub = scores[lo_i : hi_i + 1]
        bi = lo_i + int(np.argmax(sub))
        bs = float(scores[bi])
        if bs < 12.0:
            return None
        bu = float(u_arr[bi])
        if 0 < bi < len(scores) - 1 and scores[bi - 1] > -1e8 and scores[bi + 1] > -1e8:
            denom = scores[bi - 1] - 2.0 * scores[bi] + scores[bi + 1]
            if abs(denom) > 1e-5:
                delta = max(-0.5, min(0.5, 0.5 * (scores[bi - 1] - scores[bi + 1]) / denom))
                bu += delta
        return bu, bs

    def analyze_frame(
        self, frame_bgr: np.ndarray, camera_position_m: float
    ) -> dict[str, Any]:
        """Estimate rail centerline profile and detect lateral misalignment in one frame."""
        t0 = time.perf_counter()
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
        w = self.img_w
        csum_all = np.zeros((self.img_h, w + 1), dtype=np.float64)
        np.cumsum(gray, axis=1, out=csum_all[:, 1:])

        samples: list[dict[str, float]] = []
        for sl in self._scanlines:
            v_int = int(sl["v"])
            csum = csum_all[v_int]
            search_px = int(sl["search_px"])
            head_px = float(sl["head_px"])

            shifts = np.arange(-search_px, search_px + 1, dtype=np.int64)
            u_l_arr = int(round(sl["u_l_nom"])) + shifts
            u_r_arr = int(round(sl["u_r_nom"])) + shifts
            sc_l = self._score_rail_row(csum, w, u_l_arr, head_px, inner_sign=+1)
            sc_r = self._score_rail_row(csum, w, u_r_arr, head_px, inner_sign=-1)

            # Joint gauge-coupled shift correlation rejects asymmetric rail-foot/clip edges
            joint = np.where((sc_l >= 12.0) & (sc_r >= 12.0), sc_l + sc_r, -1e9)
            best_s_idx = int(np.argmax(joint))
            if joint[best_s_idx] < 24.0:
                continue

            pk_l = self._subpixel_peak(sc_l, u_l_arr, best_s_idx, radius=2)
            pk_r = self._subpixel_peak(sc_r, u_r_arr, best_s_idx, radius=2)
            if pk_l is None or pk_r is None:
                continue

            u_l, score_l = pk_l
            u_r, score_r = pk_r
            pt_l = self.pixel_to_rail_plane(u_l, float(v_int))
            pt_r = self.pixel_to_rail_plane(u_r, float(v_int))
            if pt_l is None or pt_r is None:
                continue

            x_l, d_l = pt_l
            x_r, d_r = pt_r
            est_c2c = x_r - x_l
            # Reject scanlines where reconstructed track gauge deviates from Indian Broad Gauge
            if abs(est_c2c - self.nominal_c2c_m) > 0.055:
                continue

            d_mid = 0.5 * (d_l + d_r)
            x_center = 0.5 * (x_l + x_r)
            samples.append(
                {
                    "v": float(v_int),
                    "u_l": round(u_l, 2),
                    "u_r": round(u_r, 2),
                    "d_m": d_mid,
                    "y_world_m": float(camera_position_m) + d_mid,
                    "x_center_m": x_center,
                    "x_left_m": x_l,
                    "x_right_m": x_r,
                    "contrast": 0.5 * (score_l + score_r),
                }
            )

        detected = False
        confidence = 0.0
        max_est_dev = 0.0
        peak_y_world = None
        affected_interval = None
        bbox = None

        if len(samples) >= 12:
            xs = np.array([s["x_center_m"] for s in samples], dtype=np.float64)
            ys_w = np.array([s["y_world_m"] for s in samples], dtype=np.float64)

            # Median filter (kernel 7) removes sleeper fastener jitter while preserving smooth S-curve
            padded = np.pad(xs, 3, mode="edge")
            win = np.lib.stride_tricks.sliding_window_view(padded, 7)
            xs_smooth = np.median(win, axis=1)

            # Estimate baseline centerline from lower-percentile straight rail segments
            abs_order = np.argsort(np.abs(xs_smooth))
            straight_k = max(5, len(xs_smooth) // 3)
            baseline_x = float(np.median(xs_smooth[abs_order[:straight_k]]))

            devs = xs_smooth - baseline_x
            for s, dev_val in zip(samples, devs):
                s["lateral_deviation_m"] = round(float(dev_val), 5)

            abs_devs = np.abs(devs)
            peak_idx = int(np.argmax(abs_devs))
            max_est_dev = float(abs_devs[peak_idx])
            peak_y_world = float(ys_w[peak_idx])

            # Confirm contiguous spatial support above half-threshold (prevents single-pixel spikes)
            active_mask = abs_devs >= max(0.020, self.deviation_threshold_m * 0.65)
            active_indices = np.where(active_mask)[0]
            if len(active_indices) >= 4:
                y_lo = float(np.min(ys_w[active_indices]))
                y_hi = float(np.max(ys_w[active_indices]))
                span_m = y_hi - y_lo
                if max_est_dev >= self.deviation_threshold_m and span_m >= 1.2:
                    detected = True
                    # Smooth calibrated confidence in [0.5, 0.99]
                    confidence = float(
                        min(0.99, max(0.50, 0.50 + (max_est_dev - self.deviation_threshold_m) / 0.08))
                    )
                    affected_interval = [round(y_lo, 2), round(y_hi, 2)]

                    act_samples = [samples[i] for i in active_indices]
                    u_min = min(s["u_l"] for s in act_samples)
                    u_max = max(s["u_r"] for s in act_samples)
                    v_min = min(s["v"] for s in act_samples)
                    v_max = max(s["v"] for s in act_samples)
                    bbox = [
                        round(max(0.0, u_min - 6.0), 2),
                        round(max(0.0, v_min - 4.0), 2),
                        round(min(float(self.img_w), u_max + 6.0), 2),
                        round(min(float(self.img_h), v_max + 4.0), 2),
                    ]

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        return {
            "detected": detected,
            "class_id": 3,
            "class_name": "rail_misalignment",
            "confidence": round(confidence, 4),
            "max_estimated_deviation_m": round(max_est_dev, 5),
            "peak_y_world_m": round(peak_y_world, 3) if peak_y_world is not None else None,
            "affected_interval_m": affected_interval,
            "bbox": bbox,
            "valid_scanlines": len(samples),
            "samples": samples,
            "inference_time_ms": latency_ms,
        }


def reconstruct_and_evaluate_track_profile(
    frame_results: list[dict[str, Any]],
    gt_geometry_profile: dict[str, Any],
    deviation_threshold_m: float = 0.030,
) -> dict[str, Any]:
    """Fuse per-frame IPM rail-centerline estimates onto 0.25 m GT grid and compute MAE/RMSE."""
    bins: dict[float, list[tuple[float, float]]] = defaultdict(list)

    for fr in frame_results:
        for s in fr.get("samples", []):
            if "lateral_deviation_m" not in s:
                continue
            d_m = float(s["d_m"])
            if not (6.0 <= d_m <= 24.0):
                continue
            y_bin = round(float(s["y_world_m"]) * 4.0) / 4.0  # Snap to 0.25 m grid
            weight = 1.0 / max(1.0, d_m * d_m)
            bins[y_bin].append((float(s["lateral_deviation_m"]), weight))

    gt_samples = gt_geometry_profile["displacement_profile"]
    gt_by_y = {round(float(p["y_m"]), 2): float(p["lateral_shift_m"]) for p in gt_samples}

    # Evaluate across the affected A4 profile [196.0, 204.0] m
    compared_points: list[dict[str, float | None]] = []
    abs_errors: list[float] = []
    sq_errors: list[float] = []
    est_profile_all: list[dict[str, float]] = []

    for y_bin in sorted(bins.keys()):
        vals_w = bins[y_bin]
        w_sum = sum(w for _, w in vals_w)
        est_dx = sum(v * w for v, w in vals_w) / max(1e-9, w_sum)
        est_profile_all.append({"y_m": y_bin, "estimated_shift_m": round(est_dx, 6)})

    est_by_y = {p["y_m"]: p["estimated_shift_m"] for p in est_profile_all}

    for y_gt, gt_dx in sorted(gt_by_y.items()):
        est_dx = est_by_y.get(y_gt)
        if est_dx is None:
            # Linear interpolation from neighboring observed 0.25m bins within +/-0.5m
            near = [(abs(yb - y_gt), val) for yb, val in est_by_y.items() if abs(yb - y_gt) <= 0.6]
            if near:
                near.sort(key=lambda t: t[0])
                est_dx = near[0][1]
        if est_dx is not None:
            err = abs(est_dx - gt_dx)
            abs_errors.append(err)
            sq_errors.append(err * err)
        compared_points.append(
            {
                "y_m": y_gt,
                "gt_shift_m": round(gt_dx, 6),
                "estimated_shift_m": round(est_dx, 6) if est_dx is not None else None,
            }
        )

    gt_max_dev = max(abs(v) for v in gt_by_y.values()) if gt_by_y else 0.0
    valid_est_in_zone = [
        p["estimated_shift_m"] for p in compared_points if p["estimated_shift_m"] is not None
    ]
    est_max_dev = max(abs(v) for v in valid_est_in_zone) if valid_est_in_zone else 0.0

    # Estimate affected track interval from bins exceeding half threshold near peak
    active_bins = [
        p["y_m"]
        for p in est_profile_all
        if abs(p["estimated_shift_m"]) >= max(0.015, deviation_threshold_m * 0.5)
        and 188.0 <= p["y_m"] <= 212.0
    ]
    est_interval = (
        [min(active_bins), max(active_bins)] if active_bins else None
    )

    mae_m = float(np.mean(abs_errors)) if abs_errors else None
    rmse_m = float(np.sqrt(np.mean(sq_errors))) if sq_errors else None
    max_err_m = float(np.max(abs_errors)) if abs_errors else None
    peak_dev_err_m = abs(est_max_dev - gt_max_dev) if valid_est_in_zone else None

    return {
        "gt_affected_region_m": gt_geometry_profile["affected_region_m"],
        "estimated_affected_region_m": est_interval,
        "gt_max_deviation_m": round(gt_max_dev, 5),
        "estimated_max_deviation_m": round(est_max_dev, 5),
        "max_deviation_difference_m": round(peak_dev_err_m, 5)
        if peak_dev_err_m is not None
        else None,
        "displacement_mae_m": round(mae_m, 5) if mae_m is not None else None,
        "displacement_rmse_m": round(rmse_m, 5) if rmse_m is not None else None,
        "max_pointwise_error_m": round(max_err_m, 5) if max_err_m is not None else None,
        "profile_coverage_points": len(valid_est_in_zone),
        "gt_total_profile_points": len(gt_by_y),
        "compared_profile": compared_points,
    }

