"""Deterministic video frame extraction utilities for ITMS-VirtualTrack."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    stem_prefix: str,
    expected_frame_count: int,
    frame_stride: int = 1,
    image_ext: str = "png",
) -> dict[str, Any]:
    """Extract frames sequentially from a synthetic video without dropping unreadable frames."""
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if frame_stride < 1:
        raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV failed to open video: {video_path}")

    reported_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[dict[str, Any]] = []
    decode_errors: list[int] = []

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_num = frame_idx + 1  # 1-indexed matching Blender frame numbers
        if (frame_idx % frame_stride) == 0:
            filename = f"{stem_prefix}_f{frame_num:04d}.{image_ext}"
            out_path = output_dir / filename
            if image_ext.lower() == "png":
                saved = cv2.imwrite(
                    str(out_path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 3]
                )
            else:
                saved = cv2.imwrite(str(out_path), frame)
            if not saved:
                decode_errors.append(frame_num)
            extracted.append(
                {
                    "frame_number": frame_num,
                    "frame_index": frame_idx,
                    "filename": filename,
                    "path": str(out_path),
                }
            )
        frame_idx += 1

    cap.release()

    if frame_idx != expected_frame_count:
        raise RuntimeError(
            f"Frame count mismatch for {video_path.name}: decoded {frame_idx} frames, "
            f"expected {expected_frame_count} (reported by container: {reported_count})."
        )
    if decode_errors:
        raise RuntimeError(
            f"Failed to write extracted frames {decode_errors} from {video_path.name}"
        )

    return {
        "video_path": str(video_path),
        "decoded_frames": frame_idx,
        "reported_frames": reported_count,
        "extracted_count": len(extracted),
        "frame_stride": frame_stride,
        "fps": round(fps, 3),
        "resolution": [width, height],
        "frames": extracted,
    }

