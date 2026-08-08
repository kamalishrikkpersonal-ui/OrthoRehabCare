"""Video reading utilities for OrthoRehab AI Step 1.

Reads a video file sequentially with OpenCV, exposing each decoded frame along
with its frame index and timestamp. This isolates all OpenCV video handling so
the landmark extractor stays clean and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import cv2


@dataclass
class VideoInfo:
    """Metadata about an opened video file."""

    path: str
    fps: float
    width: int
    height: int
    frame_count: int  # total frames reported by the container (may be 0 for some formats)


@dataclass
class VideoFrame:
    """A single decoded frame plus context."""

    frame_index: int
    timestamp: float  # seconds from the start of the video
    image: object     # OpenCV BGR ndarray


def read_video_info(path: str) -> VideoInfo:
    """Open a video and return its metadata.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file cannot be opened as a video.
    """
    import os

    if not os.path.exists(path):
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    if fps <= 0:
        # Some files do not report FPS; fall back to a safe default.
        fps = 30.0

    return VideoInfo(path=os.path.abspath(path), fps=fps, width=width, height=height, frame_count=frame_count)


def iter_video_frames(path: str) -> Iterator[VideoFrame]:
    """Yield every frame of a video with its index and timestamp.

    Frames are read sequentially using the container FPS. If FPS is unknown or
    zero, a default of 30 is used so timestamps remain consistent and ordered.

    Args:
        path: Path to the video file.

    Yields:
        VideoFrame objects in order of appearance.
    """
    info = read_video_info(path)
    fps = info.fps if info.fps > 0 else 30.0

    cap = cv2.VideoCapture(info.path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")

    try:
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = frame_index / fps
            yield VideoFrame(frame_index=frame_index, timestamp=timestamp, image=frame)
            frame_index += 1
    finally:
        cap.release()
