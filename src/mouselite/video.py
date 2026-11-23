"""Read frames and frame info from video files."""

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import supervision as sv


def read_frames(video_path: str | Path) -> Iterator[np.ndarray]:
    """Every frame of `video_path`, read until the decoder runs out."""
    cap = cv2.VideoCapture(str(video_path))
    ok, first = cap.read()
    if not ok:
        cap.release()
        raise ValueError(f"no frames could be read from {video_path}")

    def frames() -> Iterator[np.ndarray]:
        try:
            frame = first
            while True:
                yield frame
                ok, frame = cap.read()
                if not ok:
                    return
        finally:
            cap.release()

    return frames()


def progress_total(video_info: sv.VideoInfo) -> int | None:
    """The container's frame count for a progress bar, or None when it's not valid."""
    total = video_info.total_frames
    return total if total is not None and total > 0 else None
