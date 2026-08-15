from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from trackers.core.base import BaseTracker

from mouselite.tracker import retrack


class FakeTracker(BaseTracker):
    def reset(self) -> None:
        pass

    def update(
        self,
        detections: sv.Detections,
        frame=None,
        timestamp=None,
    ) -> sv.Detections:
        detections.tracker_id = np.arange(len(detections))
        return detections


def make_coco_dataset(tmp_path: Path, frames: int = 4, size: int = 64) -> Path:
    """Write a minimal COCO dataset (images/ + annotations.json) for retrack() to read."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    image_paths = []
    annotations = {}
    for i in range(frames):
        frame = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
        image_path = images_dir / f"frame_{i:02d}.jpg"
        cv2.imwrite(str(image_path), frame)
        image_paths.append(str(image_path))
        annotations[str(image_path)] = sv.Detections(
            xyxy=np.array([[5, 5, 20, 20]], dtype=np.float32),
            confidence=np.array([0.9], dtype=np.float32),
            class_id=np.array([0]),
        )

    dataset = sv.DetectionDataset(
        classes=["mouse"],
        images=image_paths,
        annotations=annotations,
    )
    annotations_path = tmp_path / "annotations.json"
    dataset.as_coco(annotations_path=str(annotations_path))
    return annotations_path


def test_retrack(tmp_path: Path) -> None:
    """retrack() replays a cached COCO export through a tracker, skipping detection."""
    annotations_path = make_coco_dataset(tmp_path)

    output = retrack(
        annotations_path,
        FakeTracker(),
        output_dir=tmp_path / "output",
        show_progress=False,
    )
    assert output.exists()
