import json
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from trackers.core.base import BaseTracker

from mouselite.tracker import DEFAULT_FPS, TRACKERS, read_fps, retrack


class FakeTracker(BaseTracker):
    def __init__(self, **kwargs) -> None:
        pass

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


def test_retrack(tmp_path: Path, monkeypatch) -> None:
    """retrack() replays a cached COCO export through a tracker, skipping detection."""
    monkeypatch.setitem(TRACKERS, "fake", FakeTracker)
    annotations_path = make_coco_dataset(tmp_path)

    output = retrack(
        annotations_path,
        "fake",
        output_dir=tmp_path / "output",
        show_progress=False,
    )
    assert output.exists()


def test_read_fps(tmp_path: Path) -> None:
    """read_fps() returns the frame rate recorded by `run`, or the default if absent."""
    annotations_path = make_coco_dataset(tmp_path)
    assert read_fps(annotations_path) == DEFAULT_FPS

    coco = json.loads(annotations_path.read_text())
    coco["info"] = {"fps": 25.0}
    annotations_path.write_text(json.dumps(coco))
    assert read_fps(annotations_path) == 25.0


def test_retrack_uses_recorded_fps(tmp_path: Path, monkeypatch) -> None:
    """Without an explicit fps, retrack() replays the export at its recorded frame rate."""
    monkeypatch.setitem(TRACKERS, "fake", FakeTracker)
    annotations_path = make_coco_dataset(tmp_path)
    coco = json.loads(annotations_path.read_text())
    coco["info"] = {"fps": 25.0}
    annotations_path.write_text(json.dumps(coco))

    output = retrack(
        annotations_path,
        "fake",
        output_dir=tmp_path / "output",
        show_progress=False,
    )
    assert sv.VideoInfo.from_video_path(str(output)).fps == 25.0
