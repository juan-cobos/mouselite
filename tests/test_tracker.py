import json
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np
import supervision as sv
from trackers.core.base import BaseTracker

from mouselite.tracker import TRACKERS, retrack


class FakeTracker(BaseTracker):
    seen: ClassVar[list[sv.Detections]] = []

    def __init__(self, **kwargs) -> None:
        FakeTracker.seen = []

    def reset(self) -> None:
        pass

    def update(
        self,
        detections: sv.Detections,
        frame=None,
        timestamp=None,
    ) -> sv.Detections:
        detections.tracker_id = np.arange(len(detections))
        FakeTracker.seen.append(detections)
        return detections


def make_coco_dataset(
    tmp_path: Path,
    frames: int = 4,
    size: int = 64,
    keypoints: bool = False,
) -> Path:
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
    if keypoints:
        coco = json.loads(annotations_path.read_text())
        for annotation in coco["annotations"]:
            annotation["keypoints"] = [10, 10, 2, 15, 15, 2, 0, 0, 0]  # 3 (x, y, v)
            annotation["num_keypoints"] = 2
        annotations_path.write_text(json.dumps(coco))
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


def test_retrack_carries_keypoints(tmp_path: Path, monkeypatch) -> None:
    """retrack() passes the export's keypoints through to the tracker and annotator."""
    monkeypatch.setitem(TRACKERS, "fake", FakeTracker)
    annotations_path = make_coco_dataset(tmp_path, frames=2, keypoints=True)

    retrack(annotations_path, "fake", output_dir=tmp_path / "output", show_progress=False)

    assert len(FakeTracker.seen) == 2
    for detections in FakeTracker.seen:
        assert detections.data["keypoints_xy"].shape == (1, 3, 2)
        assert detections.data["keypoints_visible"].tolist() == [[True, True, False]]


def test_retrack_writes_track_ids(tmp_path: Path, monkeypatch) -> None:
    """retrack() writes the new track ids back into annotations.json."""
    monkeypatch.setitem(TRACKERS, "fake", FakeTracker)
    annotations_path = make_coco_dataset(tmp_path, frames=2)
    assert "track_id" not in json.loads(annotations_path.read_text())["annotations"][0]

    retrack(annotations_path, "fake", output_dir=tmp_path / "output", show_progress=False)

    coco = json.loads(annotations_path.read_text())
    assert [a["track_id"] for a in coco["annotations"]] == [0, 0]
