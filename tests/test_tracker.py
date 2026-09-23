import json
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np
import pytest
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


def make_export(
    tmp_path: Path,
    frames: int = 4,
    size: int = 64,
    every: int = 1,
    keypoints: bool = False,
) -> tuple[Path, Path]:
    """Write a video and a COCO export of it (annotations.json) for retrack() to read."""
    video_path = tmp_path / "video.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (size, size)
    )
    for _ in range(frames):
        writer.write(np.random.randint(0, 255, (size, size, 3), dtype=np.uint8))
    writer.release()

    images, annotations = [], []
    for image_id, frame_idx in enumerate(range(0, frames, every), start=1):
        images.append(
            {
                "id": image_id,
                "file_name": f"video_{frame_idx:06d}.jpg",
                "height": size,
                "width": size,
                "frame_index": frame_idx,
            },
        )
        annotation = {
            "id": image_id,
            "image_id": image_id,
            "category_id": 1,
            "bbox": [5, 5, 15, 15],
            "area": 225,
            "iscrowd": 0,
            "score": 0.4,
        }
        if keypoints:
            annotation["keypoints"] = [10, 10, 2, 15, 15, 2, 0, 0, 0]  # 3 (x, y, v)
            annotation["num_keypoints"] = 2
        annotations.append(annotation)

    results_dir = tmp_path / "video_results"
    results_dir.mkdir()
    annotations_path = results_dir / "annotations.json"
    coco = {
        "info": {"video": video_path.name, "fps": 10, "total_frames": frames},
        "categories": [{"id": 1, "name": "mouse", "supercategory": "common-objects"}],
        "images": images,
        "annotations": annotations,
    }
    annotations_path.write_text(json.dumps(coco))
    return annotations_path, video_path


def test_retrack(tmp_path: Path, monkeypatch) -> None:
    """retrack() replays a cached COCO export through a tracker, skipping detection."""
    monkeypatch.setitem(TRACKERS, "fake", FakeTracker)
    annotations_path, video_path = make_export(tmp_path)

    output = retrack(
        annotations_path,
        video_path,
        "fake",
        output_dir=tmp_path / "output",
        show_progress=False,
    )
    assert output.exists()


def test_retrack_carries_keypoints(tmp_path: Path, monkeypatch) -> None:
    """retrack() passes the export's keypoints through to the tracker and annotator."""
    monkeypatch.setitem(TRACKERS, "fake", FakeTracker)
    annotations_path, video_path = make_export(tmp_path, frames=2, keypoints=True)

    retrack(
        annotations_path,
        video_path,
        "fake",
        output_dir=tmp_path / "output",
        show_progress=False,
    )

    assert len(FakeTracker.seen) == 2
    for detections in FakeTracker.seen:
        assert detections.data["keypoints_xy"].shape == (1, 3, 2)
        assert detections.data["keypoints_visible"].tolist() == [[True, True, False]]


def test_retrack_writes_track_ids(tmp_path: Path, monkeypatch) -> None:
    """retrack() writes the new track ids back into annotations.json."""
    monkeypatch.setitem(TRACKERS, "fake", FakeTracker)
    annotations_path, video_path = make_export(tmp_path, frames=2)
    assert "track_id" not in json.loads(annotations_path.read_text())["annotations"][0]

    retrack(
        annotations_path,
        video_path,
        "fake",
        output_dir=tmp_path / "output",
        show_progress=False,
    )

    coco = json.loads(annotations_path.read_text())
    assert [a["track_id"] for a in coco["annotations"]] == [0, 0]
    for name in ("trajectories.csv", "summary.csv"):
        assert (annotations_path.parent / name).exists()


def test_retrack_skipped_frames(tmp_path: Path, monkeypatch) -> None:
    """With `every` > 1, only exported frames reach the tracker, but the retracked
    video keeps every frame of the source."""
    monkeypatch.setitem(TRACKERS, "fake", FakeTracker)
    annotations_path, video_path = make_export(tmp_path, frames=6, every=2)

    output = retrack(
        annotations_path,
        video_path,
        "fake",
        output_dir=tmp_path / "output",
        show_progress=False,
    )

    assert len(FakeTracker.seen) == 3
    assert sv.VideoInfo.from_video_path(str(output)).total_frames == 6
    assert FakeTracker.seen[0].class_id.tolist() == [0]
    assert FakeTracker.seen[0].xyxy.tolist() == [[5, 5, 20, 20]]
    assert FakeTracker.seen[0].confidence.tolist() == pytest.approx([0.4])


class DroppingTracker(FakeTracker):
    """Confirms nothing, returning a fresh empty Detections as ByteTrack does."""

    def update(self, detections: sv.Detections, frame=None, timestamp=None):
        FakeTracker.seen.append(detections)
        result = sv.Detections.empty()
        result.tracker_id = np.array([], dtype=int)
        return result


def test_retrack_tracker_returns_nothing(tmp_path: Path, monkeypatch) -> None:
    """Detections the tracker drops are written back as unconfirmed."""
    monkeypatch.setitem(TRACKERS, "dropping", DroppingTracker)
    annotations_path, video_path = make_export(tmp_path, frames=2)

    retrack(
        annotations_path,
        video_path,
        "dropping",
        output_dir=tmp_path / "output",
        show_progress=False,
    )

    coco = json.loads(annotations_path.read_text())
    assert [a["track_id"] for a in coco["annotations"]] == [-1, -1]
