import json
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np
import pandas as pd
import pytest
import supervision as sv
from trackers.core.base import BaseTracker

from mouselite.pipeline import Pipeline


class FakeKeypointModel:
    class_names: ClassVar[list[str]] = ["mouse"]

    def predict(self, frame: np.ndarray, threshold: float) -> sv.KeyPoints:
        return sv.KeyPoints(
            xy=np.array([[[7, 7], [10, 10], [15, 15]]], dtype=np.float32),
            class_id=np.array([0]),
            detection_confidence=np.array([0.9], dtype=np.float32),
            keypoint_confidence=np.array([[0.9, 0.8, 0.7]], dtype=np.float32),
            visible=np.array([[True, True, True]]),
            data={"xyxy": np.array([[5.0, 5.0, 20.0, 20.0]], dtype=np.float32)},
        )


class EmptyEveryOtherModel(FakeKeypointModel):
    """Returns one detection on even calls and nothing on odd calls."""

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, frame: np.ndarray, threshold: float) -> sv.KeyPoints:
        self.calls += 1
        if self.calls % 2 == 0:
            return sv.KeyPoints.empty()
        return super().predict(frame, threshold)


class FakeTracker(BaseTracker):
    def __init__(self) -> None:
        self.updates = 0

    def reset(self) -> None:
        pass

    def update(
        self,
        detections: sv.Detections,
        frame=None,
        timestamp=None,
    ) -> sv.Detections:
        self.updates += 1
        detections.tracker_id = np.arange(len(detections))
        return detections


def make_video(path: Path, frames: int = 6, size: int = 64) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (size, size))
    for _ in range(frames):
        writer.write(np.random.randint(0, 255, (size, size, 3), dtype=np.uint8))
    writer.release()


def test_run_keypoints(tmp_path: Path) -> None:
    """Fake keypoints model + tracker, exercising run() end to end."""
    video_path = tmp_path / "smoke.mp4"
    make_video(video_path)

    pipeline = Pipeline(model=FakeKeypointModel(), tracker=FakeTracker(), every=2)
    annotated_video = pipeline.run(
        video_path,
        output_dir=tmp_path / "output",
        show_progress=False,
    )
    assert annotated_video.exists()

    annotations_path = tmp_path / "output" / "smoke_results" / "annotations.json"
    coco = json.loads(annotations_path.read_text())
    assert coco["images"]
    assert "frame_index" in coco["images"][0]
    assert "keypoints" in coco["annotations"][0]
    assert coco["annotations"][0]["track_id"] == 0
    assert coco["annotations"][0]["score"] == pytest.approx(0.9)

    trajectories = pd.read_csv(tmp_path / "output" / "smoke_results" / "trajectories.csv")
    assert len(trajectories) == len(coco["annotations"])
    assert {"frame_index", "time", "track_id", "x", "y"} <= set(trajectories.columns)
    summary = pd.read_csv(tmp_path / "output" / "smoke_results" / "summary.csv")
    assert summary.track_id.tolist() == [0]
    assert {"frames", "distance", "mean_speed", "duration"} <= set(summary.columns)


def test_run_top_k_one_skips_tracking(tmp_path: Path) -> None:
    """top_k=1 never calls the tracker, and frames with no detections stay empty."""
    video_path = tmp_path / "smoke.mp4"
    make_video(video_path)

    tracker = FakeTracker()
    pipeline = Pipeline(model=EmptyEveryOtherModel(), tracker=tracker, top_k=1)
    pipeline.run(video_path, output_dir=tmp_path / "output", show_progress=False)

    assert tracker.updates == 0

    annotations_path = tmp_path / "output" / "smoke_results" / "annotations.json"
    coco = json.loads(annotations_path.read_text())
    assert len(coco["images"]) == 6
    # one annotation per detected frame, none for the empty ones
    assert len(coco["annotations"]) == 3
    assert {a["track_id"] for a in coco["annotations"]} == {0}
    assert {a["image_id"] for a in coco["annotations"]} == {1, 3, 5}
