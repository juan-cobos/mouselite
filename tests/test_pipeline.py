import json
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np
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

    annotations_path = tmp_path / "output" / "smoke_coco" / "annotations.json"
    coco = json.loads(annotations_path.read_text())
    assert coco["images"]
    assert "frame_index" in coco["images"][0]
    assert "keypoints" in coco["annotations"][0]
    assert coco["annotations"][0]["track_id"] == 0
