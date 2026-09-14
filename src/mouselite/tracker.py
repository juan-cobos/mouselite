import json
from pathlib import Path

import cv2
import supervision as sv
from tqdm import tqdm
from trackers import (
    BoTSORTTracker,
    ByteTrackTracker,
    CBIoUTracker,
    McByteTracker,
    OCSORTTracker,
    SORTTracker,
)

from mouselite.pipeline import MetaAnnotator

TRACKERS = {
    "botsort": BoTSORTTracker,
    "ocsort": OCSORTTracker,
    "bytetrack": ByteTrackTracker,
    "sort": SORTTracker,
    "cbiou": CBIoUTracker,
    "mcbyte": McByteTracker,
}


def get_tracker(name: str, **kwargs):
    if name not in TRACKERS:
        raise ValueError(f"Unknown tracker {name!r}. Available: {list(TRACKERS)}")
    return TRACKERS[name](**kwargs)


DEFAULT_FPS = 30.0


def read_fps(annotations_path: str | Path, default: float = DEFAULT_FPS) -> float:
    """Frame rate recorded in a COCO export's `info` block, or `default` if it has none."""
    with open(annotations_path) as f:
        info = json.load(f).get("info") or {}
    return float(info.get("fps", default))


def retrack(
    annotations_path: str | Path,
    tracker: str,
    output_dir: str | Path = "output",
    fps: float | None = None,
    show_progress: bool = True,
    **tracker_kwargs,
) -> Path:
    """Replay a previously exported COCO dataset through a tracker, skipping detection.

    `tracker` is a name from `TRACKERS`; `tracker_kwargs` go to its constructor.
    `fps` defaults to the frame rate recorded in the export, else 30.
    """
    annotations_path = Path(annotations_path)
    fps = fps or read_fps(annotations_path)
    dataset = sv.DetectionDataset.from_coco(
        images_directory_path=str(annotations_path.parent / "images"),
        annotations_path=str(annotations_path),
    )
    tracker = get_tracker(tracker, frame_rate=fps, **tracker_kwargs)
    annotator = MetaAnnotator()

    stem = annotations_path.parent.name.removesuffix("_coco")
    target = Path(output_dir) / f"{stem}_retracked.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(dataset.image_paths)
    first_frame = cv2.imread(image_paths[0])
    video_info = sv.VideoInfo(
        width=first_frame.shape[1],
        height=first_frame.shape[0],
        fps=fps,
    )

    with sv.VideoSink(target_path=str(target), video_info=video_info) as sink:
        for image_path in tqdm(
            image_paths,
            disable=not show_progress,
            desc="retracking",
        ):
            frame = cv2.imread(image_path)
            detections = tracker.update(dataset.annotations[image_path], frame=frame)
            sink.write_frame(annotator.annotate(frame.copy(), detections))

    return target
