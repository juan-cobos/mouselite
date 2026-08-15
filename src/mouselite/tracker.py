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
from trackers.core.base import BaseTracker

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


def retrack(
    annotations_path: str | Path,
    tracker: BaseTracker,
    output_dir: str | Path = "output",
    fps: float = 10.0,
    show_progress: bool = True,
) -> Path:
    """Replay a previously exported COCO dataset through `tracker`, skipping detection."""
    annotations_path = Path(annotations_path)
    dataset = sv.DetectionDataset.from_coco(
        images_directory_path=str(annotations_path.parent / "images"),
        annotations_path=str(annotations_path),
    )
    tracker.reset()
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
