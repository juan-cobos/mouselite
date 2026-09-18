import json
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from supervision.utils.file import save_json_file
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
    "sort": SORTTracker,
    "botsort": BoTSORTTracker,
    "ocsort": OCSORTTracker,
    "bytetrack": ByteTrackTracker,
    "cbiou": CBIoUTracker,
    "mcbyte": McByteTracker,
}


def get_tracker(name: str, **kwargs):
    if name not in TRACKERS:
        raise ValueError(f"Unknown tracker {name!r}. Available: {list(TRACKERS)}")
    return TRACKERS[name](**kwargs)


def retrack(
    annotations_path: str | Path,
    tracker: str,
    output_dir: str | Path = "output",
    show_progress: bool = True,
    **tracker_kwargs,
) -> Path:
    """Replay a previously exported COCO dataset through a tracker, skipping detection.

    Writes the retracked video and updates each annotation's `track_id` in place.
    `tracker` is a name from `TRACKERS`; `tracker_kwargs` go to its constructor.
    """
    annotations_path = Path(annotations_path)
    dataset = sv.DetectionDataset.from_coco(
        images_directory_path=str(annotations_path.parent / "images"),
        annotations_path=str(annotations_path),
    )
    with open(annotations_path) as f:
        coco = json.load(f)
    file_names = {image["id"]: image["file_name"] for image in coco["images"]}
    annotations: dict[str, list[dict]] = {}
    for annotation in coco["annotations"]:
        annotations.setdefault(file_names[annotation["image_id"]], []).append(annotation)
    tracker = get_tracker(tracker, **tracker_kwargs)
    annotator = MetaAnnotator()

    stem = annotations_path.parent.name.removesuffix("_results")
    target = Path(output_dir) / f"{stem}_retracked.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(dataset.image_paths)
    first_frame = cv2.imread(image_paths[0])
    video_info = sv.VideoInfo(
        width=first_frame.shape[1],
        height=first_frame.shape[0],
        fps=30,
    )

    with sv.VideoSink(target_path=str(target), video_info=video_info) as sink:
        for image_path in tqdm(
            image_paths,
            disable=not show_progress,
            desc="retracking",
        ):
            frame = cv2.imread(image_path)
            rows = annotations.get(Path(image_path).name, [])
            detections = dataset.annotations[image_path]
            if rows and rows[0].get("keypoints"):
                xyv = np.asarray([row["keypoints"] for row in rows], dtype=np.float32)
                xyv = xyv.reshape(len(rows), -1, 3)
                detections.data["keypoints_xy"] = xyv[..., :2]
                detections.data["keypoints_visible"] = xyv[..., 2] > 0
            # Trackers may drop or reorder detections; carry the row index through.
            detections.data["row"] = np.arange(len(rows))
            detections = tracker.update(detections, frame=frame)
            sink.write_frame(annotator.annotate(frame.copy(), detections))

            for row in rows:
                row["track_id"] = -1  # same as the trackers use for unconfirmed
            for row, track_id in zip(
                detections.data["row"],
                detections.tracker_id,
                strict=True,
            ):
                rows[row]["track_id"] = int(track_id)

    save_json_file(coco, file_path=str(annotations_path))
    return target
