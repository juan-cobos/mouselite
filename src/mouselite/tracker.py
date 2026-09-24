import json
from pathlib import Path

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

from mouselite.analysis import Tracks
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


def _rows_to_detections(rows: list[dict]) -> sv.Detections:
    """Rebuild one frame's detections, keypoints included, from its COCO annotations."""
    if not rows:
        return sv.Detections.empty()
    xyxy = np.asarray([row["bbox"] for row in rows], dtype=np.float32)
    xyxy[:, 2:] += xyxy[:, :2]  # COCO boxes are x, y, w, h
    detections = sv.Detections(
        xyxy=xyxy,
        class_id=np.asarray([row["category_id"] - 1 for row in rows]),
        confidence=(
            np.asarray([row["score"] for row in rows], dtype=np.float32)
            if all("score" in row for row in rows)
            else None
        ),
    )
    if rows[0].get("keypoints"):
        xyv = np.asarray([row["keypoints"] for row in rows], dtype=np.float32)
        xyv = xyv.reshape(len(rows), -1, 3)
        detections.data["keypoints_xy"] = xyv[..., :2]
        detections.data["keypoints_visible"] = xyv[..., 2] > 0
    return detections


def retrack(
    annotations_path: str | Path,
    video_path: str | Path,
    tracker: str,
    output_dir: str | Path = "output",
    show_progress: bool = True,
    **tracker_kwargs,
) -> Path:
    """Replay a previously exported COCO dataset through a tracker, skipping detection.

    Frames are read from `video_path` and matched to the export by `frame_index`;
    frames the export skipped (`every` > 1) keep the last frame's detections.
    Writes the retracked video, updates each annotation's `track_id` in place, and
    rewrites `<stem>_trajectories.csv` / `<stem>_summary.csv` beside the annotations.
    `tracker` is a name from `TRACKERS`; `tracker_kwargs` go to its constructor.
    """
    annotations_path = Path(annotations_path)
    with open(annotations_path) as f:
        coco = json.load(f)
    frames: dict[int, list[dict]] = {}
    image_frames = {}
    for image in coco["images"]:
        image_frames[image["id"]] = image["frame_index"]
        frames[image["frame_index"]] = []
    for annotation in coco["annotations"]:
        frames[image_frames[annotation["image_id"]]].append(annotation)
    tracker = get_tracker(tracker, **tracker_kwargs)
    annotator = MetaAnnotator()

    stem = annotations_path.parent.name.removesuffix("_results")
    target = Path(output_dir) / f"{stem}_retracked.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)

    video_info = sv.VideoInfo.from_video_path(str(video_path))
    detections = sv.Detections.empty()
    detections.tracker_id = np.array([], dtype=int)

    with sv.VideoSink(target_path=str(target), video_info=video_info) as sink:
        for frame_idx, frame in enumerate(
            tqdm(
                sv.get_video_frames_generator(str(video_path)),
                total=video_info.total_frames,
                disable=not show_progress,
                desc="retracking",
            ),
        ):
            rows = frames.get(frame_idx)
            if rows is not None:
                detections = _rows_to_detections(rows)
                # Trackers may drop or reorder detections; carry the row index through.
                detections.data["row"] = np.arange(len(rows))
                detections = tracker.update(detections, frame=frame)

                for row in rows:
                    row["track_id"] = -1  # same as the trackers use for unconfirmed
                for row, track_id in zip(
                    # an empty result from the tracker is a fresh Detections, no data
                    detections.data.get("row", []),
                    detections.tracker_id,
                    strict=True,
                ):
                    rows[row]["track_id"] = int(track_id)

            sink.write_frame(annotator.annotate(frame.copy(), detections))

    save_json_file(coco, file_path=str(annotations_path))
    tracks = Tracks.from_coco(annotations_path)
    prefix = annotations_path.parent / Path(video_path).stem
    tracks.to_dataframe().to_csv(f"{prefix}_trajectories.csv", index=False)
    tracks.summary().to_csv(f"{prefix}_summary.csv", index=False)
    return target
