"""End-to-end video inference pipeline: prediction, tracking and annotation."""

from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import supervision as sv
from supervision.annotators.utils import resolve_color
from supervision.dataset.formats.coco import (
    classes_to_coco_categories,
    detections_to_coco_annotations,
)
from supervision.utils.file import save_json_file
from trackers.core.base import BaseTracker


class MLModel(Protocol):
    def predict(
        self,
        frame: np.ndarray,
        threshold: float,
    ) -> sv.Detections | sv.KeyPoints: ...


def _keypoints_to_detections(keypoints: sv.KeyPoints) -> sv.Detections:
    """Convert KeyPoints to Detections using the model's own box for tracking."""
    if "xyxy" in keypoints.data:
        data = dict(keypoints.data)
        xyxy = np.asarray(data.pop("xyxy"), dtype=np.float32)
        detections = sv.Detections(
            xyxy=xyxy,
            confidence=keypoints.detection_confidence,
            class_id=keypoints.class_id,
            data=data,
        )
    else:
        detections = keypoints.as_detections()

    detections.data["keypoints_xy"] = keypoints.xy
    if keypoints.visible is not None:
        detections.data["keypoints_visible"] = keypoints.visible
    return detections


def _detections_to_coco(
    detections: sv.Detections,
    image_id: int,
    annotation_id: int,
) -> tuple[list[dict], int]:
    """Build COCO annotation dicts for one image, adding `track_id` and `keypoints`."""
    coco_annotations, annotation_id = detections_to_coco_annotations(
        detections,
        image_id,
        annotation_id,
    )

    if detections.tracker_id is not None:
        for annotation, track_id in zip(
            coco_annotations,
            detections.tracker_id,
            strict=True,
        ):
            annotation["track_id"] = int(track_id)

    if "keypoints_xy" in detections.data:
        xy = np.asarray(detections.data["keypoints_xy"], dtype=np.float32)
        visible = detections.data.get("keypoints_visible")
        v = (
            np.where(np.asarray(visible), 2, 0)
            if visible is not None
            else np.full(xy.shape[:2], 2)
        )
        for annotation, keypoints in zip(
            coco_annotations,
            np.concatenate([xy, v[..., None]], axis=2),  # (K, 3): x, y, visibility
            strict=True,
        ):
            annotation["keypoints"] = keypoints.reshape(-1).tolist()
            annotation["num_keypoints"] = int((keypoints[:, 2] > 0).sum())

    return coco_annotations, annotation_id


class MetaAnnotator:
    """Applies box/mask/label/keypoint annotators based on what's in `detections`."""

    def __init__(self):
        self.box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK)
        self.mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.TRACK)
        self.label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.TRACK)
        self.kp_annotator = sv.VertexAnnotator()

    def annotate(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
        if detections.mask is not None:
            frame = self.mask_annotator.annotate(frame, detections)
        frame = self.box_annotator.annotate(frame, detections)
        frame = self.label_annotator.annotate(frame, detections, labels=labels)

        if "keypoints_xy" in detections.data:
            keypoints = sv.KeyPoints(
                xy=np.asarray(detections.data["keypoints_xy"], dtype=np.float32),
                visible=detections.data.get("keypoints_visible"),
            )
            # VertexAnnotator has no per-track color_lookup, so draw one skeleton at a
            # time, resolving each one's color the same way box/label annotators do.
            for i in range(len(keypoints)):
                self.kp_annotator.color = resolve_color(
                    self.box_annotator.color,
                    detections,
                    i,
                    color_lookup=sv.ColorLookup.TRACK,
                )
                frame = self.kp_annotator.annotate(frame, keypoints[i])

        return frame


class Pipeline:
    def __init__(
        self,
        model: MLModel,
        tracker: BaseTracker,
        threshold: float = 0.5,
        nms_threshold: float = 0.5,
        top_k: int | None = None,
        every: int = 1,
    ):
        self.model = model
        self.tracker = tracker
        self.threshold = threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.every = every
        self.annotator = MetaAnnotator()

    def run(
        self,
        video_path: str | Path,
        output_dir: str | Path = "output",
        show: bool = False,
        show_progress: bool = True,
        hud: bool = False,
    ) -> Path:
        """Run inference on `video_path`, writing an annotated video and a COCO export."""
        self.tracker.reset()
        video_path = Path(video_path)
        results_dir = Path(output_dir) / f"{video_path.stem}_results"
        target = results_dir / f"{video_path.stem}_annotated.mp4"
        annotations_path = results_dir / "annotations.json"
        images_dir = results_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        detections = sv.Detections.empty()
        fps_monitor = sv.FPSMonitor() if hud else None
        video_info = sv.VideoInfo.from_video_path(str(video_path))
        coco = {
            "info": {
                "video": video_path.name,
                "fps": video_info.fps,
                "total_frames": video_info.total_frames,
            },
            "licenses": [],
            "categories": classes_to_coco_categories(self.model.class_names),
            "images": [],
            "annotations": [],
        }
        next_annotation_id = 1

        def callback(frame: np.ndarray, frame_idx: int) -> np.ndarray:
            nonlocal detections, next_annotation_id
            if frame_idx % self.every == 0:
                detections = self.model.predict(frame, threshold=self.threshold)
                if isinstance(detections, sv.KeyPoints):
                    detections = _keypoints_to_detections(detections)
                detections = detections.with_nms(threshold=self.nms_threshold)
                if self.top_k is not None and len(detections) > self.top_k:
                    top = detections.confidence.argsort()[::-1][: self.top_k]
                    detections = detections[top]

                # No need to run tracking when there's only one target
                if self.top_k == 1:
                    detections.tracker_id = np.zeros(len(detections), dtype=int)
                else:
                    detections = self.tracker.update(detections, frame=frame)

                image_path = images_dir / f"{video_path.stem}_{frame_idx:06d}.jpg"
                cv2.imwrite(str(image_path), frame)
                image_id = len(coco["images"]) + 1
                coco["images"].append(
                    {
                        "id": image_id,
                        "file_name": image_path.name,
                        "height": frame.shape[0],
                        "width": frame.shape[1],
                        "frame_index": frame_idx,
                    },
                )
                annotations, next_annotation_id = _detections_to_coco(
                    detections,
                    image_id,
                    next_annotation_id,
                )
                coco["annotations"].extend(annotations)

            annotated = self.annotator.annotate(frame.copy(), detections)

            if fps_monitor is not None:
                fps_monitor.tick()
                sv.draw_text(
                    annotated,
                    f"FPS: {fps_monitor.fps:.1f}",
                    text_anchor=sv.Point(x=40, y=20),
                    text_color=sv.Color.WHITE,
                    background_color=sv.Color.BLACK,
                )

            if show:
                cv2.imshow("mouselite", annotated)
                cv2.waitKey(1)
            return annotated

        sv.process_video(
            source_path=str(video_path),
            target_path=str(target),
            callback=callback,
            show_progress=show_progress,
            progress_message=f"mouselite: {video_path.name}",
        )

        if show:
            cv2.destroyAllWindows()

        save_json_file(coco, file_path=annotations_path)

        return target
