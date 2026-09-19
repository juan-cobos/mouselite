"""The `Tracks` grid a COCO export is loaded into, and the tables it summarises to."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import supervision as sv
from supervision.detection.utils.converters import rle_to_mask

from mouselite.analysis.kinematics import distance_traveled, speed
from mouselite.analysis.masks import mask_axes

UNCONFIRMED = -1  # track_id the trackers give detections they never confirmed


@dataclass
class Tracks:
    """Everything in a COCO export laid out on a (frame, track) grid."""

    frame_index: np.ndarray  # (T,) int
    track_ids: np.ndarray  # (N,) int
    xyxy: np.ndarray  # (T, N, 4) float32, NaN when absent
    area: np.ndarray  # (T, N) float32, NaN when absent
    keypoints: np.ndarray | None = None  # (T, N, K, 2) float32, NaN when absent
    keypoint_names: list[str] = field(default_factory=list)
    masks: np.ndarray | None = None  # (T, N, H, W) bool, all False when absent
    fps: float | None = None
    image_size: tuple[int, int] | None = None  # (width, height)
    video: str | None = None

    @property
    def n_frames(self) -> int:
        return len(self.frame_index)

    @property
    def n_tracks(self) -> int:
        return len(self.track_ids)

    @property
    def time(self) -> np.ndarray:
        """(T,) seconds since the start of the video. Needs `fps`."""
        if self.fps is None:
            raise ValueError("Tracks has no fps: pass one to `Tracks.from_coco(..., fps=)`")
        return self.frame_index / self.fps

    @property
    def present(self) -> np.ndarray:
        """(T, N) whether each track was detected on each frame."""
        return ~np.isnan(self.xyxy[..., 0])

    @property
    def centroids(self) -> np.ndarray:
        """(T, N, 2) box centres."""
        return (self.xyxy[..., :2] + self.xyxy[..., 2:]) / 2

    def keypoint(self, name: str | int) -> np.ndarray:
        """(T, N, 2) one named (or indexed) keypoint of every track."""
        if self.keypoints is None:
            raise ValueError("this export has no keypoints")
        k = self.keypoint_names.index(name) if isinstance(name, str) else name
        return self.keypoints[:, :, k]

    def select(self, track_ids) -> "Tracks":
        """The same grid restricted to `track_ids`, in the order given."""
        idx = [int(np.flatnonzero(self.track_ids == t)[0]) for t in track_ids]
        return Tracks(
            frame_index=self.frame_index,
            track_ids=self.track_ids[idx],
            xyxy=self.xyxy[:, idx],
            area=self.area[:, idx],
            keypoints=None if self.keypoints is None else self.keypoints[:, idx],
            keypoint_names=self.keypoint_names,
            masks=None if self.masks is None else self.masks[:, idx],
            fps=self.fps,
            image_size=self.image_size,
            video=self.video,
        )

    def summary(
        self,
        scale: float = 1.0,
        immobile_below: float | None = None,
    ) -> pd.DataFrame:
        """One row per track: how long it was seen, how far and how fast it moved."""
        xy = self.centroids
        v = speed(xy, self.frame_index, self.fps, scale)
        present = self.present
        rows = []
        for n, track_id in enumerate(self.track_ids):
            seen = np.flatnonzero(present[:, n])
            moving = v[~np.isnan(v[:, n]), n]  # empty if never seen on consecutive frames
            row = {
                "track_id": int(track_id),
                "frames": len(seen),
                "first_frame": int(self.frame_index[seen[0]]),
                "last_frame": int(self.frame_index[seen[-1]]),
                "coverage": len(seen) / self.n_frames,
                "distance": float(distance_traveled(xy[:, n], scale)),
                "mean_speed": float(moving.mean()) if len(moving) else np.nan,
                "max_speed": float(moving.max()) if len(moving) else np.nan,
                "mean_area": float(np.nanmean(self.area[:, n])) * scale**2,
            }
            if self.masks is not None:
                major, minor, _ = mask_axes(self.masks[:, n : n + 1])
                row["mean_elongation"] = float(np.nanmean(major / minor))
            if self.fps is not None:
                row["duration"] = (row["last_frame"] - row["first_frame"] + 1) / self.fps
            if immobile_below is not None:
                row["immobile_fraction"] = (
                    float((moving < immobile_below).mean()) if len(moving) else np.nan
                )
            rows.append(row)
        return pd.DataFrame(rows)

    def to_dataframe(self) -> pd.DataFrame:
        """Long table with one row per (frame, track) the animal was seen on."""
        t, n = np.nonzero(self.present)
        table = {"frame_index": self.frame_index[t]}
        if self.fps is not None:
            table["time"] = self.time[t]
        table["track_id"] = self.track_ids[n]
        centroids = self.centroids[t, n]
        table["x"], table["y"] = centroids[:, 0], centroids[:, 1]
        for i, col in enumerate(("x1", "y1", "x2", "y2")):
            table[col] = self.xyxy[t, n, i]
        table["area"] = self.area[t, n]
        if self.keypoints is not None:
            for k, name in enumerate(self.keypoint_names):
                table[f"{name}_x"] = self.keypoints[t, n, k, 0]
                table[f"{name}_y"] = self.keypoints[t, n, k, 1]
        return pd.DataFrame(table)

    def to_deeplabcut(
        self,
        path: str | Path | None = None,
        scorer: str = "mouselite",
    ) -> pd.DataFrame:
        """DeepLabCut-style prediction table, the format most downstream tools read.

        Columns are `(scorer, [individuals,] bodyparts, coords)` with `coords` in
        `x, y, likelihood`, one row per frame indexed by `frame_index`. The
        `individuals` level is only added for multi-animal exports, as DLC does. Without
        keypoints the box centre is written as a single `centroid` bodypart. MouseLite
        exports no per-keypoint confidence, so `likelihood` is 1 where a point was seen
        and 0 where it was not. Written to `path` as CSV (or HDF5 for `.h5`) when given.
        """
        if self.keypoints is not None:
            xy, bodyparts = self.keypoints, self.keypoint_names
        else:
            xy, bodyparts = self.centroids[:, :, None], ["centroid"]
        likelihood = (~np.isnan(xy[..., 0])).astype(np.float32)
        values = np.concatenate([np.nan_to_num(xy), likelihood[..., None]], axis=-1)

        individuals = [f"individual{t}" for t in self.track_ids]
        levels = [[scorer], individuals, bodyparts, ["x", "y", "likelihood"]]
        names = ["scorer", "individuals", "bodyparts", "coords"]
        if self.n_tracks == 1:
            levels.pop(1), names.pop(1)
        df = pd.DataFrame(
            values.reshape(self.n_frames, -1),
            index=self.frame_index,
            columns=pd.MultiIndex.from_product(levels, names=names),
        )
        if path is not None:
            path = Path(path)
            if path.suffix == ".h5":
                df.to_hdf(path, key="df_with_missing", mode="w")
            else:
                df.to_csv(path)
        return df

    @classmethod
    def from_coco(
        cls,
        annotations_path: str | Path,
        fps: float | None = None,
        min_frames: int = 1,
        masks: bool = False,
    ) -> "Tracks":
        """Read a MouseLite `annotations.json`.

        Unconfirmed detections (`track_id == -1`) are dropped, as are tracks seen on
        fewer than `min_frames` frames. `fps` overrides the one recorded in the export's
        `info`. `masks=True` also decodes a segmentation export's masks into a
        `(T, N, H, W)` bool array: one byte per pixel per track per frame, so a long
        recording at full resolution takes gigabytes; run with `--every` or `select`
        fewer tracks if that is too much.
        """
        with open(annotations_path) as f:
            coco = json.load(f)

        images = sorted(coco["images"], key=lambda im: im.get("frame_index", im["id"]))
        frame_index = np.array([im.get("frame_index", im["id"] - 1) for im in images])
        row_of_image = {im["id"]: t for t, im in enumerate(images)}
        size_of_image = {im["id"]: (im["width"], im["height"]) for im in images}

        annotations = [
            a for a in coco["annotations"] if a.get("track_id", UNCONFIRMED) != UNCONFIRMED
        ]
        counts = pd.Series([a["track_id"] for a in annotations]).value_counts()
        track_ids = np.array(sorted(counts.index[counts >= min_frames]), dtype=int)
        col_of_track = {t: n for n, t in enumerate(track_ids)}

        shape = (len(images), len(track_ids))
        n_keypoints = max((len(a.get("keypoints", [])) for a in annotations), default=0)
        n_keypoints //= 3
        xyxy = np.full((*shape, 4), np.nan, dtype=np.float32)
        area = np.full(shape, np.nan, dtype=np.float32)
        keypoints = (
            np.full((*shape, n_keypoints, 2), np.nan, dtype=np.float32)
            if n_keypoints
            else None
        )
        image_size = (images[0]["width"], images[0]["height"]) if images else None
        mask_array = (
            np.zeros((*shape, image_size[1], image_size[0]), dtype=bool)
            if masks and image_size
            else None
        )
        for a in annotations:
            if a["track_id"] not in col_of_track:
                continue
            t, n = row_of_image[a["image_id"]], col_of_track[a["track_id"]]
            x, y, w, h = a["bbox"]
            xyxy[t, n] = (x, y, x + w, y + h)
            area[t, n] = a.get("area", w * h)
            if keypoints is not None and a.get("keypoints"):
                xyv = np.asarray(a["keypoints"], dtype=np.float32).reshape(-1, 3)
                xy = np.where(xyv[:, 2:] > 0, xyv[:, :2], np.nan)
                keypoints[t, n, : len(xy)] = xy
            if mask_array is not None and a.get("segmentation"):
                mask_array[t, n] = _decode_segmentation(
                    a["segmentation"], size_of_image[a["image_id"]]
                )

        category = next(iter(coco.get("categories", [])), {})
        keypoint_names = list(category.get("keypoints") or [])
        if keypoints is not None and len(keypoint_names) != n_keypoints:
            keypoint_names = [f"kp{k}" for k in range(n_keypoints)]

        info = coco.get("info", {})
        return cls(
            frame_index=frame_index,
            track_ids=track_ids,
            xyxy=xyxy,
            area=area,
            keypoints=keypoints,
            keypoint_names=keypoint_names,
            masks=mask_array,
            fps=fps if fps is not None else info.get("fps"),
            image_size=image_size,
            video=info.get("video"),
        )


def _decode_segmentation(segmentation, resolution_wh: tuple[int, int]) -> np.ndarray:
    """(H, W) bool mask from a COCO `segmentation`: polygons, or an uncompressed RLE."""
    if isinstance(segmentation, dict):
        return rle_to_mask(segmentation["counts"], resolution_wh)
    mask = np.zeros(resolution_wh[::-1], dtype=bool)
    for polygon in segmentation:
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if len(points) >= 3:
            mask |= sv.polygon_to_mask(points, resolution_wh).astype(bool)
    return mask
