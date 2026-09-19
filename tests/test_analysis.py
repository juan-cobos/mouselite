import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mouselite import analysis


def make_export(path: Path, every: int = 1, fps: float | None = 10.0) -> None:
    """Two tracks over 6 sampled frames: track 0 walks right 10 px per frame from (0, 0),
    track 1 sits still at (100, 100) and is missing on the 3rd frame. One unconfirmed
    detection (track_id -1) is thrown in, plus keypoints "nose" and "tail" on track 0.
    """
    images, annotations = [], []
    for t in range(6):
        images.append(
            {
                "id": t + 1,
                "file_name": f"v_{t * every:06d}.jpg",
                "width": 200,
                "height": 200,
                "frame_index": t * every,
            }
        )
        x = 10.0 * t
        annotations.append(
            {
                "id": len(annotations) + 1,
                "image_id": t + 1,
                "category_id": 1,
                "bbox": [x - 5, -5, 10, 10],
                "area": 100,
                "iscrowd": 0,
                "track_id": 0,
                "keypoints": [x + 5, 0, 2, x - 5, 0, 2],  # nose ahead, tail behind
                "num_keypoints": 2,
            }
        )
        if t != 2:
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": t + 1,
                    "category_id": 1,
                    "bbox": [95, 95, 10, 10],
                    "area": 100,
                    "iscrowd": 0,
                    "track_id": 1,
                    "keypoints": [100, 100, 2, 0, 0, 0],  # tail not visible
                    "num_keypoints": 1,
                }
            )
    annotations.append({**annotations[0], "id": len(annotations) + 1, "track_id": -1})
    coco = {
        "info": {"video": "v.mp4", "fps": fps} if fps else {},
        "licenses": [],
        "categories": [{"id": 1, "name": "mouse", "keypoints": ["nose", "tail"]}],
        "images": images,
        "annotations": annotations,
    }
    path.write_text(json.dumps(coco))


def make_segmentation_export(path: Path) -> None:
    """One track over 3 frames with an ellipse mask (major 40, minor 20, tilted 30°),
    given as a polygon on the first two frames and as uncompressed RLE on the third;
    the third frame's mask is a 10x10 square instead.
    """
    theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    ellipse = np.stack([20 * np.cos(theta), 10 * np.sin(theta)], axis=1)
    c, s = np.cos(np.pi / 6), np.sin(np.pi / 6)
    ellipse = ellipse @ np.array([[c, s], [-s, c]]) + [50, 40]
    polygon = [float(v) for v in ellipse.reshape(-1)]

    square = np.zeros((100, 100), dtype=bool)
    square[80:90, 10:20] = True
    flat = square.flatten(order="F").astype(np.int8)  # COCO RLE is column-major
    edges = np.flatnonzero(np.diff(np.concatenate([[0], flat, [0]])))
    counts = np.diff(np.concatenate([[0], edges, [flat.size]])).tolist()

    images = [
        {"id": t + 1, "file_name": f"s_{t:06d}.jpg", "width": 100, "height": 100}
        | {"frame_index": t}
        for t in range(3)
    ]
    annotations = [
        {
            "id": t + 1,
            "image_id": t + 1,
            "category_id": 1,
            "bbox": [30, 25, 40, 30],
            "area": float(np.pi * 20 * 10),
            "segmentation": [polygon],
            "iscrowd": 0,
            "track_id": 0,
        }
        for t in range(2)
    ]
    annotations.append(
        {
            "id": 3,
            "image_id": 3,
            "category_id": 1,
            "bbox": [10, 80, 10, 10],
            "area": 100.0,
            "segmentation": {"counts": counts, "size": [100, 100]},
            "iscrowd": 1,
            "track_id": 0,
        }
    )
    coco = {
        "info": {"fps": 10.0},
        "licenses": [],
        "categories": [{"id": 1, "name": "mouse"}],
        "images": images,
        "annotations": annotations,
    }
    path.write_text(json.dumps(coco))


def test_from_coco(tmp_path: Path) -> None:
    make_export(tmp_path / "annotations.json")
    tracks = analysis.Tracks.from_coco(tmp_path / "annotations.json")

    assert tracks.track_ids.tolist() == [0, 1]  # -1 dropped
    assert tracks.xyxy.shape == (6, 2, 4)
    assert tracks.keypoints.shape == (6, 2, 2, 2)
    assert tracks.keypoint_names == ["nose", "tail"]
    assert tracks.fps == 10.0
    assert tracks.image_size == (200, 200)
    assert tracks.present[:, 0].all()
    assert tracks.present[:, 1].tolist() == [True, True, False, True, True, True]
    assert tracks.masks is None
    np.testing.assert_allclose(tracks.area[:, 0], 100)  # the COCO field, box w * h here
    assert np.isnan(tracks.area[2, 1])
    np.testing.assert_allclose(tracks.centroids[:, 0, 0], np.arange(6) * 10.0)
    np.testing.assert_allclose(tracks.centroids[3, 1], [100, 100])
    assert np.isnan(tracks.keypoint("tail")[0, 1]).all()  # visibility 0 -> NaN
    np.testing.assert_allclose(tracks.time, np.arange(6) / 10)


def test_from_coco_min_frames_and_fps_override(tmp_path: Path) -> None:
    make_export(tmp_path / "annotations.json", fps=None)
    tracks = analysis.Tracks.from_coco(tmp_path / "annotations.json", fps=25, min_frames=6)
    assert tracks.track_ids.tolist() == [0]
    assert tracks.fps == 25
    with pytest.raises(ValueError, match="fps"):
        _ = analysis.Tracks.from_coco(tmp_path / "annotations.json").time


def test_masks(tmp_path: Path) -> None:
    make_segmentation_export(tmp_path / "annotations.json")
    tracks = analysis.Tracks.from_coco(tmp_path / "annotations.json", masks=True)

    assert tracks.keypoints is None
    assert tracks.masks.shape == (3, 1, 100, 100) and tracks.masks.dtype == bool
    assert tracks.area[0, 0] == pytest.approx(np.pi * 200)
    # fillPoly rasterises the outline inclusively, so ~half the perimeter is added
    assert tracks.masks[0, 0].sum() == pytest.approx(np.pi * 200, rel=0.15)
    assert tracks.masks[2, 0].sum() == 100 and tracks.masks[2, 0, 85, 15]  # the RLE

    centroids = analysis.mask_centroids(tracks.masks)
    np.testing.assert_allclose(centroids[0, 0], [50, 40], atol=0.5)
    np.testing.assert_allclose(centroids[2, 0], [14.5, 84.5], atol=0.01)

    major, minor, angle = analysis.mask_axes(tracks.masks)
    assert major[0, 0] == pytest.approx(40, rel=0.1)  # outline thickening, as above
    assert minor[0, 0] == pytest.approx(20, rel=0.1)
    assert angle[0, 0] == pytest.approx(np.pi / 6, abs=0.02)
    assert (major[2, 0] / minor[2, 0]) == pytest.approx(1)  # square

    table = tracks.summary(scale=0.5)
    assert table.loc[0, "mean_area"] == pytest.approx((2 * np.pi * 200 + 100) / 3 * 0.25)
    assert table.loc[0, "mean_elongation"] == pytest.approx((2 * 2 + 1) / 3, rel=0.05)

    unloaded = analysis.Tracks.from_coco(tmp_path / "annotations.json")
    assert unloaded.masks is None and "mean_elongation" not in unloaded.summary()
    assert tracks.select([0]).masks.shape == (3, 1, 100, 100)

    empty = np.zeros((2, 1, 4, 4), dtype=bool)
    assert np.isnan(analysis.mask_centroids(empty)).all()
    assert np.isnan(analysis.mask_axes(empty)[0]).all()


def test_interpolate_and_smooth() -> None:
    xy = np.array([[0, 0], [np.nan, np.nan], [2, 2], [np.nan] * 2, [np.nan] * 2, [5, 5]])
    xy = xy[:, None, :].astype(np.float32)  # (T, 1, 2)

    filled = analysis.interpolate(xy)
    np.testing.assert_allclose(filled[:, 0, 0], [0, 1, 2, 3, 4, 5])

    filled = analysis.interpolate(xy, max_gap=1)
    assert filled[1, 0, 0] == 1 and np.isnan(filled[3:5, 0, 0]).all()

    leading = np.concatenate([np.full((1, 1, 2), np.nan, dtype=np.float32), filled])
    assert np.isnan(analysis.interpolate(leading)[0]).all()  # no extrapolation

    smoothed = analysis.smooth(analysis.interpolate(xy), window=3)
    np.testing.assert_allclose(smoothed[1:-1, 0, 0], [1, 2, 3, 4])  # ends see 2 samples
    assert np.isnan(analysis.smooth(xy, window=3)[1, 0, 0])  # NaNs are kept


def test_speed_and_distance(tmp_path: Path) -> None:
    make_export(tmp_path / "annotations.json", every=2)
    tracks = analysis.Tracks.from_coco(tmp_path / "annotations.json")
    xy = tracks.centroids

    v = analysis.speed(xy, tracks.frame_index, tracks.fps)
    assert v.shape == (6, 2) and np.isnan(v[0]).all()
    np.testing.assert_allclose(v[1:, 0], 10 / (2 / 10))  # 10 px every 2 frames at 10 fps
    assert np.isnan(v[2:4, 1]).all() and v[4, 1] == 0  # around the missing frame

    per_frame = analysis.speed(xy, tracks.frame_index)
    np.testing.assert_allclose(per_frame[1:, 0], 5)
    scaled = analysis.speed(xy, tracks.frame_index, scale=0.5)
    np.testing.assert_allclose(scaled[1:, 0], 2.5)

    np.testing.assert_allclose(analysis.distance_traveled(xy), [50, 0])
    np.testing.assert_allclose(analysis.distance_traveled(xy, scale=0.1), [5, 0])


def test_heading_and_distances(tmp_path: Path) -> None:
    make_export(tmp_path / "annotations.json")
    tracks = analysis.Tracks.from_coco(tmp_path / "annotations.json")
    nose, tail = tracks.keypoint("nose"), tracks.keypoint("tail")

    np.testing.assert_allclose(analysis.heading(tail, nose)[:, 0], 0)  # facing right
    np.testing.assert_allclose(analysis.keypoint_distance(tail, nose)[:, 0], 10)
    assert np.isnan(analysis.heading(tail, nose)[:, 1]).all()  # tail never seen

    d = analysis.pairwise_distance(tracks.centroids)
    assert d.shape == (6, 2, 2)
    np.testing.assert_allclose(d[0, 0, 1], np.hypot(100, 100))
    assert np.isnan(d[2, 0, 1])


def test_in_polygon_bouts_occupancy(tmp_path: Path) -> None:
    make_export(tmp_path / "annotations.json")
    tracks = analysis.Tracks.from_coco(tmp_path / "annotations.json")
    xy = tracks.centroids

    left = [(-10, -10), (25, -10), (25, 10), (-10, 10)]
    inside = analysis.in_polygon(xy, left)
    assert inside[:, 0].tolist() == [True, True, True, False, False, False]
    assert not inside[:, 1].any()
    assert not analysis.in_polygon(xy, [(90, 90), (110, 90), (110, 110), (90, 110)])[2, 1]

    assert analysis.bouts(inside[:, 0]) == [(0, 3)]
    assert analysis.bouts(np.array([1, 0, 1, 1, 0, 1], bool), min_frames=2) == [(2, 4)]
    assert analysis.bouts(np.zeros(4, bool)) == []

    maps = analysis.occupancy(xy, tracks.image_size, bins=2)
    assert maps.shape == (2, 2, 2)
    assert maps[0].sum() == 6 and maps[1].sum() == 5
    assert maps[1, 1, 1] == 5  # bottom-right cell, image indexing


def test_summary_and_tables(tmp_path: Path) -> None:
    make_export(tmp_path / "annotations.json")
    tracks = analysis.Tracks.from_coco(tmp_path / "annotations.json")

    table = tracks.summary(immobile_below=1.0).set_index("track_id")
    assert table.loc[0, "frames"] == 6 and table.loc[1, "frames"] == 5
    assert table.loc[1, "coverage"] == pytest.approx(5 / 6)
    assert table.loc[0, "distance"] == pytest.approx(50)
    assert table.loc[0, "mean_speed"] == pytest.approx(100)  # 10 px/frame at 10 fps
    assert table.loc[0, "duration"] == pytest.approx(0.6)
    assert table.loc[0, "immobile_fraction"] == 0 and table.loc[1, "immobile_fraction"] == 1

    df = tracks.to_dataframe()
    assert len(df) == 11
    expected = {"frame_index", "time", "track_id", "x", "y", "x1", "area", "nose_x"}
    assert expected <= set(df)
    assert df[df.track_id == 1].frame_index.tolist() == [0, 1, 3, 4, 5]

    dlc = tracks.to_deeplabcut(tmp_path / "dlc.csv")
    assert dlc.columns.names == ["scorer", "individuals", "bodyparts", "coords"]
    assert dlc.shape == (6, 2 * 2 * 3)
    assert dlc[("mouselite", "individual1", "tail", "likelihood")].tolist() == [0.0] * 6
    assert dlc[("mouselite", "individual0", "nose", "x")].iloc[3] == 35
    reread = pd.read_csv(tmp_path / "dlc.csv", header=[0, 1, 2, 3], index_col=0)
    assert reread.shape == dlc.shape

    single = tracks.select([0]).to_deeplabcut()
    assert single.columns.names == ["scorer", "bodyparts", "coords"]
