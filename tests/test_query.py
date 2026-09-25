import numpy as np
import pytest
import supervision as sv

from mouselite.query import QueryTracker
from mouselite.tracker import get_tracker


def boxes(*centres: tuple[float, float], confidence=None, keypoints=None) -> sv.Detections:
    """10x10 boxes around each (x, y) centre."""
    xy = np.asarray(centres, dtype=np.float32).reshape(-1, 2)
    detections = sv.Detections(
        xyxy=np.concatenate([xy - 5, xy + 5], axis=1),
        confidence=None if confidence is None else np.asarray(confidence, np.float32),
        class_id=np.zeros(len(xy), dtype=int),
    )
    if keypoints is not None:
        detections.data["keypoints_xy"] = np.asarray(keypoints, dtype=np.float32)
    return detections


def test_relational_query_orders_every_frame() -> None:
    tracker = QueryTracker("id0.y > id1.y")
    assert tracker.k == 2
    assert tracker.update(boxes((0, 100), (0, 10))).tracker_id.tolist() == [0, 1]
    assert tracker.update(boxes((0, 10), (0, 100))).tracker_id.tolist() == [1, 0]


def test_ties_and_violations_are_undecided() -> None:
    tracker = QueryTracker("id0.y > id1.y")
    assert tracker.update(boxes((0, 50), (5, 50))).tracker_id.tolist() == [-1, -1]
    tracker = QueryTracker("id0.y > 300 and id1.y > 300")
    assert tracker.update(boxes((0, 10), (0, 20))).tracker_id.tolist() == [-1, -1]


def test_lone_detection_uses_single_animal_clauses() -> None:
    tracker = QueryTracker("id0.y > 300 and id1.y <= 300")
    assert tracker.update(boxes((0, 400))).tracker_id.tolist() == [0]
    assert tracker.update(boxes((0, 100))).tracker_id.tolist() == [1]
    # A relational clause can't place a lone detection.
    assert QueryTracker("id0.y > id1.y").update(boxes((0, 1))).tracker_id.tolist() == [-1]


def test_empty_frame() -> None:
    assert QueryTracker("id0.y > id1.y").update(sv.Detections.empty()).tracker_id.size == 0


def test_extra_detections_keep_the_most_confident() -> None:
    tracker = QueryTracker("id0.x < id1.x")
    detections = boxes((50, 0), (0, 0), (100, 0), confidence=[0.9, 0.2, 0.8])
    assert tracker.update(detections).tracker_id.tolist() == [0, -1, 1]


def test_detection_fields_abs_and_indexing() -> None:
    tracker = QueryTracker("id0.area > id1.area and abs(id0.xyxy[0] - id1.xyxy[0]) > 1")
    detections = sv.Detections(
        xyxy=np.array([[0, 0, 10, 10], [20, 0, 50, 30]], dtype=np.float32),
        class_id=np.zeros(2, dtype=int),
    )
    assert tracker.update(detections).tracker_id.tolist() == [1, 0]

    detections = boxes((0, 0), (5, 10), confidence=[0.4, 0.9])
    detections.class_id = np.array([1, 0])
    detections.data["class_name"] = np.array(["female", "male"])
    for query in (
        "id0.class_id == 0 and id1.class_id == 1",
        "id0.class_name == 'male' and id1.class_name != 'male'",
        "id0.confidence > id1.confidence",
        "id0.xyxy[-1] > id1.xyxy[-1]",
    ):
        assert QueryTracker(query).update(detections).tracker_id.tolist() == [1, 0], query


def test_keypoints_index_and_invisible_are_nan() -> None:
    tracker = QueryTracker("id0.keypoints_xy[1, 0] < id1.keypoints_xy[1][0]")
    detections = boxes((0, 0), (0, 0), keypoints=[[[0, 0], [9, 0]], [[0, 0], [3, 0]]])
    assert tracker.update(detections).tracker_id.tolist() == [1, 0]

    detections.data["keypoints_visible"] = np.array([[True, True], [True, False]])
    assert tracker.update(detections).tracker_id.tolist() == [-1, -1]


def test_runtime_field_errors() -> None:
    with pytest.raises(ValueError, match=r"keypoints_xy.*Available: area"):
        QueryTracker("id0.keypoints_xy[0, 0] > 1").update(boxes((0, 0)))
    for query in ("id0.xyxy > id1.xyxy", "id0.y > 'a'"):
        with pytest.raises(ValueError, match="Index array fields"):
            QueryTracker(query).update(boxes((0, 0), (1, 1)))


@pytest.mark.parametrize(
    "query",
    [
        "foo.y > 1",
        "id0.mask > 1",
        "id0.tracker_id == 0",
        "id0.__class__ == 1",
        "id0.xyxy[id1.y] > 1",
        "id0.xyxy[0:2] > 1",
        "(1, 2) == id0.y",
        "id0 > id1",
        "len(id0.y) > 1",
        "abs > 1",
        "id0.y.__class__",
        "__import__('os')",
        "[id0.y][0] > 1",
        "1 > 0",
    ],
)
def test_rejects_unsafe_or_invalid_queries(query: str) -> None:
    with pytest.raises((ValueError, SyntaxError)):
        QueryTracker(query)


def test_get_tracker_query() -> None:
    assert isinstance(get_tracker(None, query="id0.y > id1.y"), QueryTracker)
    with pytest.raises(ValueError, match="no tracker options"):
        get_tracker(None, query="id0.y > id1.y", lost_track_buffer=5)
