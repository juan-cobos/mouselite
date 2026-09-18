"""Speed, distance and orientation from `(T, ..., 2)` point arrays."""

import numpy as np


def _dt(n_steps: int, frame_index: np.ndarray | None, fps: float | None) -> np.ndarray:
    """(n_steps,) time between consecutive frames, in seconds or in frames."""
    if frame_index is None:
        dt = np.ones(n_steps, dtype=np.float32)
    else:
        dt = np.diff(frame_index).astype(np.float32)
    return dt / fps if fps else dt


def speed(
    xy: np.ndarray,
    frame_index: np.ndarray | None = None,
    fps: float | None = None,
    scale: float = 1.0,
) -> np.ndarray:
    """Speed of each point from one frame to the next, shape `xy.shape[:-1]`."""
    xy = np.asarray(xy, dtype=np.float32)
    step = np.linalg.norm(np.diff(xy, axis=0), axis=-1) * scale
    dt = _dt(len(step), frame_index, fps).reshape((-1,) + (1,) * (step.ndim - 1))
    first = np.full((1, *step.shape[1:]), np.nan, dtype=np.float32)
    return np.concatenate([first, step / dt])


def distance_traveled(xy: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Total path length of each point over the recording, shape `xy.shape[1:-1]`."""
    step = np.linalg.norm(np.diff(np.asarray(xy, dtype=np.float32), axis=0), axis=-1)
    return np.nansum(step, axis=0) * scale


def heading(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle in radians of the vector from point `a` to point `b`, shape `a.shape[:-1]`.

    Image coordinates, so 0 points right and pi/2 points down. Pass e.g.
    `tracks.keypoint("tail_base"), tracks.keypoint("nose")` for body orientation.
    """
    d = np.asarray(b, dtype=np.float32) - np.asarray(a, dtype=np.float32)
    return np.arctan2(d[..., 1], d[..., 0])


def keypoint_distance(a: np.ndarray, b: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Distance between two points frame by frame (e.g. body length), `a.shape[:-1]`."""
    d = np.asarray(b, dtype=np.float32) - np.asarray(a, dtype=np.float32)
    return np.linalg.norm(d, axis=-1) * scale


def pairwise_distance(xy: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """(T, N, N) distance between every pair of tracks, from `(T, N, 2)` points."""
    xy = np.asarray(xy, dtype=np.float32)
    return np.linalg.norm(xy[:, :, None] - xy[:, None, :], axis=-1) * scale
