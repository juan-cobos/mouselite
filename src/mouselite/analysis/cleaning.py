"""Fill and smooth trajectories before measuring them."""

import numpy as np
import pandas as pd


def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start (inclusive) and stop (exclusive) indices of each run of True in a 1D mask."""
    padded = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    edges = np.diff(padded.astype(np.int8))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def interpolate(xy: np.ndarray, max_gap: int | None = None) -> np.ndarray:
    """Linearly fill NaN gaps along the first (frame) axis, never extrapolating.

    Only gaps of at most `max_gap` frames are filled when it is given. Works on any
    `(T, ...)` array, e.g. `Tracks.xyxy`, `Tracks.keypoints` or `Tracks.centroids`.
    """
    xy = np.asarray(xy, dtype=np.float32)
    flat = xy.reshape(len(xy), -1)
    out = flat.copy()
    t = np.arange(len(flat))
    for j in range(flat.shape[1]):
        col = flat[:, j]
        valid = ~np.isnan(col)
        if valid.sum() < 2:
            continue
        filled = np.interp(t, t[valid], col[valid])
        first, last = t[valid][[0, -1]]
        filled[:first] = filled[last + 1 :] = np.nan
        if max_gap is not None:
            for start, stop in zip(*_runs(~valid), strict=True):
                if stop - start > max_gap:
                    filled[start:stop] = np.nan
        out[:, j] = filled
    return out.reshape(xy.shape)


def smooth(xy: np.ndarray, window: int = 5) -> np.ndarray:
    """Centred rolling median along the frame axis (DeepLabCut's default filter)."""
    xy = np.asarray(xy, dtype=np.float32)
    flat = xy.reshape(len(xy), -1)
    out = pd.DataFrame(flat).rolling(window, center=True, min_periods=1).median()
    out = out.to_numpy(dtype=np.float32, copy=True)
    out[np.isnan(flat)] = np.nan
    return out.reshape(xy.shape)
