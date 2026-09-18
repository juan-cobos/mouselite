"""Where animals were and for how long: zones, bouts, occupancy."""

import numpy as np

from mouselite.analysis.cleaning import _runs


def in_polygon(xy: np.ndarray, polygon) -> np.ndarray:
    """Whether each `(..., 2)` point lies inside `polygon` (a list of `(x, y)` vertices).

    Missing points are False. Use a rectangle `[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]`
    for a zone of the arena and `bouts` on the result for visits.
    """
    poly = np.asarray(polygon, dtype=np.float32)
    xy = np.asarray(xy, dtype=np.float32)
    x, y = xy[..., 0], xy[..., 1]
    inside = np.zeros(x.shape, dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        for (x0, y0), (x1, y1) in zip(poly, np.roll(poly, 1, axis=0), strict=True):
            crosses = (y0 > y) != (y1 > y)
            inside ^= crosses & (x < (x1 - x0) * (y - y0) / (y1 - y0) + x0)
    inside[np.isnan(x)] = False
    return inside


def bouts(mask: np.ndarray, min_frames: int = 1) -> list[tuple[int, int]]:
    """`(start, stop)` row ranges of each run of True in a 1D mask, `stop` exclusive.

    Rows are positions along the frame axis, not `frame_index` values: index
    `tracks.frame_index` (or `tracks.time`) with them. Freezing, for instance, is
    `bouts(speed(tracks.centroids, ...)[:, n] < threshold, min_frames=fps)`.
    """
    return [
        (int(start), int(stop))
        for start, stop in zip(*_runs(mask), strict=True)
        if stop - start >= min_frames
    ]


def occupancy(
    xy: np.ndarray,
    image_size: tuple[int, int],
    bins: int | tuple[int, int] = 32,
) -> np.ndarray:
    """(N, rows, cols) heatmap of frames spent in each cell, from `(T, N, 2)` points.

    Cells cover the `(width, height)` image; index it like an image (`[row, col]`).
    """
    xy = np.asarray(xy, dtype=np.float32)
    width, height = image_size
    ny, nx = (bins, bins) if isinstance(bins, int) else bins
    maps = []
    for n in range(xy.shape[1]):
        pts = xy[:, n][~np.isnan(xy[:, n, 0])]
        hist, _, _ = np.histogram2d(
            pts[:, 1], pts[:, 0], bins=(ny, nx), range=[[0, height], [0, width]]
        )
        maps.append(hist)
    return np.stack(maps) if maps else np.zeros((0, ny, nx))
