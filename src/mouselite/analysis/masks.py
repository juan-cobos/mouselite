"""Posture from the `(T, N, H, W)` masks of a segmentation export."""

import numpy as np


def _mask_moments(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixel count `(T, N)`, centroid `(T, N, 2)` and covariance `(T, N, 2, 2)` of masks."""
    masks = np.asarray(masks)
    ys, xs = np.mgrid[: masks.shape[-2], : masks.shape[-1]].astype(np.float32)
    weights = np.stack([np.ones_like(xs), xs, ys, xs * xs, ys * ys, xs * ys])
    # Sum each weight over the mask pixels, a few frames at a time to bound the float
    # copy of the masks (a full-resolution recording is gigabytes even as bool).
    chunk = max(1, 2**26 // int(np.prod(masks.shape[1:])))
    sums = np.concatenate(
        [
            np.einsum("...hw,mhw->...m", masks[i : i + chunk].astype(np.float32), weights)
            for i in range(0, len(masks), chunk)
        ]
    )
    count, sx, sy, sxx, syy, sxy = np.moveaxis(sums, -1, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cx, cy = sx / count, sy / count
        cxx, cyy, cxy = sxx / count - cx * cx, syy / count - cy * cy, sxy / count - cx * cy
    centroid = np.stack([cx, cy], axis=-1)
    cov = np.stack([np.stack([cxx, cxy], -1), np.stack([cxy, cyy], -1)], axis=-2)
    return count, centroid, cov


def mask_centroids(masks: np.ndarray) -> np.ndarray:
    """(T, N, 2) centre of mass of each `(T, N, H, W)` mask; NaN where the mask is empty."""
    return _mask_moments(masks)[1]


def mask_axes(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Body shape from each `(T, N, H, W)` mask, as three `(T, N)` arrays.

    Returns the lengths of the major and minor axes of the mask's equivalent ellipse,
    in pixels, and the angle of the major axis in radians (`0` = horizontal, positive
    towards the bottom of the image, in `[-pi/2, pi/2]`). `major / minor` is how
    stretched the animal is: high when running or rearing sideways, near 1 when
    hunched or grooming. The angle is the body axis without a head/tail sign; use
    `heading` on keypoints when the direction matters. NaN where the mask is empty.
    """
    _, _, cov = _mask_moments(masks)
    cxx, cyy, cxy = cov[..., 0, 0], cov[..., 1, 1], cov[..., 0, 1]
    half_trace = (cxx + cyy) / 2
    spread = np.sqrt(((cxx - cyy) / 2) ** 2 + cxy**2)
    # Eigenvalues are the variances along the axes; 4 * sqrt(var) is the ellipse's
    # full axis length, so that a filled ellipse gives back its own axes.
    major = 4 * np.sqrt(np.maximum(half_trace + spread, 0))
    minor = 4 * np.sqrt(np.maximum(half_trace - spread, 0))
    angle = 0.5 * np.arctan2(2 * cxy, cxx - cyy)
    return major, minor, angle
