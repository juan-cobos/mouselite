"""Downstream analysis of a MouseLite COCO export: trajectories, kinematics, summaries."""

from mouselite.analysis.cleaning import interpolate, smooth
from mouselite.analysis.kinematics import (
    distance_traveled,
    heading,
    keypoint_distance,
    pairwise_distance,
    speed,
)
from mouselite.analysis.masks import mask_axes, mask_centroids
from mouselite.analysis.space import bouts, in_polygon, occupancy
from mouselite.analysis.tracks import Tracks

__all__ = [
    "Tracks",
    "bouts",
    "distance_traveled",
    "heading",
    "in_polygon",
    "interpolate",
    "keypoint_distance",
    "mask_axes",
    "mask_centroids",
    "occupancy",
    "pairwise_distance",
    "smooth",
    "speed",
]
