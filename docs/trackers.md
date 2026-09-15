# Trackers

Detection tells you *where* the animals are in each frame; tracking tells you *which*
is which across frames. MouseLite delegates tracking entirely to Roboflow's
[`trackers`](https://github.com/roboflow/trackers) package and exposes six of its
algorithms by name. It adds nothing on top: `get_tracker(name, **kwargs)` is
`TRACKERS[name](**kwargs)`.

| Name | Class | Uses | Notes |
| ---- | ----- | ---- | ----- |
| `bytetrack` | `ByteTrackTracker` | boxes + confidence | **Default for `run`.** Two-stage association: high-confidence detections first, then low-confidence ones against what's left. Fast, robust to flicker. |
| `ocsort` | `OCSORTTracker` | boxes | Observation-centric SORT: interpolates through occlusions and penalises direction changes. Often the better choice for two mice that cross. Default in the demo. |
| `botsort` | `BoTSORTTracker` | boxes + confidence + camera motion | ByteTrack plus camera-motion compensation (`sparseOptFlow` by default). Costs extra per frame; useful for hand-held or vibrating cameras, wasted on a fixed one. |
| `sort` | `SORTTracker` | boxes | The original Kalman + Hungarian matcher. Simplest and cheapest; loses identity on any occlusion. |
| `cbiou` | `CBIoUTracker` | boxes + confidence | Cascaded-buffered IoU: expands boxes before matching so fast movers still overlap their previous position. |
| `mcbyte` | `McByteTracker` | boxes + confidence (+ masks) | ByteTrack extended with an optional mask-based manager. Mask matching is off by default (`enable_mask_manager=False`); MouseLite does not turn it on. |

## Which one

- **One animal**: it doesn't matter. Pass `--top-k 1` and the pipeline skips the
  tracker entirely (see [How it works](how-it-works.md)).
- **Two or more animals, fixed camera**: start with `bytetrack`. If identities swap
  when the animals huddle or cross, try `ocsort`, which is the one the demo defaults
  to for that reason.
- **Moving camera**: `botsort`.

`retrack` exists so you can try these on the same predictions without paying for
inference again:

```bash
mouselite run video.mp4 --kind keypoints --top-k 2            # bytetrack, once
mouselite retrack output/video_coco/annotations.json --tracker ocsort
mouselite retrack output/video_coco/annotations.json --tracker botsort
```

Each `retrack` overwrites `track_id` in the export and writes
`output/video_retracked.mp4`; rename the video between runs if you want to keep
several.

## Tuning

The two knobs `mouselite retrack` exposes are the ones that matter most for mice.
Everything else is reachable from Python by passing keyword arguments through
`get_tracker` or `retrack`.

### `lost_track_buffer` (all trackers, default 30)

How many frames a track is kept alive with no matching detection before it is
dropped and the animal, when it reappears, gets a new id. The trackers convert it
to seconds as `lost_track_buffer / 30` and count elapsed time at `frame_rate`, so
the value means "frames at 30 fps" whatever the video's rate — which is why `run`
and `retrack` pass the video's real frame rate as `frame_rate`.

Mice disappear under nests, behind each other and into corners. If ids keep
changing after a short occlusion, raise it (`--lost-track-buffer 90` is a second
and a half at 60 fps). Too high and an animal that genuinely left can absorb the
next detection that appears near where it was.

### `minimum_iou_threshold` (default 0.1–0.3, per tracker)

Minimum box overlap for a detection to be matched to an existing track. Lower it
if fast movements break tracks; raise it if two nearby animals steal each other's
ids. Defaults: `bytetrack` 0.1, `ocsort` and `sort` 0.3. `botsort`, `cbiou` and
`mcbyte` split this into `minimum_iou_threshold_first_assoc` /
`_second_assoc` / `_unconfirmed_assoc` and do not accept the single-name form —
passing `--minimum-iou-threshold` to them fails with a `TypeError` from the
constructor.

### Others worth knowing

| Argument | Trackers | Default | Meaning |
| -------- | -------- | ------- | ------- |
| `track_activation_threshold` | all but `ocsort` | 0.7 (`sort`: 0.25) | confidence a detection needs to start a new track |
| `minimum_consecutive_frames` | all | 2 (`ocsort`, `sort`: 3) | frames a new track must be matched before it is confirmed and gets an id; before that its detections export as `track_id = -1` |
| `high_conf_det_threshold` | ByteTrack family, `ocsort` | 0.6 | split between the first- and second-stage association |
| `enable_cmc`, `cmc_method` | `botsort`, `mcbyte` | `True`, `"sparseOptFlow"` | camera-motion compensation |

Run `python -c "import inspect, trackers; print(inspect.signature(trackers.OCSORTTracker))"`
for the full list of a given tracker.
