# Python API

The CLI is a thin wrapper over a few importable pieces. Nothing is re-exported from
`mouselite` itself; import from the submodules:

```python
from mouselite.models import MODELS, get_model
from mouselite.pipeline import MetaAnnotator, Pipeline
from mouselite.tracker import TRACKERS, get_tracker, retrack
from mouselite import analysis
```

`mouselite.tracker` imports `trackers` (and so torch) at module level; `mouselite.models`
imports `rfdetr` lazily inside `get_model`.

## `mouselite.models`

### `MODELS`

```python
MODELS = {
    "detection":    {"nano": "RFDETRNano",    "small": "RFDETRSmall",    "medium": "RFDETRMedium",    "large": "RFDETRLarge"},
    "segmentation": {"nano": "RFDETRSegNano", "small": "RFDETRSegSmall", "medium": "RFDETRSegMedium", "large": "RFDETRSegLarge"},
    "keypoints":    "RFDETRKeypointPreview",
}
```

Maps `kind` (and `size`, for the sized kinds) to the `rfdetr` class name. `list-models`
prints it.

### `get_model(kind, size="medium", checkpoint=None, dtype="float32", batch_size=1, compile=False)`

| Parameter | Type | Meaning |
| --------- | ---- | ------- |
| `kind` | `str` | key of `MODELS` |
| `size` | `str` | key of `MODELS[kind]`; ignored for `"keypoints"` |
| `checkpoint` | `str \| Path \| None` | local weights. When `None`, `mouselite-<kind>-<size>.pt` (or `mouselite-keypoints.pt`) is fetched from the Hugging Face Hub and cached |
| `dtype` | `str` | passed to `model.inference`; any `torch.dtype` name |
| `batch_size` | `int` | passed to `model.inference`; only used when `compile=True` |
| `compile` | `bool` | passed to `model.inference`; traces with `torch.jit.trace` |

Returns an `rfdetr` model with `.inference(...)` already called. Raises `KeyError` for
an unknown `kind`/`size`.

```python
model = get_model("segmentation", size="large", dtype="float16", compile=True)
model = get_model("detection", size="small", checkpoint="runs/best.pt")
```

## `mouselite.tracker`

### `TRACKERS`

`dict[str, type[BaseTracker]]` mapping CLI names to `trackers` classes:
`botsort`, `ocsort`, `bytetrack`, `sort`, `cbiou`, `mcbyte`. Mutable — add your own.

### `get_tracker(name, **kwargs)`

Returns `TRACKERS[name](**kwargs)`. Raises `ValueError` listing the valid names for
an unknown `name`. `run` calls it with no keyword arguments; frame-based settings
such as `lost_track_buffer` count the frames the tracker actually sees.

### `retrack(annotations_path, video_path, tracker, output_dir="output", show_progress=True, **tracker_kwargs) -> Path`

Replay an export through a tracker; see [How it works](how-it-works.md#the-retrack-path)
for the mechanics and caveats.

| Parameter | Meaning |
| --------- | ------- |
| `annotations_path` | the `<stem>_annotations.json` written by `Pipeline.run` |
| `video_path` | the video the export was made from; frames are matched by `frame_index` |
| `tracker` | a key of `TRACKERS` (a name, not an instance) |
| `output_dir` | where `<stem>_retracked.mp4` is written, at the source's frame rate |
| `**tracker_kwargs` | forwarded to the tracker constructor |

Returns the path of the retracked video. Side effect: `track_id` is rewritten on
every annotation in `annotations_path`.

## `mouselite.pipeline`

### `Pipeline(model, tracker, threshold=0.5, nms_threshold=0.5, top_k=None, every=1)`

| Parameter | Meaning |
| --------- | ------- |
| `model` | a model from `get_model` |
| `tracker` | a `trackers.core.base.BaseTracker` (or anything with `reset()` and `update(detections, frame=...)` that sets `tracker_id`) |
| `threshold` | confidence floor handed to `model.predict` |
| `nms_threshold` | IoU for `Detections.with_nms` |
| `top_k` | keep at most this many predictions per frame, by confidence; `1` disables tracking |
| `every` | inference stride |

A `Pipeline` is reusable: `run` calls `tracker.reset()` first. It is not thread-safe
(the tracker and annotator carry state).

### `Pipeline.run(video_path, output_dir="output", show=False, show_progress=True, hud=False) -> Path`

Processes the whole video and returns the annotated video's path,
`<output_dir>/<stem>_results/<stem>_annotated.mp4`. The COCO export goes beside it as
`<stem>_annotations.json`; each image records its `frame_index` in the video rather than
being written to disk. `show` opens an OpenCV window; `hud` burns in an FPS
counter.

```python
from mouselite.models import get_model
from mouselite.pipeline import Pipeline
from mouselite.tracker import get_tracker

model = get_model("keypoints")
tracker = get_tracker("bytetrack", lost_track_buffer=60)

pipeline = Pipeline(model, tracker, threshold=0.5, top_k=2, every=2)
video = pipeline.run("cage.mp4", output_dir="output")
export = video.parent / "cage_annotations.json"
```

To process several videos with one loaded model, reuse the `Pipeline`: `run` resets
the tracker each time.

### `MetaAnnotator`

```python
annotator = MetaAnnotator()
annotated = annotator.annotate(frame.copy(), detections)   # detections must have tracker_id
```

Draws masks (if `detections.mask`), boxes, `#<track_id>` labels, and keypoints (if
`detections.data["keypoints_xy"]`), coloured by track. `annotate` draws on the array
it is given, hence the `copy()`. Useful for rendering your own frames from an export
or a custom loop.

### Processing frames yourself

If you want the predictions without the video and export machinery, skip `Pipeline`:

```python
import cv2
import supervision as sv
from mouselite.models import get_model
from mouselite.pipeline import MetaAnnotator, _keypoints_to_detections
from mouselite.tracker import get_tracker

model = get_model("keypoints")
tracker = get_tracker("bytetrack")
annotator = MetaAnnotator()

for frame in sv.get_video_frames_generator("cage.mp4"):
    preds = model.predict(frame, threshold=0.5)
    if isinstance(preds, sv.KeyPoints):
        preds = _keypoints_to_detections(preds)
    preds = tracker.update(preds.with_nms(threshold=0.5), frame=frame)
    # preds.xyxy, preds.tracker_id, preds.data["keypoints_xy"] …
```

`_keypoints_to_detections` is private by name but stable in practice; it is the
one conversion the pipeline relies on.

## `mouselite.analysis`

Downstream statistics from an export: `Tracks` lays a run out on a `(frame, track)`
grid, and the free functions take arrays off that grid.

### `Tracks.from_coco(annotations_path, fps=None, min_frames=1, masks=False) -> Tracks`

Reads a MouseLite `<stem>_annotations.json` back. Unconfirmed detections (`track_id == -1`)
are dropped, as are tracks seen on fewer than `min_frames` frames. `fps` overrides the
one recorded in the export's `info`, which matters for anything per second.
`masks=True` also decodes a segmentation export's masks into a `(T, N, H, W)` bool
array — one byte per pixel per track per frame, so a long recording at full resolution
takes gigabytes.

```python
from mouselite import analysis

tracks = analysis.Tracks.from_coco("output/cage_results/cage_annotations.json", fps=30)
```

### `Tracks`

| Attribute | Shape | What it holds |
| --- | --- | --- |
| `frame_index` | `(T,)` | the frame each row came from |
| `track_ids` | `(N,)` | the animal each column belongs to |
| `xyxy` | `(T, N, 4)` | boxes, NaN where the animal was not seen |
| `area` | `(T, N)` | box or mask area, NaN where absent |
| `keypoints` | `(T, N, K, 2)` | pose keypoints, or `None` |
| `keypoint_names` | `K` names | the names the model was trained with |
| `masks` | `(T, N, H, W)` | only with `masks=True`, else `None` |
| `fps`, `image_size`, `video` | | what the export recorded about the video |

Derived views: `n_frames`, `n_tracks`, `time` (seconds, needs `fps`), `present`
`(T, N)` whether each animal was seen, `centroids` `(T, N, 2)` box centres,
`keypoint(name_or_index)` `(T, N, 2)` for one named keypoint, and
`select(track_ids)` for the same grid restricted to some animals.

### `Tracks.summary(scale=1.0, immobile_below=None) -> DataFrame`

One row per animal: `frames`, `first_frame`, `last_frame`, `coverage`, `distance`,
`mean_speed`, `max_speed`, `mean_area`, plus `duration` when `fps` is known,
`mean_elongation` for a segmentation export, and `immobile_fraction` when
`immobile_below` is given. This is what `<stem>_summary.csv` holds. `scale` converts pixels
to your unit (cm per pixel), and applies to distances, speeds and areas.

### `Tracks.to_dataframe() -> DataFrame`

Long table, one row per `(frame, track)` the animal was seen on: `frame_index`,
`time` when `fps` is known, `track_id`, `x`, `y`, the box corners, `area`, and a
`<name>_x` / `<name>_y` pair per keypoint. This is what `<stem>_trajectories.csv` holds.

### `Tracks.to_deeplabcut(path=None, scorer="mouselite") -> DataFrame`

The DeepLabCut-style prediction table most downstream tools read: columns
`(scorer, [individuals,] bodyparts, coords)`, one row per frame. The `individuals`
level is only added for multi-animal exports, as DLC does; without keypoints the box
centre is written as a single `centroid` bodypart. MouseLite exports no per-keypoint
confidence, so `likelihood` is 1 where a point was seen and 0 where it was not.
Written to `path` as CSV, or HDF5 for a `.h5` suffix.

### Kinematics, cleaning and space

Every function takes plain arrays — `(T, N, 2)` positions off the grid — and returns
arrays, so they compose with whatever you already use.

| Function | Gives |
| --- | --- |
| `speed(xy, frame_index=None, fps=None, scale=1.0)` | `(T, N)` per frame, in units per second with `fps`, else per frame |
| `distance_traveled(xy, scale=1.0)` | `(N,)` path length, ignoring gaps |
| `heading(a, b)` | `(T, N)` angle from point `a` to point `b`, e.g. tail base to nose |
| `keypoint_distance(a, b, scale=1.0)` | `(T, N)` distance between two keypoints |
| `pairwise_distance(xy, scale=1.0)` | `(T, N, N)` distance between every pair of animals |
| `interpolate(xy, max_gap=None)` | gaps of at most `max_gap` frames filled linearly, never extrapolating |
| `smooth(xy, window=5)` | centred rolling median, DeepLabCut's default filter |
| `in_polygon(xy, polygon)` | `(T, N)` whether each animal is inside a region |
| `occupancy(xy, image_size, bins=32)` | `(N, rows, cols)` frames spent in each cell, for a heatmap |
| `bouts(mask, min_frames=1)` | `(start, stop)` row ranges of each run of `True` in a 1D mask, `stop` exclusive |
| `mask_centroids(masks)` | `(T, N, 2)` centre of mass of each segmentation mask |
| `mask_axes(masks)` | major axis, minor axis and body angle of each mask; `major / minor` is how stretched the animal is |

```python
nose, tail = tracks.keypoint("nose"), tracks.keypoint("tail_base")
xy = analysis.smooth(analysis.interpolate(tracks.centroids, max_gap=5))

v = analysis.speed(xy, tracks.frame_index, tracks.fps, scale=0.05)  # cm/s
facing = analysis.heading(tail, nose)
apart = analysis.pairwise_distance(xy, scale=0.05)[:, 0, 1]         # animals 0 and 1
close = analysis.bouts(apart < 4, min_frames=10)                    # contact episodes
```

## `mouselite.train`

### `train(dataset_dir, kind, size="medium", weights=BASE, output_dir="output/train", epochs=10, batch_size=4, **kwargs) -> None`

What `mouselite train` calls once the dataset is in COCO form. Builds the model with
`models.build_model(kind, size, weights=weights)` — `weights` is `models.BASE`
(RF-DETR's pretrained weights), `models.MOUSELITE` (the released weights) or a
checkpoint path — runs `rfdetr`'s `train` on `dataset_dir`, and writes `metrics.png`
next to the checkpoints in `output_dir`.

`kwargs` are forwarded to `rfdetr`'s `train` untouched, which is how you reach
anything the CLI doesn't expose:

```python
from mouselite.models import MOUSELITE
from mouselite.train import train

train(
    "dataset",
    "keypoints",
    weights=MOUSELITE,
    output_dir="output/train",
    epochs=30,
    keypoint_oks_sigmas=[0.03, 0.05, 0.08],  # per-bodypart, for the OKS metric
    grad_accum_steps=2,
)
```

`train` knows nothing about DeepLabCut or other formats; conversion is a separate
step, below, that the CLI runs first when given `--from`.

## `mouselite.format`

### `FORMATS`

`dict[str, loader]` of the formats `convert` accepts — the values `list-formats`
prints. Each loader takes a project path and returns `(image_paths, xy, category)`:
the labeled frames, an array of shape `(frames, individuals, keypoints, 2)` with NaN
for unlabeled keypoints, and the COCO category dict (`keypoints`, `skeleton`).
Currently `deeplabcut` and `lightning-pose`; add a loader here and it becomes a
`--from` value.

### `convert(dataset_dir, output_dir, from_format, train_fraction=0.8, seed=0, class_name="mouse", symlink=True) -> Path`

Turns a project in `from_format` into the `train/` + `valid/` COCO layout RF-DETR
trains on, at `output_dir`, and returns it. Frames are shuffled with `seed` and split
by `train_fraction`; images are symlinked into the split folders, or copied with
`symlink=False`. See [Fine-tuning](training.md#if-coming-from-deeplabcut-or-lightning-pose) for what the
DeepLabCut conversion does with animals, bodyparts and skeletons.

```python
from mouselite.format import convert
from mouselite.train import train

dataset = convert("dlc-project/", "output/train/dataset", "deeplabcut")
train(dataset, "keypoints", output_dir="output/train", epochs=30)
```
