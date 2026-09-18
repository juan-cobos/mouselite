# Python API

The CLI is a thin wrapper over a few importable pieces. Nothing is re-exported from
`mouselite` itself; import from the submodules:

```python
from mouselite.models import MODELS, get_model
from mouselite.pipeline import MetaAnnotator, Pipeline
from mouselite.tracker import TRACKERS, get_tracker, retrack
from mouselite.train import train            # [train] extra
from mouselite.format import FORMATS, convert  # [convert] extra, included in [train]
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

### `retrack(annotations_path, tracker, output_dir="output", show_progress=True, **tracker_kwargs) -> Path`

Replay an export through a tracker; see [How it works](how-it-works.md#the-retrack-path)
for the mechanics and caveats.

| Parameter | Meaning |
| --------- | ------- |
| `annotations_path` | the `annotations.json`; `images/` must be beside it |
| `tracker` | a key of `TRACKERS` (a name, not an instance) |
| `output_dir` | where `<stem>_retracked.mp4` is written, at 30 fps |
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

### `Pipeline.run(video_path, output_dir="output", show=False, show_progress=True, hud=False, save_path=None) -> Path`

Processes the whole video and returns the annotated video's path. Writes the COCO
export to `save_path` or `<output_dir>/<stem>_coco/annotations.json`, with frames in
an `images/` folder beside it. `show` opens an OpenCV window; `hud` burns in an FPS
counter.

```python
from mouselite.models import get_model
from mouselite.pipeline import Pipeline
from mouselite.tracker import get_tracker

model = get_model("keypoints")
tracker = get_tracker("ocsort", lost_track_buffer=60)

pipeline = Pipeline(model, tracker, threshold=0.5, top_k=2, every=2)
video = pipeline.run("cage.mp4", output_dir="output")
export = video.parent / "cage_coco" / "annotations.json"
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
Currently `{"deeplabcut": from_deeplabcut}`; add a loader here and it becomes a
`--from` value.

### `convert(dataset_dir, output_dir, from_format, train_fraction=0.8, seed=0, class_name="mouse", symlink=True) -> Path`

Turns a project in `from_format` into the `train/` + `valid/` COCO layout RF-DETR
trains on, at `output_dir`, and returns it. Frames are shuffled with `seed` and split
by `train_fraction`; images are symlinked into the split folders, or copied with
`symlink=False`. See [Fine-tuning](training.md#if-coming-from-deeplabcut) for what the
DeepLabCut conversion does with animals, bodyparts and skeletons.

```python
from mouselite.format import convert
from mouselite.train import train

dataset = convert("dlc-project/", "output/train/dataset", "deeplabcut")
train(dataset, "keypoints", output_dir="output/train", epochs=30)
```
