# MouseLite

Real-time mouse detection, segmentation and pose estimation.

MouseLite wraps fine-tuned [RF-DETR](https://github.com/roboflow/rf-detr) models in a
video pipeline: it predicts per frame, links predictions across frames with a
multi-object tracker, writes an annotated video, and exports the predictions as a
COCO dataset you can re-track or analyse later.

Three model kinds are available:

| Kind           | Sizes                             | Output                     |
| -------------- | --------------------------------- | -------------------------- |
| `detection`    | `nano`, `small`, `medium`, `large`| bounding boxes             |
| `segmentation` | `nano`, `small`, `medium`, `large`| boxes + instance masks     |
| `keypoints`    | single checkpoint                 | boxes + pose keypoints     |

## Installation

Requires Python ≥ 3.12.

```bash
pip install mouselite
```

With [uv](https://docs.astral.sh/uv/):

```bash
uv add mouselite
```

The Gradio demo is an optional extra:

```bash
pip install "mouselite[app]"
```

From a checkout:

```bash
git clone https://github.com/juan-cobos/mouselite
cd mouselite
uv sync --extra app
```

Model weights are downloaded from the Hugging Face Hub on first use and cached;
pass `--checkpoint` to use your own instead.

## Usage (CLI)

```bash
mouselite --help
```

### `run` — inference on a video

```bash
mouselite run video.mp4 --kind keypoints
```

Writes two things under `--output-dir` (default `output/`):

```
output/
├── video_annotated.mp4          # annotated video
└── video_coco/
    ├── annotations.json         # COCO export (boxes, masks, keypoints, track ids)
    └── images/                  # the frames inference ran on
```

Common options:

| Option             | Default     | Meaning                                                     |
| ------------------ | ----------- | ----------------------------------------------------------- |
| `--kind`           | *required*  | `detection`, `segmentation` or `keypoints`                   |
| `--size`           | `medium`    | model size; ignored by `keypoints`                           |
| `--checkpoint`     | —           | path to your own weights, skipping the Hub download          |
| `--tracker`        | `bytetrack` | tracking algorithm (see `list-trackers`)                     |
| `--threshold`      | `0.5`       | minimum confidence for a prediction to be kept               |
| `--nms-threshold`  | `0.5`       | drop the lower-scoring of two predictions overlapping above this |
| `--top-k`          | —           | keep only the N highest-scoring predictions per frame        |
| `--every`          | `1`         | run inference on 1 of every N frames, reusing predictions in between |
| `--output-dir`     | `output`    | where the video and COCO export are written                  |
| `--save-path`      | —           | write `annotations.json` somewhere else                      |
| `--show`           | off         | preview the annotated frames in a window while running       |
| `--hud`            | off         | draw a live FPS counter on the output                        |
| `--dtype`          | `float32`   | inference precision                                          |
| `--batch-size`     | `1`         | inference batch size                                          |
| `--compile`        | off         | `torch.compile` the model — slower to start, faster per frame |
| `--no-show-progress` | —         | silence the progress bar                                      |

Two animals, pose, half the frames, with a preview window:

```bash
mouselite run video.mp4 --kind keypoints --top-k 2 --every 2 --tracker ocsort --show
```

### `retrack` — re-run tracking without re-running inference

Tracking is usually what you end up tuning, and it is far cheaper than inference.
`retrack` replays an existing COCO export through a different tracker:

```bash
mouselite retrack output/video_coco/annotations.json --tracker ocsort --fps 30
```

Writes `output/video_retracked.mp4`.

### `app` — Gradio demo

```bash
mouselite app                       # needs the [app] extra
mouselite app --no-share --port 7860
```

Upload a video, pick a model and tracker, run, and retrack the same predictions with
a different tracker without paying for inference again.

### `list-models` / `list-trackers`

```bash
$ mouselite list-models
detection: nano, small, medium, large
segmentation: nano, small, medium, large
keypoints

$ mouselite list-trackers
botsort
ocsort
bytetrack
sort
cbiou
mcbyte
```

## Usage (Python API)

The CLI is a thin wrapper over three pieces: a model, a tracker, and a `Pipeline`
that joins them.

```python
import supervision as sv

from mouselite.models import get_model
from mouselite.pipeline import Pipeline
from mouselite.tracker import get_tracker

model = get_model("keypoints")
fps = sv.VideoInfo.from_video_path("video.mp4").fps
tracker = get_tracker("ocsort", frame_rate=fps)

pipeline = Pipeline(model, tracker, threshold=0.5, top_k=2)
annotated_path = pipeline.run("video.mp4", output_dir="output")
```

### `get_model`

```python
model = get_model(
    "segmentation",       # "detection", "segmentation" or "keypoints"
    size="large",         # ignored for "keypoints"
    checkpoint=None,      # path to your own weights; otherwise pulled from the Hub
    dtype="float32",
    batch_size=1,
    compile=False,
)
```

Returns an RF-DETR model already put in inference mode. Any object with a
`predict(frame, threshold) -> sv.Detections | sv.KeyPoints` method and a `class_names`
attribute works in its place — that is the whole `MLModel` protocol the pipeline
depends on.

### `get_tracker`

```python
tracker = get_tracker("bytetrack", frame_rate=30)
```

Any name from `mouselite.tracker.TRACKERS`; keyword arguments go straight to the
underlying [`trackers`](https://github.com/roboflow/trackers) class.

### `Pipeline`

```python
pipeline = Pipeline(
    model,
    tracker,
    threshold=0.5,        # confidence floor
    nms_threshold=0.5,    # NMS IoU threshold
    top_k=None,           # cap on predictions per frame, by confidence
    every=1,              # run inference on 1 of every N frames
)

annotated_path = pipeline.run(
    "video.mp4",
    output_dir="output",
    save_path=None,       # override the annotations.json location
    show=False,           # live preview window
    hud=False,            # FPS overlay
    show_progress=True,
)
```

`run` returns the path of the annotated video and writes the COCO export beside it.
Keypoint predictions are converted to `sv.Detections` for tracking — keeping the
model's own box rather than a box fitted to the keypoints — and carried through to the
export as COCO `keypoints`/`num_keypoints` fields.

### `retrack`

```python
from mouselite.tracker import get_tracker, retrack

retracked_path = retrack(
    "output/video_coco/annotations.json",
    get_tracker("botsort", frame_rate=30),
    output_dir="output",
    fps=30,
)
```

## Training

The code behind the released models — fine-tuning RF-DETR and the DeepLabCut
SuperAnimal baseline, plus the scripts that scored them — lives in
[`training/`](training/README.md). It is for reproducing the paper; to just run the
models, use the package above.

## Acknowledgements

MouseLite is built on work by others:

- **[RF-DETR](https://github.com/roboflow/rf-detr)** — the real-time detection
  transformer behind every MouseLite model. The detection, segmentation and
  keypoints-preview architectures are RF-DETR's; MouseLite fine-tunes them on mice.
- **[supervision](https://github.com/roboflow/supervision)** — detection and keypoint
  containers, NMS, annotators, video I/O and the COCO format helpers. It is the
  vocabulary the whole pipeline is written in.
- **[trackers](https://github.com/roboflow/trackers)** — every multi-object tracker
  MouseLite offers. ByteTrack, BoT-SORT, OC-SORT, SORT, C-BIoU and McByte all come from
  it unchanged; MouseLite only picks one and hands it detections.
- **[DeepLabCut](https://github.com/DeepLabCut/DeepLabCut)** — the reference point for
  markerless animal pose estimation, and the SuperAnimal baseline MouseLite is evaluated
  against. This project exists because of the problem DeepLabCut defined and the
  community it built around it.
