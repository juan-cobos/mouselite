# CLI reference

```
mouselite [OPTIONS] COMMAND [ARGS]...
```

| Command | Purpose |
| ------- | ------- |
| [`run`](#mouselite-run) | inference + tracking on one or more videos → annotated video + COCO export each |
| [`retrack`](#mouselite-retrack) | re-run tracking on an existing export, no inference |
| [`train`](#mouselite-train) | fine-tune a model on your own frames, optionally converting them first |
| [`app`](#mouselite-app) | launch the Gradio demo |
| [`list-models`](#list-commands) | print valid `--kind` / `--size` values |
| [`list-trackers`](#list-commands) | print valid `--tracker` values |
| [`list-formats`](#list-commands) | print valid `train --from` values |

Every command accepts `--help`. Boolean options follow typer's convention: `--show`
turns a flag on, `--no-show` turns it off; the `[default: …]` in `--help` shows which
applies. `mouselite --install-completion` sets up shell completion.

The heavy imports (torch, rfdetr) are deferred until a command needs them, so
`--help` and the `list-*` commands start instantly.

## `mouselite run`

```
mouselite run VIDEO_PATH... --kind KIND [OPTIONS]
```

Runs the model on each `VIDEO_PATH`, links predictions across frames with a tracker,
and writes, per video:

```
<output-dir>/
└── <stem>_results/
    ├── <stem>_annotated.mp4
    ├── annotations.json
    └── images/
```

where `<stem>` is the video's file name without extension. Prints `wrote <path>` for
each video on success. The export format is described in [COCO export](coco-export.md).

### Model

| Option | Default | Meaning |
| ------ | ------- | ------- |
| `--kind` | *required* | `detection`, `segmentation` or `keypoints` |
| `--size` | `medium` | `nano`, `small`, `medium` or `large`. Ignored for `keypoints`, which is a single checkpoint. |
| `--checkpoint PATH` | — | Your own `.pt` weights. Skips the Hugging Face download; `--kind`/`--size` still choose the architecture, so they must match the checkpoint. |
| `--dtype` | `float32` | Floating-point dtype the model is cast to (any `torch.dtype` name, e.g. `float16`, `bfloat16`). Half precision is faster on GPUs that support it. |
| `--compile` / `--no-compile` | off | Trace the model with `torch.jit.trace` for faster per-frame inference at the cost of a slower start. Worth it for long videos. |
| `--batch-size` | `1` | Batch size the traced model is optimised for. Only meaningful with `--compile`; the pipeline itself always predicts one frame at a time. |

### Filtering predictions

Applied in this order, on every inference frame:

| Option | Default | Meaning |
| ------ | ------- | ------- |
| `--threshold` | `0.5` | Minimum confidence for a prediction to be kept. |
| `--nms-threshold` | `0.5` | Non-maximum suppression: of two predictions overlapping above this IoU, the lower-scoring one is dropped. |
| `--top-k N` | — | Keep only the `N` highest-scoring predictions. Set it to the number of animals in the cage. `--top-k 1` also skips tracking altogether. |

### Tracking

| Option | Default | Meaning |
| ------ | ------- | ------- |
| `--tracker` | `ocsort` | One of `list-trackers`. See [Trackers](trackers.md). |

`run` constructs the tracker with its defaults. To pass tracker arguments, use
`retrack` or the [Python API](python-api.md).

### Frames and output

| Option | Default | Meaning |
| ------ | ------- | ------- |
| `--every N` | `1` | Run inference on 1 of every `N` frames; the frames in between are annotated with the last predictions. Only inference frames are exported. |
| `--output-dir` | `output` | Directory for the video and the export. Created if missing. |
| `--show` / `--no-show` | off | Open a window and preview each annotated frame as it is produced. Needs a display. |
| `--hud` / `--no-hud` | off | Burn a live `FPS: …` counter (of the pipeline, not the video) into the top-left of the output. |
| `--show-progress` / `--no-show-progress` | on | Progress bar on stderr. |

### Examples

```bash
# Pose on two animals, every other frame, watch it as it goes
mouselite run cage.mp4 --kind keypoints --top-k 2 --every 2 --tracker ocsort --show

# Segmentation with the largest model, half precision, compiled
mouselite run cage.mp4 --kind segmentation --size large --dtype float16 --compile

# Your own detection weights for the small architecture
mouselite run cage.mp4 --kind detection --size small --checkpoint runs/best.pt

# Single animal: no tracker involved
mouselite run cage.mp4 --kind detection --top-k 1

# Every video in a folder, plus one more, with the model loaded once
mouselite run recordings/ extra.mp4 --kind keypoints --top-k 2
```

## `mouselite retrack`

```
mouselite retrack ANNOTATIONS_PATH --tracker TRACKER [OPTIONS]
```

Replays the export at `ANNOTATIONS_PATH` (an `annotations.json` written by `run`,
with its `images/` folder beside it) through a fresh tracker. No model is loaded.

Writes `<output-dir>/<stem>_retracked.mp4` (at 30 fps), where `<stem>` is the export
folder's name minus `_results`, and rewrites `track_id` on every annotation **in place**.

| Option | Default | Meaning |
| ------ | ------- | ------- |
| `--tracker` | *required* | One of `list-trackers`. |
| `--output-dir` | `output` | Where the retracked video goes. |
| `--lost-track-buffer N` | tracker's (30) | Frames a track survives without a match. |
| `--minimum-iou-threshold X` | tracker's (0.1–0.3) | Minimum IoU to match a detection to a track. Not accepted by `botsort`, `cbiou` or `mcbyte`, which split it into several arguments. |
| `--show-progress` / `--no-show-progress` | on | Progress bar. |

The two tracker options are forwarded to the tracker's constructor only when given,
so omitting them keeps the tracker's own defaults. See [Trackers](trackers.md) for
what they do and for the confidence caveat that applies to retracking.

```bash
mouselite retrack output/cage_results/annotations.json --tracker ocsort
mouselite retrack output/cage_results/annotations.json --tracker ocsort --lost-track-buffer 90 --minimum-iou-threshold 0.15
```

## `mouselite train`

```
mouselite train DATASET_DIR --kind KIND [--from FORMAT] [OPTIONS]
```

Fine-tunes a model when the released weights don't perform as expected on your
recordings. `DATASET_DIR` is a COCO dataset with `train/` and `valid/` folders (each
with its images and `_annotations.coco.json`), or — with `--from` — a project in one
of the formats `list-formats` prints, which is converted into
`<output-dir>/dataset/` first. Requires the `train` extra (which includes the
converters); without it the command exits with status 1 and an install hint. See
[Fine-tuning](training.md) for the workflow and what the conversion does.

### Model

| Option | Default | Meaning |
| ------ | ------- | ------- |
| `--kind` | *required* | `detection`, `segmentation` or `keypoints`. |
| `--size` | `medium` | `nano`, `small`, `medium` or `large`. Ignored for `keypoints`. |
| `--weights` | `base` | Starting weights: `base` (RF-DETR's pretrained), `mouselite` (the released weights) or a checkpoint path. |

### Dataset

| Option | Default | Meaning |
| ------ | ------- | ------- |
| `--from FORMAT` | — | Convert `DATASET_DIR` from `FORMAT` before training. Currently `deeplabcut`: the project folder or its `config.yaml`. Refused with `--kind segmentation`, since pose projects carry no masks. |
| `--symlink` / `--no-symlink` | symlink | With `--from`: symlink the images into the converted dataset, or copy them. |

### Training

| Option | Default | Meaning |
| ------ | ------- | ------- |
| `--output-dir` | `output/train` | Checkpoints, `metrics.csv`/`metrics.png` and, with `--from`, the `dataset/` folder. |
| `--epochs` | `10` | Epochs. For `keypoints`, train for more than 10 so `checkpoint_best_ema.pth` is meaningful (see [Fine-tuning](training.md#what-you-get)). |
| `--batch-size` | `4` | Lower it on small GPUs. |
| `--lr` | RF-DETR's | Learning rate. |
| `--resolution` | RF-DETR's | Input resolution. |
| `--device` | auto | Torch device. |

Prints `converted <format> dataset to <path>` when `--from` is used, then
`wrote checkpoints and metrics.png to <output-dir>` on success.

### Examples

```bash
# A DeepLabCut project, straight from its folder
mouselite train dlc-project/ --kind keypoints --from deeplabcut --epochs 30

# The same, copying the images so the dataset is self-contained
mouselite train dlc-project/ --kind keypoints --from deeplabcut --no-symlink

# A COCO dataset, continuing from the released weights
mouselite train dataset/ --kind detection --size small --weights mouselite --epochs 20
```

## `mouselite app`

```
mouselite app [--share/--no-share] [--host HOST] [--port PORT]
```

Serves the Gradio demo. Requires the `app` extra (`pip install "mouselite[app]"`);
without it the command exits with status 1 and an install hint.

| Option | Default | Meaning |
| ------ | ------- | ------- |
| `--share` / `--no-share` | share | Create a public `*.gradio.live` tunnel. Use `--no-share` for local-only. |
| `--host` | Gradio's (`127.0.0.1`) | Interface to bind. `0.0.0.0` to expose on the LAN. |
| `--port` | Gradio's (`7860`) | Port to bind. |

The demo uploads a video, runs `run` with the chosen kind/size/tracker/thresholds
into a temp directory, offers the `annotations.json` for download, and can `retrack`
the same predictions with another tracker. Models are cached in memory (two at a
time) so switching back and forth does not reload weights.

## `mouselite list-models`, `list-trackers` and `list-formats` { #list-commands }

```
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

$ mouselite list-formats
deeplabcut
```

These read `mouselite.models.MODELS`, `mouselite.tracker.TRACKERS` and
`mouselite.format.FORMATS`, so anything you register there from Python shows up too.
`list-formats` needs the `convert` extra, which `train` includes.

## Exit status and errors

- `0` on success; the last line of stdout is `wrote <path>`.
- Unknown `--tracker` raises `ValueError: Unknown tracker '…'. Available: […]`.
- Unknown `--kind` or `--size` raises `KeyError` from the model registry.
- Unknown `--from` raises `ValueError: Unknown format '…'. Available: […]`;
  `--from` with `--kind segmentation` exits with status 1 and a message.
- `train`, `app` and `list-formats` exit with status 1 and a `pip install` hint when
  their extra is missing.
- A tracker argument the chosen tracker does not accept (for example
  `--minimum-iou-threshold` with `botsort`) raises `TypeError` from its constructor.
- `--show` without a display fails inside OpenCV (`cv2.imshow`); drop the flag on
  headless machines.
- `run` on a directory without video files exits with status 2 and a usage error.
