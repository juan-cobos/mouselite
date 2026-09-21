# MouseLite

<div align="center"><img src="assets/logo.svg" alt="MouseLite" width="300"></div>

Real-time mouse detection, segmentation and pose estimation.

MouseLite wraps fine-tuned [RF-DETR](https://github.com/roboflow/rf-detr) models in a
video pipeline: it predicts per frame, links predictions across frames with a
multi-object tracker, writes an annotated video, and exports the predictions as a
COCO dataset you can re-track or analyse later.

Three model kinds trained on [MTMB](https://github.com/juan-cobos/mtmb) dataset are available:

| Kind           | Sizes                             | Output                     |
| -------------- | --------------------------------- | -------------------------- |
| `detection`    | `nano`, `small`, `medium`, `large`| bounding boxes             |
| `segmentation` | `nano`, `small`, `medium`, `large`| boxes + instance masks     |
| `keypoints`    | single checkpoint                 | boxes + pose keypoints     |

## Installation

```bash
pip install mouselite                    # or: uv add mouselite
pip install "mouselite[app]"             # with the Gradio demo
pip install "mouselite[train]"           # to fine-tune on your own data
```

Requires Python ≥ 3.11. Model weights are downloaded from the Hugging Face Hub on
first use and cached.

## CLI

```bash
mouselite --help
```

### CLI examples

```bash
# Run keypoints predictions on 'video.mp4' with maximum 2 animals and ocsort tracker
mouselite run video.mp4 --kind keypoints --top-k 2 --tracker ocsort

# Same but on all videos in the directory
mouselite run /my-folder/ --kind keypoints --top-k 2 --tracker ocsort

# Re-run tracking without re-running inference 
mouselite retrack output/video_results/annotations.json --tracker bytetrack --lost-track-buffer 90 --minimum-iou-threshold 0.15
```

See [the docs](https://juan-cobos.github.io/mouselite/CLI/) for all the CLI options.

### `app` — Gradio demo

```bash
mouselite app                       # needs the [app] extra
```

Upload a video, pick a model and tracker, run, download the annotated video with its
COCO export and CSV tables, and retrack the same predictions with a different tracker
without paying for inference again.

### `train` — fine-tune when the released models fall short

If the released weights don't perform well on your recordings, fine-tune them on a
few hundred labeled frames of your own footage:

```bash
mouselite train dataset/ --kind keypoints --epochs 30
```

`dataset/` is a COCO dataset with `train/` and `valid/` folders. If you're coming
from DeepLabCut or Lightning Pose, point `train` at the project instead and add
`--from`; the labeled frames are converted before training starts:

```bash
mouselite train dlc-project/ --kind keypoints --from deeplabcut --epochs 30
mouselite train lp-project/ --kind keypoints --from lightning-pose --epochs 30
```

Then run your videos with the new weights:

```bash
mouselite run video.mp4 --kind keypoints --checkpoint output/train/checkpoint_best_ema.pth
```

See [the docs](https://juan-cobos.github.io/mouselite/training/) for the options and
what the conversion does.

## Python API

The CLI is a thin wrapper over three pieces: a model, a tracker, and a `Pipeline`
that joins them.

```python
from mouselite.models import get_model
from mouselite.pipeline import Pipeline
from mouselite.tracker import get_tracker

model = get_model("keypoints")
tracker = get_tracker("ocsort")

pipeline = Pipeline(model, tracker, threshold=0.5, top_k=2)
annotated_path = pipeline.run("video.mp4", output_dir="output")
retracked_path = retrack(
    annotated_path,
    "bytetrack",
    output_dir="output",
    lost_track_buffer=90,   # any further kwargs go to the tracker class
)
```

## Reproducibility

The code behind the *released* models — fine-tuning RF-DETR and the DeepLabCut SuperAnimal baseline,
plus the scripts that scored them — lives in [`paper/`](paper/README.md). It is for reproducing the
paper.

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
