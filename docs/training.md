# Fine-tuning

The released models were trained on a broad mix of cages, cameras and mouse strains,
and they should work out of the box on most recordings. When they don't — a top-down
camera at an unusual angle, an arena with strong reflections, a coat colour the
training set lacked, or keypoints you need that the released head doesn't predict —
the fix is to fine-tune on a few hundred frames of your own footage rather than to
tune thresholds. `mouselite train` does that from the terminal.

Training needs the `train` extra, which also brings the format converters:

```bash
pip install "mouselite[train]"
```

## From a COCO dataset

`mouselite train` takes a dataset in the layout RF-DETR expects — a `train/` and a
`valid/` folder, each holding the images and an `_annotations.coco.json`:

```
dataset/
├── train/
│   ├── _annotations.coco.json
│   └── *.png
└── valid/
    ├── _annotations.coco.json
    └── *.png
```

This is what Roboflow exports, and what `mouselite run` writes (minus the split). For
`keypoints`, the category must carry a `keypoints` list; RF-DETR reads the number of
keypoints from it.

```bash
mouselite train dataset/ --kind keypoints --epochs 30
```

`--kind` picks the architecture; `--weights` picks where fine-tuning starts from:
`base` (RF-DETR's own pretrained weights, the default), `mouselite` (the released
weights, usually the better starting point for mice) or a checkpoint path.

## If coming from DeepLabCut or Lightning Pose

Labels made in [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) or
[Lightning Pose](https://github.com/paninski-lab/lightning-pose) don't need to be
exported by hand. Point `train` at the project folder and add `--from`:

```bash
mouselite train dlc-project/ --kind keypoints --from deeplabcut --epochs 30
mouselite train lp-project/ --kind keypoints --from lightning-pose --epochs 30
```

The labeled frames are split 80/20 and written as a COCO dataset under
`<output-dir>/dataset/` before training starts, with the project's bodyparts as
keypoints. Images are symlinked rather than copied; pass `--no-symlink` to copy them.
Single- and multi-animal DeepLabCut projects both work, as do single- and multiview
Lightning Pose projects (each view's frames are more training images; a `visible`
column is honoured). `--from` is refused for
`--kind segmentation`, since pose projects have no masks; `mouselite list-formats`
prints the accepted formats.

## What you get

```
<output-dir>/
├── dataset/                   # only with --from
├── checkpoint_best_ema.pth    # best validation checkpoint (EMA weights) — use this
├── last_ema.pth               # EMA weights after the last epoch
├── last.ckpt                  # full trainer state, for resuming
├── metrics.csv
└── metrics.png                # loss and validation curves per epoch
```

Then run your video with the new weights:

```bash
mouselite run video.mp4 --kind keypoints --checkpoint output/train/checkpoint_best_ema.pth
```

## Options

| Option | Default | Meaning |
| ------ | ------- | ------- |
| `--kind` | *required* | `detection`, `segmentation` or `keypoints`. |
| `--size` | `medium` | Architecture size; ignored for `keypoints`. |
| `--weights` | `base` | `base`, `mouselite` or a checkpoint path. |
| `--from FORMAT` | — | Convert `DATASET_DIR` from this format first. See `list-formats`. |
| `--symlink` / `--no-symlink` | symlink | With `--from`: link or copy the images. |
| `--output-dir` | `output/train` | Where checkpoints, metrics and the converted dataset go. |
| `--epochs` | `10` | Training epochs. |
| `--batch-size` | `4` | Lower it if you run out of GPU memory. |
| `--lr` | RF-DETR's | Learning rate. |
| `--resolution` | RF-DETR's | Input resolution; must be divisible by RF-DETR's patch stride. |
| `--device` | auto | `cuda`, `cpu`, … |

Anything RF-DETR's `train` accepts beyond these — `keypoint_oks_sigmas`,
`grad_accum_steps`, … — is reachable from the [Python API](python-api.md#mouselitetrain).
