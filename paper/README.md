# MouseLite paper

Training and evaluation code behind the MouseLite paper: fine-tuning RF-DETR
(detection, segmentation and the keypoints-preview head) and the DeepLabCut
SuperAnimal baseline on the Multi-Task Mouse Behaviour Dataset, plus the
scripts that scored them.

If you just want to run the released models, or fine-tune them on your own
recordings, use the `mouselite` package at the repository root instead
(`mouselite run`, `mouselite train`) — this directory is for reproducing the
paper.

## Layout

```
paper/
├── train.py            train the RF-DETR keypoints-preview model
├── train_all_kind.py   train every released detection/segmentation size
├── eval.py             score an RF-DETR run's best checkpoint
├── train_dlc.py        fine-tune the DeepLabCut SuperAnimal snapshots
├── eval_dlc.py         score a DeepLabCut run, or the SuperAnimal weights zero-shot
├── src/                data.py (splits), builder.py, score.py, utils.py -- shared by train.py/eval.py
├── scripts/            latency and param comparison
└── dlc/                DeepLabCut's own isolated environment (see below)
```

## Two environments, deliberately

DeepLabCut pins `numpy<2`, `matplotlib<3.9` and `albumentations<=1.4.3`
against RF-DETR's `numpy 2.x`, and ships no Python 3.13 wheels. The two
resolve into incompatible environments, so they get one each and meet only on
disk: `mtmb` writes a split, `train_dlc.py`/`eval_dlc.py` run against it in
the DLC environment, and predictions come back as COCO results for `eval.py`
to score in the RF-DETR environment -- one evaluator, one set of OKS sigmas,
for both model families.

```
uv sync                                    # RF-DETR side (train.py, eval.py, scripts/)
uv run python train.py

uv sync --project dlc                      # DeepLabCut side
uv run --project dlc python train_dlc.py
```

Always run from this directory (`paper/`), including the `--project dlc`
commands: `src.data` derives every path (splits, runs) from the working
directory, so invoking from `dlc/` itself would land them in the wrong place.
