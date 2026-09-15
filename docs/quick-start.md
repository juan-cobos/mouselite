# Quick start

We recommend [uv](https://docs.astral.sh/uv/) — it takes care of the Python version
(≥ 3.11) and the environment. Add MouseLite to a project with

```bash
uv add mouselite
```

or install it into an environment of your own with `pip install mouselite`. Either
way you get a `mouselite` command.

If you don't want to install anything, `uvx` runs the command straight from a cache:

```bash
uvx mouselite run video.mp4 --kind keypoints --top-k 2
```

is the same as

```bash
mouselite run video.mp4 --kind keypoints --top-k 2
```

after installing. The rest of this page shows the installed form; prefix with `uvx`
if you went the other way.

Run the pose model on a video with two mice:

```bash
mouselite run video.mp4 --kind keypoints --top-k 2
```

The first run downloads the weights from the Hugging Face Hub; after that it goes
straight to inference. When it finishes you have:

```
output/
├── video_annotated.mp4          # boxes, track ids and skeletons drawn on every frame
└── video_coco/
    ├── annotations.json         # one entry per animal per frame, with track_id and keypoints
    └── images/                  # the frames inference ran on
```

Open the video to check the result. If the two animals swap ids when they cross,
try another tracker on the same predictions, no inference this time:

```bash
mouselite retrack output/video_coco/annotations.json --tracker ocsort
```

That writes `output/video_retracked.mp4` and updates `track_id` in `annotations.json`.

## The demo

The same run/retrack loop is available as a browser UI. The Gradio dependency is an
optional extra, which `uvx` can pull in without installing anything:

```bash
uvx --from "mouselite[app]" mouselite app
```

Open the printed local URL, upload a video, pick a model and tracker, hit **Run**,
then **Retrack** with a different tracker. Download the annotations when ready.

## Next

- Boxes only: `--kind detection`. Masks: `--kind segmentation`. Both take
  `--size nano|small|medium|large`; `keypoints` is a single model.
- Long video? `--every 2` runs inference on every other frame. `--compile` is
  slower to start and faster per frame.
- Watch it as it runs: `--show`.
- Every option: [CLI reference](cli.md). What's in `annotations.json`:
  [COCO export](coco-export.md). Which tracker to pick: [Trackers](trackers.md).
