# Quick start

MouseLite runs from the terminal through [uv](https://docs.astral.sh/uv/), which
takes care of Python for you: no separate Python install, no environments. Install
uv once:

=== "macOS / Linux"

    Open **Terminal** and paste:

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "Windows"

    Open **PowerShell** and paste:

    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

Close and reopen the terminal so it picks up the new command.

```bash
uvx mouselite --help
```

There is nothing else to install; the rest of this page prefixes every command with
`uvx` for that reason.

??? note "Already have a Python environment?"

    `pip install mouselite` (or `uv add mouselite` in a uv project) gives you a
    `mouselite` command, and you can drop the `uvx` prefix from the examples below.

Run the pose model on a video with two mice:

```bash
uvx mouselite run video.mp4 --kind keypoints --top-k 2
```

The first run downloads the weights from the Hugging Face Hub; after that it goes
straight to inference. When it finishes you have:

```
output/
└── video_results/
    ├── video_annotated.mp4      # boxes, track ids and skeletons drawn on every frame
    ├── annotations.json         # one entry per animal per frame, with track_id and keypoints
    └── images/                  # the frames inference ran on
```

Open the video to check the result. If the two animals swap ids when they cross,
try another tracker on the same predictions, no inference this time:

```bash
uvx mouselite retrack output/video_results/annotations.json --tracker bytetrack
```

That writes `output/video_retracked.mp4` and updates `track_id` in `annotations.json`.

## The demo

The same run/retrack loop is available as a browser UI. It needs the `app` extra,
which `uvx` pulls in the same way:

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
- Numbers out of it — distance, speed, time immobile:
  `mouselite analyze output/video_results/annotations.json`.
- Every option: [CLI reference](cli.md). What's in `annotations.json`:
  [COCO export](coco-export.md). Which tracker to pick: [Trackers](trackers.md).
