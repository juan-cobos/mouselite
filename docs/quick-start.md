# Quick start

No programming is required. Setting up means pasting a few lines into a *terminal*.

## 1. Open a terminal

=== "Windows"

    Press the Windows key, type `PowerShell` and press Enter.

=== "macOS"

    Press ⌘ Cmd + Space, type `Terminal` and press Enter.

=== "Linux"

    Press Ctrl + Alt + T, or open **Terminal** from your applications.

## 2. Install uv (first time only)

[uv](https://docs.astral.sh/uv/) is a free program that installs MouseLite and
everything it needs (Python included). Paste the line for
your system (Ctrl+V on Windows, ⌘ Cmd+V on macOS) and press Enter.

=== "Windows"

    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

=== "macOS / Linux"

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

When it has finished, **close the terminal and open a new one** (step 1), otherwise the
next command will not be found.

??? note "Already have a Python environment?"

    `pip install mouselite` (or `uv add mouselite` in a uv project) gives you a
    `mouselite` command, and you can drop the `uvx` prefix from every example below.

## 3. Start MouseLite (every time)

The same line works on every system:

```bash
uvx --from "mouselite[app]" mouselite app
```

The first time, this takes a few minutes: MouseLite and everything it needs are being
downloaded. After that it starts much faster.

Your browser then opens on the app. If it does not, look in the terminal for the line

```text
* Running on local URL:  http://127.0.0.1:7860
```

and open that address yourself.

**Keep the terminal open while you use MouseLite**: closing it stops the program.

<!-- TODO Juan: screenshot of the interface here, e.g. ![The MouseLite app](assets/app.png) -->

## 4. Analyse a video

1. Drag your video onto **Upload a video**.
2. In **Kind**, choose what MouseLite should mark: `keypoints` for points on the body
   (the default), `segmentation` for the outline of each mouse, or `detection` for a box
   around each mouse.
3. Set **Max animals** to the number of mice in the video.
4. Click **Run** and wait for the progress bar to finish.
5. Watch the **Annotated output** video, then download the results.

The other settings can stay as they are. On a slow computer, setting **Inference
stride** to 2 roughly halves the time, because MouseLite then analyses one frame out of
two. If two mice swap identities after a contact, choose another **Tracker** and click
**Retrack**: this takes seconds, since the mice are not detected again.

You get three kinds of result: `annotations.json` in the COCO format, `trajectories.csv`, and `summary.csv`, the
distance and speed of each animal.

Always watch the annotated video before trusting the numbers. If mice are missed or
points land in the wrong place, you can [fine-tune MouseLite](training.md) on a few
hundred labelled frames of your own videos, or reuse the labels of a DeepLabCut or
Lightning Pose project.

<!-- TODO Juan: one sentence on what the released models were trained on (camera view, arenas, lighting, coat colours), so users can judge whether their set-up is covered. -->

## 5. Many videos at once

Open a terminal (step 1) and run, with `PATH/TO/VIDEOS` replaced by the folder holding
them and `2` by the number of mice per video:

```bash
uvx mouselite run PATH/TO/VIDEOS --kind keypoints --top-k 2
```

Results land in an `output` folder created where the terminal is; add
`--output-dir PATH/TO/RESULTS` to put them elsewhere. 

To try another tracker without running inference again:

```bash
uvx mouselite retrack output/video_results/annotations.json --tracker bytetrack
```

That writes `output/video_retracked.mp4` and updates `track_id` in `annotations.json`
and the CSV files.

## If something goes wrong

- **"uvx" is not recognised, or "command not found"**: open a new terminal after
  installing uv (step 2).
- **The browser did not open**: open the address printed in the terminal, see step 3.
- **A video does not play in the browser**: the analysis is not affected. Download the
  annotated video and open it with [VLC](https://www.videolan.org/vlc/), which is free.
- **The installation is blocked on an institute computer**: show this page to your IT
  department.
- **Anything else**: [open an issue](https://github.com/juan-cobos/mouselite/issues) and
  paste what the terminal shows.

## Next

- Boxes only: `--kind detection`. Masks: `--kind segmentation`. Both take
  `--size nano|small|medium|large`; `keypoints` is a single model.
- Long video? `--every 2` runs inference on every other frame. `--compile` is slower to
  start and faster per frame. Watch it as it runs with `--show`.
- For cm instead of px, gap filling or smoothing, use
  [`mouselite.analysis`](python-api.md#mouseliteanalysis).
- Every option: [CLI reference](cli.md). What's in `annotations.json`:
  [COCO export](coco-export.md). Which tracker to pick: [Trackers](trackers.md).
