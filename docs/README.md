---
title: Overview
---

<div class="ml-hero">
--8<-- "assets/logo.svg"
<p class="ml-tagline">Real-time mouse detection, segmentation and pose estimation.</p>
</div>

# Documentation

The [top-level README](https://github.com/juan-cobos/mouselite#readme) covers install, the CLI essentials and the
Python API basics. The pages here go deeper.

| Page | Read it when you want to… |
| ---- | ------------------------- |
| [Quick start](quick-start.md) | run one video, retrack it, try the demo |
| [How it works](how-it-works.md) | understand what happens to a frame between the video and the export |
| [COCO export](coco-export.md) | analyse `annotations.json` in your own code |
| [Trackers](trackers.md) | pick a tracker and tune it for your recordings |
| [Fine-tuning](training.md) | train on your own frames when the released models fall short |
| [CLI reference](cli.md) | see every command and option |
| [Python API](python-api.md) | drive the pipeline from Python |
| [Development](development.md) | run the tests, lint, or change the package |

Training and evaluation code for the released models lives in
[`training/`](https://github.com/juan-cobos/mouselite/tree/main/paper) and has its own README.
