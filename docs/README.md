---
title: Overview
---

<div class="ml-hero">
--8<-- "assets/logo.svg"
<p class="ml-tagline">Real-time mouse detection, segmentation and pose estimation.</p>
</div>

# Documentation

MouseLite finds every mouse in a video, keeps its identity from frame to frame, and
optionally places keypoints on its body. You get back an annotated video to check by
eye, a COCO export, and CSV tables of positions and per-track statistics.

| Page | Read it when you want to… |
| ---- | ------------------------- |
| [Quick start](quick-start.md) | install MouseLite and analyse your first video, step by step |
| [How it works](how-it-works.md) | understand what happens to a frame between the video and the export |
| [Trackers](trackers.md) | pick a tracker and tune it for your recordings |
| [Fine-tuning](training.md) | train on your own frames when the released models fall short |
| [CLI reference](cli.md) | see every command and option |
| [Python API](python-api.md) | drive the pipeline, and analyse an export, from Python |
| [Development](development.md) | run the tests, lint, or change the package |

Training and evaluation code for the released models lives in
[`paper/`](https://github.com/juan-cobos/mouselite/tree/main/paper) and has its own README.
