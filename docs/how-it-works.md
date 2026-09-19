# How it works

MouseLite is three modules and a CLI on top of them:

```
src/mouselite/
├── models.py     get_model(): pick an RF-DETR class, fetch weights, put it in inference mode
├── cli.py        typer commands: run, retrack, train, app, list-*
├── app.py        the optional Gradio demo (needs the [app] extra)
├── format.py     convert(): DeepLabCut / Lightning Pose projects → COCO (needs [convert])
├── train.py      train(): fine-tuning (needs [train])
└── analysis/     downstream statistics from an export
    ├── tracks.py       Tracks: the (frame, track) grid, from_coco(), summary tables
    ├── cleaning.py     interpolate(), smooth()
    ├── kinematics.py   speed(), distance_traveled(), heading(), …
    ├── masks.py        mask_centroids(), mask_axes()
    └── space.py        in_polygon(), bouts(), occupancy()
```

`pipeline.py` is the core. Everything else either builds its inputs (a model, a
tracker) or wraps it (the CLI, the demo).

## The `run` path, frame by frame

`Pipeline.run` hands a callback to `supervision.process_video`, which reads the video
with OpenCV and calls the callback once per frame with `(frame, frame_idx)`. Every
frame goes through the same steps:

```
frame ──▶ is frame_idx % every == 0 ?
              │ yes                                    │ no
              ▼                                        ▼
        model.predict(frame, threshold)         reuse last detections
              │
        KeyPoints → Detections  (keypoints only)
              │
        NMS (nms_threshold)
              │
        top-k by confidence     (if top_k set)
              │
        tracker.update()        (skipped when top_k == 1)
              │
        write frame to images/  +  append COCO image & annotations
              │
              └──────────────┬─────────────────────────┘
                             ▼
                   MetaAnnotator.annotate()  →  annotated video frame
                   (+ FPS HUD, + preview window)
```

Points worth knowing:

- **`every` is a stride, not a skip.** Inference and the export only happen on
  frames where `frame_idx % every == 0`, but the annotated video keeps *every* frame:
  in-between frames are drawn with the last predictions. The export therefore has
  one image per inference frame, and the tracker only ever sees those frames.
- **Keypoints are tracked as boxes.** RF-DETR's keypoint model returns
  `sv.KeyPoints` with the model's own box in `data["xyxy"]`. `_keypoints_to_detections`
  turns that into `sv.Detections` using that box — not one fitted around the
  keypoints — and stashes the keypoints in `data["keypoints_xy"]` /
  `data["keypoints_visible"]` so they survive tracking and reach the export.
- **NMS runs after the confidence threshold and before top-k**, so `top_k=2`
  means "the two best *distinct* animals", not two overlapping boxes on one.
- **`top_k=1` bypasses the tracker entirely.** With one animal there is nothing to
  associate, so every detection is given `tracker_id = 0` and `tracker.update` is
  never called. This is also why single-animal runs are indifferent to `--tracker`.
- **The tracker is reset at the start of each `run`**, so one `Pipeline` can process
  several videos in sequence without ids leaking from one to the next.

## Annotation

`MetaAnnotator` looks at what a `Detections` object carries and draws accordingly:
masks if `detections.mask` is present, then boxes, then a `#<track_id>` label, then
keypoints if `data["keypoints_xy"]` is present. Colours are looked up by
`tracker_id`, so an animal keeps its colour across frames. Supervision's
`VertexAnnotator` has no per-track colour mode, so keypoints are drawn one animal at
a time with the colour resolved the same way the box annotator resolves it.

The same annotator is used by `run` and `retrack`, so the two videos look alike.

## The `retrack` path

`retrack` never touches a model. It loads the export with
`sv.DetectionDataset.from_coco` (which decodes boxes and, for segmentation exports,
masks), reads keypoints back out of the raw annotation rows, and feeds each frame's
detections to a fresh tracker in the order the images sort. It writes
`<stem>_retracked.mp4` from the exported frames and rewrites `track_id` on every
annotation in place.

Two consequences of replaying from COCO rather than from the model:

- **Confidence is gone.** COCO has no score field and the export does not add one,
  so trackers see every detection at confidence `1.0`. ByteTrack-family trackers use
  a second, low-confidence association stage; on a retrack that stage never has
  anything to do, so a retrack with `bytetrack` is not bit-for-bit what `run
  --tracker bytetrack` would have produced. Tune with `retrack`, but expect small
  differences if you then re-run inference with the tracker you picked.
- **Only exported frames exist.** If you ran with `--every 3`, the retracked video
  has a third of the frames. It is always written at 30 fps, so it plays faster or
  slower than the source unless the source was 30 fps with `--every 1`.

Trackers may drop or reorder detections, so `retrack` tags each one with its row
index in the export before `update` and uses the surviving indices to write
`track_id` back. Rows the tracker dropped get `track_id = -1`, the value the
trackers themselves use for unconfirmed tracks.

## Models

`get_model` maps `(kind, size)` to an `rfdetr` class name via `models.MODELS`, fetches
the matching `mouselite-<kind>-<size>.pt` (or `mouselite-keypoints.pt`) from the
Hugging Face Hub with `hf_hub_download` unless `checkpoint` is given, constructs the
model with `pretrain_weights`, and calls `model.inference(...)` with the requested
`dtype`, `batch_size` and `compile` flag. The Hub download is cached in the usual
`huggingface_hub` cache, so it happens once per machine.
