# How it works

MouseLite wraps fine-tuned [RF-DETR](https://github.com/roboflow/rf-detr) models in a
video pipeline: it predicts per frame, links predictions across frames with a
multi-object tracker, writes an annotated video, and exports the predictions as a
COCO dataset you can re-track or analyse later.

## Models

Three model kinds, fine-tuned on the [MTMB](https://github.com/juan-cobos/mtmb)
dataset:

| `--kind`       | `--size`                           | Output                     |
| -------------- | ---------------------------------- | -------------------------- |
| `detection`    | `nano`, `small`, `medium`, `large` | bounding boxes             |
| `segmentation` | `nano`, `small`, `medium`, `large` | boxes + instance masks     |
| `keypoints`    | single checkpoint                  | boxes + pose keypoints     |

`get_model` fetches the matching checkpoint from the Hugging Face Hub on first use and
returns it ready for inference; the download is cached, so it happens once per machine.
Tracking is handled by [trackers](https://github.com/roboflow/trackers).

## The `run` path, frame by frame

`Pipeline.run` hands a callback to `supervision.process_video`, which reads the video
with OpenCV and calls the callback once per frame with `(frame, frame_idx)`:

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
        append COCO image (with frame_index) & annotations
              │
              └──────────────┬─────────────────────────┘
                             ▼
                   MetaAnnotator.annotate()  →  annotated video frame
                   (+ FPS HUD, + preview window)
```

Points worth knowing:

- **`every` is a stride, not a skip.** Inference and the export only happen on
  frames where `frame_idx % every == 0`, but the annotated video keeps *every* frame:
  in-between frames are drawn with the last predictions, and the tracker only ever
  sees inference frames.
- **NMS runs after the confidence threshold and before top-k**, so `top_k=2`
  means "the two best *distinct* animals", not two overlapping boxes on one.
- **`top_k=1` bypasses the tracker entirely.** With one animal there is nothing to
  associate, so every detection is given `tracker_id = 0`. This is why single-animal
  runs are indifferent to `--tracker`.
- **Keypoints are tracked as boxes** — the model's own box, not one fitted around the
  keypoints — with the keypoints carried alongside so they survive tracking and reach
  the export.

`MetaAnnotator` then draws whatever a detection carries: masks, boxes, a `#<track_id>`
label, keypoints. Colours are looked up by `tracker_id`, so an animal keeps its colour
across frames. `run` and `retrack` share the annotator, so the two videos look alike.

## The `retrack` path

`retrack` never touches a model. It reads the source video and the export side by
side, matching frames to COCO images by `frame_index`, feeds each exported frame's
detections to a fresh tracker, writes `<stem>_retracked.mp4`, and rewrites `track_id`
on every annotation in place. Each annotation's `score` (the model's confidence) is
fed back to the tracker, so confidence-based trackers such as ByteTrack behave as they
would on a live run. Exports written before `score` was added have none, and every
detection is then seen at `1.0`.

- **`every` carries over.** If you ran with `--every 3`, the tracker sees a third of
  the frames, as it did during `run`; the retracked video still has every frame of the
  source, at its frame rate, with in-between frames drawn from the last exported one.
