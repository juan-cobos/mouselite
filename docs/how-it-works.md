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

`retrack` never touches a model. It loads the export, feeds each frame's detections to
a fresh tracker in the order the images sort, writes `<stem>_retracked.mp4` from the
exported frames, and rewrites `track_id` on every annotation in place. Two consequences
of replaying from COCO rather than from the model:

- **Confidence is gone.** COCO has no score field, so trackers see every detection at
  `1.0`. ByteTrack-family trackers use a second, low-confidence association stage that
  never has anything to do on a retrack: tune with `retrack`, but expect small
  differences if you then re-run inference with the tracker you picked.
- **Only exported frames exist.** If you ran with `--every 3`, the retracked video has
  a third of the frames. It is always written at 30 fps, so it plays faster or slower
  than the source unless the source was 30 fps with `--every 1`.
