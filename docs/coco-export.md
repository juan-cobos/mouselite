# The COCO export

`mouselite run` writes, next to the annotated video:

```
output/
└── <video-stem>_coco/
    ├── annotations.json
    └── images/
        ├── <video-stem>_000000.jpg
        ├── <video-stem>_000001.jpg
        └── …
```

`annotations.json` is a standard [COCO](https://cocodataset.org/#format-data)
detection/keypoint file with three MouseLite additions: `info.fps`,
`images[].frame_index` and `annotations[].track_id`. Anything that reads COCO
(supervision, pycocotools, FiftyOne, CVAT, …) can load it and will ignore the
extras.

## Layout

```json
{
  "info": { "fps": 15.0 },
  "licenses": [],
  "categories": [
    { "id": 1, "name": "mouse", "supercategory": "common-objects" }
  ],
  "images": [
    { "id": 1, "file_name": "cage_000000.jpg", "height": 480, "width": 640, "frame_index": 0 },
    { "id": 2, "file_name": "cage_000002.jpg", "height": 480, "width": 640, "frame_index": 2 }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [112.0, 203.5, 88.0, 61.0],
      "area": 5368.0,
      "segmentation": [],
      "iscrowd": 0,
      "track_id": 3,
      "keypoints": [141.0, 210.2, 2.0, 150.7, 218.9, 2.0, 197.3, 251.0, 0.0],
      "num_keypoints": 2
    }
  ]
}
```

### `info`

| Field | Meaning |
| ----- | ------- |
| `fps` | Frame rate of the *exported* frames: the source video's fps divided by `--every`. `retrack` reads this to write its video at the right speed. If an export has no `info.fps` (hand-built, or from a build before it was recorded), `retrack` assumes 30. |

### `images`

One entry per frame inference ran on — every frame with `--every 1`, one in N
otherwise. Ids start at 1 and increase in frame order.

| Field | Meaning |
| ----- | ------- |
| `id` | 1-based, sequential |
| `file_name` | `<video-stem>_<frame_index:06d>.jpg`, relative to `images/` |
| `width`, `height` | frame size in pixels |
| `frame_index` | **MouseLite extra.** 0-based index of the frame in the source video. With `--every N` these go 0, N, 2N, …; use this, not `id`, to line up with the original recording. |

### `categories`

One entry per model class, id `class_id + 1` as the COCO spec requires. The released
models have a single `mouse` class. The keypoint model's category does **not** carry
a `keypoints` name list or `skeleton`; keypoint order is the model's training order.

### `annotations`

One entry per detection. Ids start at 1 and are unique across the file.

| Field | Present for | Meaning |
| ----- | ----------- | ------- |
| `id`, `image_id`, `category_id` | all | standard COCO |
| `bbox` | all | `[x, y, width, height]` in pixels, top-left origin |
| `area` | all | mask pixel count for segmentation, otherwise `width * height` |
| `segmentation` | all | `[]` for detection and keypoints; for segmentation, a list of polygons (`[[x1, y1, x2, y2, …], …]`) or, when a mask has holes or several disjoint parts, an uncompressed RLE dict `{"counts": [...], "size": [h, w]}` with `iscrowd` set to 1 |
| `iscrowd` | all | `0`, except for RLE-encoded masks as above |
| `track_id` | all | **MouseLite extra.** Identity assigned by the tracker; the same animal keeps the same id across frames. `-1` means the tracker saw the detection but never confirmed it as a track. With `--top-k 1` every detection is `0`. Rewritten in place by `retrack`. |
| `keypoints` | keypoints | Flat `[x1, y1, v1, x2, y2, v2, …]`. Visibility `v` is `2` (visible) or `0` (not visible). MouseLite never emits `1` (labelled but occluded), because the model does not distinguish it. |
| `num_keypoints` | keypoints | Count of keypoints with `v > 0` |

There is **no confidence score** in the export. COCO's ground-truth format has none,
and `run` applies `--threshold`, NMS and `--top-k` before exporting, so what is in
the file is what was kept. See [How it works](how-it-works.md#the-retrack-path) for
what this means for `retrack`.

## Reading it

### With pandas

```python
import json
import pandas as pd

with open("output/cage_coco/annotations.json") as f:
    coco = json.load(f)

images = pd.DataFrame(coco["images"]).set_index("id")
ann = pd.DataFrame(coco["annotations"])
ann["frame_index"] = ann["image_id"].map(images["frame_index"])
ann[["x", "y", "w", "h"]] = pd.DataFrame(ann["bbox"].tolist(), index=ann.index)
ann["cx"] = ann["x"] + ann["w"] / 2
ann["cy"] = ann["y"] + ann["h"] / 2

# one trajectory per animal, ignoring unconfirmed detections
tracks = ann[ann["track_id"] >= 0].sort_values(["track_id", "frame_index"])
```

For keypoints, `ann["keypoints"]` is a list per row; reshape it to `(K, 3)`:

```python
import numpy as np

kp = np.stack(ann["keypoints"].map(np.asarray)).reshape(len(ann), -1, 3)
xy, visible = kp[..., :2], kp[..., 2] > 0
```

Time in seconds is `frame_index / source_fps`, where `source_fps` is the original
video's frame rate (`info.fps * every`), not `info.fps`.

### With supervision

```python
import supervision as sv

dataset = sv.DetectionDataset.from_coco(
    images_directory_path="output/cage_coco/images",
    annotations_path="output/cage_coco/annotations.json",
)
for image_path, image, detections in dataset:
    ...  # detections.xyxy, detections.mask (segmentation), detections.class_id
```

`from_coco` decodes boxes and masks but not the `track_id` or `keypoints` fields; read
those from the JSON directly, as `mouselite.tracker.retrack` does.

## Re-tracking rewrites the file

`mouselite retrack` updates `track_id` on every annotation in `annotations.json` **in
place**. Nothing else in the file changes. If you want to compare several trackers,
copy the export first or keep the retracked videos, which are what actually differ.
