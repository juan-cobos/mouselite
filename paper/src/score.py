"""Score COCO results against a split -- one scorer, both model families."""

from pathlib import Path

import numpy as np
from faster_coco_eval import COCO
from faster_coco_eval.core.faster_eval_api import COCOeval

#: ``COCOeval.stats`` in order, per iou type. Keypoints has ten rows to the box's
#: twelve because COCO drops the ``small`` area range for keypoints -- landmark
#: annotations on tiny instances are not considered reliable enough to score.
BBOX_STATS = (
    "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
    "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large",
)  # fmt: skip
KEYPOINT_STATS = (
    "AP", "AP50", "AP75", "AP_medium", "AP_large",
    "AR", "AR50", "AR75", "AR_medium", "AR_large",
)  # fmt: skip

#: COCO's own caps, and what its summary rows are defined in terms of: the box
#: rows report AR at each of the three, the keypoint rows are all at 20. Both
#: families are well inside these -- lowering the cap to 4 moves RF-DETR's box AP
#: by 0.007 and DeepLabCut's not at all -- so nothing here is truncated.
MAX_DETS = {"bbox": [1, 10, 100], "keypoints": [20]}

#: The fields COCO's results format defines per iou type, beside ``image_id``,
#: ``category_id`` and ``score``. Keypoint records carry no ``bbox`` on purpose:
#: ``loadRes`` picks its branch by which keys are present, and a ``bbox`` makes it
#: take each detection's area from the box rather than from the keypoint extent,
#: which shifts detections across the medium/large boundary. It moves only
#: ``AP_medium`` and ``AP_large``, but it moves them differently for a family that
#: emits boxes with its keypoints than for one that does not.
RESULT_FIELDS = {"bbox": ("bbox",), "keypoints": ("keypoints",)}

#: What COCO reports for a row it cannot compute. Returned throughout when a run
#: detected nothing at all, so a dead fold scores rather than crashing the sweep.
EMPTY_STAT = -1.0


def _results_for(iou_type: str, results: list[dict]) -> list[dict]:
    """One iou type's records, carrying only the fields COCO defines for it."""
    return [
        {
            "image_id": result["image_id"],
            "category_id": result["category_id"],
            "score": result["score"],
            **{field: result[field] for field in RESULT_FIELDS[iou_type]},
        }
        for result in results
    ]


def evaluate(
    annotations: Path,
    results: list[dict],
    sigmas: list[float] | np.ndarray,
) -> dict[str, float]:
    """Score COCO results, boxes and keypoints, against COCO annotations.

    ``results`` is the COCO results format -- ``image_id``, ``category_id``,
    ``score``, plus ``bbox`` (xywh) and ``keypoints`` (flat ``x, y, v`` triples).
    Images absent from it are simply undetected: ``params.imgIds`` covers every
    image in ``annotations``, so they count as false negatives rather than going
    missing from the run.

    ``sigmas`` is passed rather than defaulted: leaving it to a fallback is what
    silently scored the early runs at COCO's 17 person sigmas.
    """
    coco_gt = COCO(str(annotations))
    metrics: dict[str, float] = {}

    for iou_type, names in (("bbox", BBOX_STATS), ("keypoints", KEYPOINT_STATS)):
        if not results:
            metrics |= {f"{iou_type}/{name}": EMPTY_STAT for name in names}
            continue

        coco_dt = coco_gt.loadRes(_results_for(iou_type, results))
        kwargs = {"kpt_oks_sigmas": sigmas} if iou_type == "keypoints" else {}
        coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type, **kwargs)
        coco_eval.params.maxDets = MAX_DETS[iou_type]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        metrics |= {
            f"{iou_type}/{name}": float(value)
            for name, value in zip(names, coco_eval.stats)
        }

    return metrics
