"""Score a finished run's best checkpoint on the pooled split's test set."""

import sys
from pathlib import Path

import numpy as np
from faster_coco_eval import COCO
from PIL import Image
from rfdetr import RFDETRKeypointPreview
from rfdetr.datasets import infer_coco_keypoint_schema
from src.builder import RESOLUTION, make_rfdetr_keypoints
from src.data import (
    EVERY,
    LOO_NAME,
    RUN_PREFIX,
    RUNS_DIR,
    build_pooled,
    dataset_for_run,
    oks_sigmas,
    run_dir,
)
from src.score import evaluate
from src.utils import report_metrics

SPLIT = "test"
CHECKPOINTS = (
    "checkpoint_best_ema.pth",
    "checkpoint_best_regular.pth",
    "checkpoint_best_total.pth",
)
THRESHOLD = 0.01
BATCH_SIZE = 8


def best_checkpoint(output_dir: Path) -> Path:
    """The weights to score, picked from what the run actually wrote."""
    for name in CHECKPOINTS:
        if (path := output_dir / name).exists():
            print(f"  checkpoint {path.name}  {path.stat().st_size / 1e6:.0f} MB")
            return path
    raise SystemExit(
        f"no checkpoint in {output_dir}; looked for {', '.join(CHECKPOINTS)}",
    )


def predict_split(
    model: RFDETRKeypointPreview,
    split_dir: Path,
    coco: COCO,
) -> list[dict]:
    """Run the model over a split, as COCO result records."""
    category_ids = sorted(coco.cats)
    images = [coco.imgs[i] for i in sorted(coco.imgs)]
    predictions: list[dict] = []

    for start in range(0, len(images), BATCH_SIZE):
        batch = images[start : start + BATCH_SIZE]
        arrays = [
            np.asarray(Image.open(split_dir / image["file_name"]).convert("RGB"))
            for image in batch
        ]
        results = model.predict(arrays, threshold=THRESHOLD, include_source_image=False)
        if not isinstance(results, list):
            results = [results]

        for image, keypoints in zip(batch, results):
            if keypoints.class_id is None or len(keypoints.xy) == 0:
                continue

            class_id = np.asarray(keypoints.class_id, dtype=int)
            keep = class_id < len(category_ids)
            # xyxy out of the model, xywh into COCO.
            boxes = np.asarray(keypoints.data["xyxy"], dtype=np.float32)[keep]
            boxes[:, 2:] -= boxes[:, :2]
            scores = np.asarray(keypoints.detection_confidence, dtype=np.float32)[keep]
            xy = np.asarray(keypoints.xy, dtype=np.float32)[keep]
            confidence = np.asarray(keypoints.keypoint_confidence, dtype=np.float32)
            points = np.concatenate([xy, confidence[keep][..., None]], axis=-1)

            for label, box, score, instance in zip(
                class_id[keep],
                boxes,
                scores,
                points,
            ):
                predictions.append(
                    {
                        "image_id": int(image["id"]),
                        "category_id": int(category_ids[label]),
                        "bbox": box.tolist(),
                        "keypoints": instance.reshape(-1).tolist(),
                        "score": float(score),
                    },
                )
        print(
            f"  predicted {min(start + BATCH_SIZE, len(images))}/{len(images)}",
            end="\r",
        )

    print()
    return predictions


def main(output_dir: Path | None = None) -> dict[str, float]:
    """Score one run's best checkpoint on the split it was not trained on.

    Given a run directory the split comes from its name, so a leave-one-out fold
    is scored on its own held-out assay. Given nothing, the pooled split is built
    and scored -- the default run.
    """
    if output_dir is None:
        dataset_dir = build_pooled(EVERY)
        output_dir = run_dir(dataset_dir)
    else:
        dataset_dir = dataset_for_run(output_dir)
    print(f"\nevaluating {output_dir} on {dataset_dir / SPLIT}")

    schema = infer_coco_keypoint_schema(
        str(dataset_dir / "train" / "_annotations.coco.json"),
    )
    model = make_rfdetr_keypoints(
        schema,
        RESOLUTION,
        weights=best_checkpoint(output_dir),
    )
    annotations = dataset_dir / SPLIT / "_annotations.coco.json"
    predictions = predict_split(model, dataset_dir / SPLIT, COCO(str(annotations)))

    metrics = evaluate(annotations, predictions, oks_sigmas(schema))
    report_metrics(metrics, output_dir, SPLIT)
    return metrics


def discover_folds() -> list[Path]:
    """Every finished leave-one-out fold under ``runs/``, in fold order.

    Scoring one fold says little on its own -- the point of the sweep is the
    spread across held-out assays -- so a bare invocation scores all of them.
    """
    return sorted(
        run
        for run in RUNS_DIR.glob(f"{RUN_PREFIX}_{LOO_NAME}_*")
        if any((run / name).exists() for name in CHECKPOINTS)
    )


if __name__ == "__main__":
    # ``None`` is the fallback rather than the default: with no fold fitted yet
    # there is nothing to sweep, and the pooled split is what a bare run means.
    runs = [Path(a) for a in sys.argv[1:]] or discover_folds() or [None]
    scored = {(r.name if r else "pooled"): main(r) for r in runs}

    if len(scored) > 1:
        print(f"\n{'run':<52} {'box AP':>8} {'kp AP':>8} {'kp AP50':>8}")
        for name, m in scored.items():
            print(
                f"  {name:<50} {m['bbox/AP']:>8.4f} "
                f"{m['keypoints/AP']:>8.4f} {m['keypoints/AP50']:>8.4f}",
            )
