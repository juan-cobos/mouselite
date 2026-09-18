"""Per-image inference latency for an RF-DETR run."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
# Run as a script, so the project is not importable without this. The model has
# to be built by the project's own builder: latency measured on a different
# geometry than the accuracy numbers would not belong on the same curve.
sys.path.insert(0, str(ROOT))

from rfdetr.datasets import infer_coco_keypoint_schema

from eval import SPLIT, THRESHOLD, best_checkpoint
from src.builder import RESOLUTION, make_rfdetr_keypoints
from src.data import dataset_for_run

WARMUP = 30
BATCH_SIZE = 1
DTYPE = "float16"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def default_run() -> Path:
    """The first run on disk, so a bare invocation measures something that exists.

    Latency is a property of the architecture at a fixed resolution, not of which
    split trained the weights, so any finished run answers the question.
    """
    runs = sorted((ROOT / "runs").glob("keypoint_preview_*"))
    if not runs:
        raise SystemExit(f"no RF-DETR run under {ROOT / 'runs'}")
    return runs[0]


def load_images(split_dir: Path, annotations: Path) -> list[tuple[np.ndarray, int]]:
    """Every image in the split as RGB, with how many instances it holds."""
    coco = json.loads(annotations.read_text())
    counts: dict[int, int] = {}
    for ann in coco["annotations"]:
        counts[ann["image_id"]] = counts.get(ann["image_id"], 0) + 1

    images = []
    for image in coco["images"]:
        array = np.asarray(Image.open(split_dir / image["file_name"]).convert("RGB"))
        images.append((array, counts.get(image["id"], 0)))
    return images


def measure(
    model,
    images: list[tuple[np.ndarray, int]],
    threshold: float,
) -> list[dict]:
    """One timed forward pass per image, synchronised, after a warm-up."""
    cuda = torch.cuda.is_available()
    for array, _ in images[:WARMUP]:
        model.predict(array, threshold=threshold, include_source_image=False)
    if cuda:
        torch.cuda.synchronize()

    records = []
    for i, (array, instances) in enumerate(images):
        start = time.perf_counter()
        result = model.predict(array, threshold=threshold, include_source_image=False)
        if cuda:
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        if isinstance(result, list):
            result = result[0]
        detections = 0 if result.class_id is None else len(result.xy)
        records.append(
            {"ms": elapsed, "instances": instances, "detections": detections},
        )
        print(f"  timed {i + 1}/{len(images)}", end="\r")

    print()
    return records


def summarise(records: list[dict]) -> dict:
    """Median and p95 rather than a mean: the tail is what a pipeline stalls on."""
    ms = np.array([r["ms"] for r in records])
    return {
        "images": len(ms),
        "median_ms": float(np.median(ms)),
        "mean_ms": float(ms.mean()),
        "p95_ms": float(np.percentile(ms, 95)),
        "min_ms": float(ms.min()),
        "max_ms": float(ms.max()),
        "fps_median": float(1000 / np.median(ms)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--dtype", type=str, default=DTYPE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", type=str, default=DEVICE)

    args = parser.parse_args()
    args.run = args.run or default_run()

    dataset_dir = dataset_for_run(args.run)
    annotations = dataset_dir / SPLIT / "_annotations.coco.json"
    schema = infer_coco_keypoint_schema(
        str(dataset_dir / "train" / "_annotations.coco.json"),
    )
    model = make_rfdetr_keypoints(schema, RESOLUTION, weights=best_checkpoint(args.run))
    model.inference(compile=True, batch_size=BATCH_SIZE, dtype=DTYPE)

    print(f"\nmeasuring {args.run.name} on {dataset_dir / SPLIT}")
    images = load_images(dataset_dir / SPLIT, annotations)
    print(
        f"  {len(images)} images decoded, warm-up {WARMUP}, threshold {args.threshold}",
    )

    records = measure(model, images, args.threshold)
    summary = summarise(records)

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    payload = {
        "run": args.run.name,
        "split": SPLIT,
        "batch_size": args.batch_size,
        "threshold": args.threshold,
        "resolution": RESOLUTION,
        "warmup": WARMUP,
        "device": args.device,
        "torch": torch.__version__,
        "dtype": args.dtype,
        **summary,
        "records": records,
    }
    path = args.run / "latency_rfdetr.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n  device      {device}")
    print(
        f"  median      {summary['median_ms']:.1f} ms   ({summary['fps_median']:.1f} FPS)",
    )
    print(f"  mean        {summary['mean_ms']:.1f} ms")
    print(f"  p95         {summary['p95_ms']:.1f} ms")
    print(f"  min / max   {summary['min_ms']:.1f} / {summary['max_ms']:.1f} ms")
    print(f"\n  {path}")


if __name__ == "__main__":
    main()
