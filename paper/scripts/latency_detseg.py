"""Per-image latency for the detection and segmentation runs, with annotated examples."""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import supervision as sv
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
# Run as a script, so the project is not importable without this.
sys.path.insert(0, str(ROOT))

from latency import BATCH_SIZE, DEVICE, DTYPE, WARMUP, load_images, summarise
from rfdetr.detr import RFDETR

from eval import SPLIT, best_checkpoint

KINDS = ("detection", "segmentation")
#: Smallest first, so a table reads along the latency/accuracy trade-off.
SIZES = ("nano", "small", "medium", "large")
THRESHOLD = 0.5

#: The pooled split holds nine assays, so this many examples is one per arena.
EXAMPLES = 9
EXAMPLES_DIR = "examples"
RESULTS = "latency_detseg.json"


def kind_and_size(run: Path) -> tuple[str, str]:
    """What a run trained, read off the ``{kind}_{size}_{split}`` directory name."""
    parts = run.name.split("_")
    return parts[0], parts[1] if len(parts) > 1 else ""


def discover_runs() -> list[Path]:
    """Every finished detection and segmentation run on disk, nano to large."""
    runs = [
        run
        for kind in KINDS
        for run in (ROOT / "runs").glob(f"{kind}_*")
        if any(run.glob("checkpoint_best_*.pth"))
    ]
    if not runs:
        raise SystemExit(f"no trained {' or '.join(KINDS)} run under {ROOT / 'runs'}")

    def order(run: Path) -> tuple[int, int, str]:
        kind, size = kind_and_size(run)
        # An unknown size sorts after the four the models ship, not before.
        sizes = SIZES + (size,)
        return KINDS.index(kind), sizes.index(size), run.name

    return sorted(runs, key=order)


def dataset_for_run(run: Path) -> Path:
    """The split a run was fitted on, read back from the config it wrote."""
    config = json.loads((run / "training_config.json").read_text())
    dataset_dir = ROOT / "datasets" / Path(config["train_config"]["dataset_dir"]).name
    if not dataset_dir.is_dir():
        raise SystemExit(f"no split at {dataset_dir} for run {run.name}")
    return dataset_dir


def load_model(run: Path, device: str, dtype: str, batch_size: int) -> RFDETR:
    """The run's best checkpoint, rebuilt as the variant it was trained as."""
    model = RFDETR.from_checkpoint(
        best_checkpoint(run),
        trust_checkpoint=True,
        device=device,
    )
    print(
        f"  model {type(model).__name__} (resolution {model.model_config.resolution})",
    )
    model.inference(compile=True, batch_size=batch_size, dtype=dtype)
    return model


def predict(model: RFDETR, array: np.ndarray, threshold: float) -> sv.Detections:
    """One image through the model, as the single ``Detections`` it holds."""
    detections = model.predict(array, threshold=threshold, include_source_image=False)
    return detections[0] if isinstance(detections, list) else detections


def measure(
    model: RFDETR,
    images: list[tuple[np.ndarray, int]],
    threshold: float,
) -> list[dict]:
    """One timed forward pass per image, synchronised, after a warm-up."""
    cuda = torch.cuda.is_available()
    for array, _ in images[:WARMUP]:
        predict(model, array, threshold)
    if cuda:
        torch.cuda.synchronize()

    records = []
    for i, (array, instances) in enumerate(images):
        start = time.perf_counter()
        detections = predict(model, array, threshold)
        if cuda:
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        records.append(
            {"ms": elapsed, "instances": instances, "detections": len(detections)},
        )
        print(f"  timed {i + 1}/{len(images)}", end="\r")

    print()
    return records


def annotate(image: np.ndarray, detections: sv.Detections) -> np.ndarray:
    """The predictions drawn over the frame: mask fill, box, class and score."""
    if len(detections) == 0:
        return image

    size = (image.shape[1], image.shape[0])
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=size)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=size)
    lookup = sv.ColorLookup.INDEX

    annotated = image.copy()
    if detections.mask is not None:
        annotated = sv.MaskAnnotator(opacity=0.5, color_lookup=lookup).annotate(
            annotated,
            detections,
        )
    annotated = sv.BoxAnnotator(thickness=thickness, color_lookup=lookup).annotate(
        annotated,
        detections,
    )
    labels = [
        f"{name} {score:.2f}"
        for name, score in zip(detections["class_name"], detections.confidence)
    ]
    return sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.TOP_LEFT,
        color_lookup=lookup,
    ).annotate(annotated, detections, labels)


def example_images(coco: dict, count: int) -> list[dict]:
    """``count`` test images, taken one assay at a time."""
    by_task: dict[str, list[dict]] = defaultdict(list)
    for image in coco["images"]:
        by_task[image.get("task", "")].append(image)

    tasks = sorted(by_task)
    picked = []
    for i in range(count):
        frames = by_task[tasks[i % len(tasks)]]
        picked.append(frames[(i // len(tasks)) % len(frames)])
    return picked


def save_examples(
    model: RFDETR,
    split_dir: Path,
    coco: dict,
    output_dir: Path,
    count: int,
    threshold: float,
) -> Path:
    """Annotate a handful of test frames and write them beside the weights."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for image in example_images(coco, count):
        array = np.asarray(Image.open(split_dir / image["file_name"]).convert("RGB"))
        detections = predict(model, array, threshold)
        path = output_dir / f"{Path(image['file_name']).stem}.jpg"
        Image.fromarray(annotate(array, detections)).save(path, quality=95)
        print(f"  {path.name:<40} {len(detections)} detections")
    return output_dir


def run_one(run: Path, args: argparse.Namespace) -> dict:
    """Time one run over the test split, then draw its examples."""
    dataset_dir = dataset_for_run(run)
    split_dir = dataset_dir / SPLIT
    coco = json.loads((split_dir / "_annotations.coco.json").read_text())

    print(f"\nmeasuring {run.name} on {split_dir}")
    model = load_model(run, args.device, args.dtype, args.batch_size)

    images = load_images(split_dir, split_dir / "_annotations.coco.json")
    print(
        f"  {len(images)} images decoded, warm-up {WARMUP}, threshold {args.threshold}",
    )
    records = measure(model, images, args.threshold)
    summary = summarise(records)

    kind, size = kind_and_size(run)
    payload = {
        "run": run.name,
        "kind": kind,
        "size": size,
        "model": type(model).__name__,
        "split": SPLIT,
        "batch_size": args.batch_size,
        "threshold": args.threshold,
        "resolution": model.model_config.resolution,
        "warmup": WARMUP,
        "device": args.device,
        "torch": torch.__version__,
        "dtype": args.dtype,
        **summary,
    }
    path = run / "latency_rfdetr.json"
    path.write_text(
        json.dumps(payload | {"records": records}, indent=2),
        encoding="utf-8",
    )

    print(
        f"  median      {summary['median_ms']:.1f} ms   ({summary['fps_median']:.1f} FPS)",
    )
    print(f"  mean        {summary['mean_ms']:.1f} ms")
    print(f"  p95         {summary['p95_ms']:.1f} ms")
    print(f"  min / max   {summary['min_ms']:.1f} / {summary['max_ms']:.1f} ms")
    print(f"  {path}")

    if args.examples:
        print(f"\n  {args.examples} examples at threshold {args.threshold}")
        directory = save_examples(
            model,
            split_dir,
            coco,
            run / EXAMPLES_DIR,
            args.examples,
            args.threshold,
        )
        print(f"  {directory}")

    # Eight models compile in one process, so let each one go before the next.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def save_results(summaries: list[dict], path: Path) -> Path:
    """The whole sweep in one file, keyed by kind and then by size."""
    results: dict[str, dict[str, dict]] = defaultdict(dict)
    for summary in summaries:
        results[summary["kind"]][summary["size"]] = summary

    payload = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "results": {kind: results[kind] for kind in KINDS if kind in results},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def report(summaries: list[dict], args: argparse.Namespace) -> None:
    """The sweep as one table, a block per kind and a row per size."""
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"\n  {device}   {args.dtype}, batch {args.batch_size}\n")
    print(f"  {'size':<12} {'res':>5} {'median':>10} {'p95':>10} {'FPS':>7}")
    for kind in KINDS:
        rows = [summary for summary in summaries if summary["kind"] == kind]
        if not rows:
            continue
        print(f"  {kind}")
        for summary in rows:
            print(
                f"    {summary['size']:<10} {summary['resolution']:>5}"
                f" {summary['median_ms']:>9.1f}ms {summary['p95_ms']:>9.1f}ms"
                f" {summary['fps_median']:>7.1f}",
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", type=Path, help="run dirs; default all")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--dtype", type=str, default=DTYPE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument(
        "--examples",
        type=int,
        default=EXAMPLES,
        help="annotated frames; 0 to skip",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs" / RESULTS,
        help="where the collected results are written",
    )
    args = parser.parse_args()

    runs = args.runs or discover_runs()
    summaries = [run_one(run, args) for run in runs]

    report(summaries, args)
    print(f"\n  {save_results(summaries, args.output)}")


if __name__ == "__main__":
    main()
