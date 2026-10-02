import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deeplabcut.pose_estimation_pytorch.config.pose import PoseConfig
from deeplabcut.pose_estimation_pytorch.modelzoo.inference_helpers import (
    create_superanimal_inference_runners,
)

from eval_dlc import (
    SUPER_ANIMAL,
    best_snapshot,
    dataset_for_run,
    fitted_models,
)

SPLIT = "test"
WARMUP = 30
DTYPE = "float16"
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = torch.cuda.is_available()


def default_run() -> Path:
    """The first trained DeepLabCut run on disk."""
    runs = sorted(
        p for p in (ROOT / "runs").glob("*_dlc") if any(p.glob("checkpoints/snapshot-*.pt"))
    )
    if not runs:
        raise SystemExit(f"no trained DeepLabCut run under {ROOT / 'runs'}")
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


def build_runners(run_dir: Path, compile: bool):
    """The detector and pose runners, both built for one image at a time."""
    checkpoints = run_dir / "checkpoints"
    model_name, detector_name = fitted_models(checkpoints / "pytorch_config.yaml")
    config = PoseConfig.from_any(checkpoints / "pytorch_config.yaml")
    individuals = config.metadata.individuals
    # One section, read by both runners.
    config.inference.compile.enabled = compile

    pose_runner, detector_runner, _ = create_superanimal_inference_runners(
        superanimal_name=SUPER_ANIMAL,
        model_name=model_name,
        detector_name=detector_name,
        max_individuals=len(individuals),
        batch_size=1,
        detector_batch_size=1,
        customized_model_config=config,
        customized_pose_checkpoint=best_snapshot(checkpoints),
        customized_detector_checkpoint=best_snapshot(checkpoints, detector=True),
    )
    if detector_runner is None:
        raise SystemExit(f"no detector built from {checkpoints}")
    return pose_runner, detector_runner, len(individuals)


def uncompile(runner) -> bool:
    """Unwrap the compiled model a runner holds, if that is what it is holding."""
    if (eager := getattr(runner.model, "_orig_mod", None)) is None:
        return False
    runner.model = eager
    return True


def warm_up(pose_runner, detector_runner, images, cuda: bool) -> bool:
    def run() -> None:
        for array, _ in images[:WARMUP]:
            boxes = detector_runner.inference(images=[array])
            pose_runner.inference([(array, boxes[0])])
        if cuda:
            torch.cuda.synchronize()

    try:
        run()
        return True
    except Exception as error:
        if not any(uncompile(runner) for runner in (detector_runner, pose_runner)):
            raise
        print(
            f"  torch.compile failed ({type(error).__name__}: {error}), eager instead",
        )

    run()
    return False


def measure(pose_runner, detector_runner, images, cuda: bool) -> list[dict]:
    """Detector then pose, timed apart, one image at a time."""
    records = []
    for i, (array, instances) in enumerate(images):
        start = time.perf_counter()
        boxes = detector_runner.inference(images=[array])
        if cuda:
            torch.cuda.synchronize()
        detected = time.perf_counter()
        poses = pose_runner.inference([(array, boxes[0])])
        if cuda:
            torch.cuda.synchronize()
        end = time.perf_counter()

        # What the pose stage was actually asked to crop, which is what its share
        # of the frame's cost scales with.
        scores = np.asarray(poses[0].get("bbox_scores", []), dtype=np.float32).reshape(
            -1,
        )
        records.append(
            {
                "ms": (end - start) * 1000,
                "detector_ms": (detected - start) * 1000,
                "pose_ms": (end - detected) * 1000,
                "instances": instances,
                "detections": int((scores > 0).sum()),
            },
        )
        print(f"  timed {i + 1}/{len(images)}", end="\r")

    print()
    return records


def summarise(records: list[dict]) -> dict:
    """Median and p95 rather than a mean, per stage and for the frame."""
    summary: dict[str, float] = {"images": len(records)}
    for key, prefix in (("ms", ""), ("detector_ms", "detector_"), ("pose_ms", "pose_")):
        ms = np.array([r[key] for r in records])
        summary |= {
            f"{prefix}median_ms": float(np.median(ms)),
            f"{prefix}mean_ms": float(ms.mean()),
            f"{prefix}p95_ms": float(np.percentile(ms, 95)),
        }
    total = np.array([r["ms"] for r in records])
    summary |= {
        "min_ms": float(total.min()),
        "max_ms": float(total.max()),
        "fps_median": float(1000 / np.median(total)),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", type=Path, default=None)
    parser.add_argument(
        "--no-compile",
        dest="compile",
        action="store_false",
        help="skip torch.compile and time the stages eager",
    )
    args = parser.parse_args()
    run_dir = (args.run or default_run()).resolve()

    dataset_dir = dataset_for_run(run_dir)
    annotations = dataset_dir / SPLIT / "_annotations.coco.json"
    print(f"\nmeasuring {run_dir.name} on {dataset_dir / SPLIT}")

    pose_runner, detector_runner, individuals = build_runners(run_dir, args.compile)
    images = load_images(dataset_dir / SPLIT, annotations)
    print(
        f"  {len(images)} images decoded, warm-up {WARMUP}, max individuals {individuals}",
    )

    cuda = torch.cuda.is_available()
    compiled = warm_up(pose_runner, detector_runner, images, cuda) and args.compile
    records = measure(pose_runner, detector_runner, images, cuda)
    summary = summarise(records)

    device = torch.cuda.get_device_name(0) if cuda else "cpu"
    payload = {
        "run": run_dir.name,
        "split": SPLIT,
        "batch_size": 1,
        "max_individuals": individuals,
        "device": device,
        "torch": torch.__version__,
        "dtype": DTYPE,
        "warmup": WARMUP,
        "compile": compiled,
        "tf32": True,
        **summary,
        "records": records,
    }
    path = run_dir / "latency_dlc.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n  device      {device}   {DTYPE}, compiled {compiled}")
    print(
        f"  median      {summary['median_ms']:.1f} ms   ({summary['fps_median']:.1f} FPS)",
    )
    print(f"    detector  {summary['detector_median_ms']:.1f} ms")
    print(f"    pose      {summary['pose_median_ms']:.1f} ms")
    print(f"  mean        {summary['mean_ms']:.1f} ms")
    print(f"  p95         {summary['p95_ms']:.1f} ms")
    print(f"  min / max   {summary['min_ms']:.1f} / {summary['max_ms']:.1f} ms")
    print(f"\n  {path}")


if __name__ == "__main__":
    main()
