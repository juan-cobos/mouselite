"""Score DeepLabCut on the held-out split of a run, or of a fold zero-shot.

    uv run --project dlc python eval_dlc.py [--zero-shot] [run or split ...]

Without ``--zero-shot`` the argument is a run directory and what is scored are the
snapshots it adapted. With it, the argument is an export under ``datasets/`` and
what is scored are the SuperAnimal weights as DeepLabCut publishes them -- no
fitting anywhere, the same held-out splits, so the two columns line up.

Zero-shot collects into ``ZERO_SHOT_DIR``: one project per fold, holding the split
it was scored over and the predictions made against it, and every fold's scores
flat alongside them, named by fold so a plot can read the sweep as one series.
"""

import json
import sys
from pathlib import Path

import numpy as np
from deeplabcut.pose_estimation_pytorch.config import PoseConfig
from deeplabcut.pose_estimation_pytorch.data.cocoloader import COCOLoader
from deeplabcut.pose_estimation_pytorch.modelzoo.inference_helpers import (
    create_superanimal_inference_runners,
)
from mtmb.splits import to_deeplabcut

from src.data import (
    DATASETS_DIR,
    DLC_SUFFIX,
    EVERY,
    OKS_SIGMA,
    RUNS_DIR,
    build_leave_one_out,
    mouse_dataset,
)
from src.score import evaluate
from src.utils import report_metrics
from train_dlc import VALID_MODE, instances_per_image

SUPER_ANIMAL = "superanimal_topviewmouse"
MODEL_NAME = "hrnet_w32"
DETECTOR_NAME = "fasterrcnn_resnet50_fpn_v2"
BATCH_SIZE = 8
DETECTOR_BATCH_SIZE = 2

#: The split written into the project to be scored, and what names the outputs.
#: Not ``test``: the project already has a ``test.json`` that is really valid, and
#: overwriting it would cost the record of what the fit selected against.
HELDOUT = "heldout"

ZERO_SHOT_FLAG = "--zero-shot"
ZERO_SHOT_DIR = RUNS_DIR / "zeroshot"


def dlc_runs() -> list[Path]:
    """Every DeepLabCut run on disk that has weights."""
    runs = sorted(p for p in (Path.cwd() / "runs").glob(f"*{DLC_SUFFIX}") if p.is_dir())
    trained = [p for p in runs if any(p.glob("checkpoints/snapshot-*.pt"))]
    for skipped in sorted(set(runs) - set(trained)):
        print(f"  skipping {skipped.name}: no snapshot written")
    return trained


def zero_shot_project(dataset_dir: Path) -> Path:
    """Where one fold's zero-shot project is built, under ``ZERO_SHOT_DIR``."""
    return ZERO_SHOT_DIR / fold_key(dataset_dir)


def zero_shot_splits() -> list[Path]:
    """One export per held-out task, under this sweep's fold numbering."""
    return build_leave_one_out(EVERY, valid_mode=VALID_MODE)


def fold_key(dataset_dir: Path) -> str:
    """``fold_<n>_<assay>`` -- what names a zero-shot metrics file, and keys a plot.

    ``_valid_<assay>`` is dropped: it names the assay a fit validated on, and
    nothing is fitted zero-shot, so carrying it would only split one held-out
    assay across two names. An export outside the rotation keeps its own name,
    having no fold to be known by.
    """
    if "_fold_" not in dataset_dir.name:
        return dataset_dir.name
    fold = "fold_" + dataset_dir.name.split("_fold_", 1)[1]
    return fold.split("_valid_", 1)[0]


def split_dir(arg: str) -> Path:
    """A split named on the command line, by path or by export name."""
    for path in (Path(arg), DATASETS_DIR / arg):
        if path.is_dir():
            return path
    raise SystemExit(f"no split at {arg} or {DATASETS_DIR / arg}")


def superanimal_config(project_root: Path, max_individuals: int) -> Path:
    """Write the vendor model's own inference config into a zero-shot project."""
    config = PoseConfig.build_for_superanimal_inference(
        super_animal=SUPER_ANIMAL,
        model_name=MODEL_NAME,
        detector_name=DETECTOR_NAME,
        max_individuals=max_individuals,
    )
    path = project_root / "pytorch_config.yaml"
    config.to_yaml(path, overwrite=True)
    print(f"  vendor config     {path}")
    return path


def dataset_for_run(run_dir: Path) -> Path:
    """The RF-DETR export holding the split this run was cut from.

    Normally the run directory is the export's name plus ``_dlc``. A renamed run
    no longer says, so the manifest the export carried is read back:
    ``runs/pooled_dlc`` came from ``pooled_every10``, and going by the directory
    alone would look for a split that was never built.
    """
    names = [run_dir.name.removesuffix(DLC_SUFFIX)]
    if (manifest := run_dir / "manifest.json").exists():
        names.append(json.loads(manifest.read_text())["name"])

    for name in names:
        if (dataset_dir := DATASETS_DIR / name).is_dir():
            return dataset_dir
    raise SystemExit(f"no split under {DATASETS_DIR} for run {run_dir.name}")


def best_snapshot(checkpoints: Path, detector: bool = False) -> Path:
    """The weights to score, picked from what the run actually wrote."""
    snapshots = [
        path
        for path in checkpoints.glob("snapshot-*.pt")
        if ("detector" in path.name) == detector
    ]
    kind = "detector" if detector else "pose"
    if best := sorted(path for path in snapshots if "best" in path.name):
        chosen = best[-1]
    elif numbered := sorted(path for path in snapshots if "best" not in path.name):
        chosen = numbered[-1]  # zero-padded, so lexical order is epoch order
    else:
        raise SystemExit(f"no {kind} snapshot in {checkpoints}")

    print(f"  {kind + ' snapshot':<17} {chosen.name}")
    return chosen


def predict_split(
    customized_model_config: Path,
    customized_detector_checkpoint: Path | None,
    customized_pose_checkpoint: Path | None,
    images_dir: Path,
    image_names: list[str],
    max_individuals: int,
) -> dict[str, dict]:
    """Run the SuperAnimal model over a split.

    Either checkpoint left ``None`` is downloaded from the vendor's HuggingFace
    release instead of read off a run -- which is the whole of the zero-shot path.
    """
    pose_runner, detector_runner, _ = create_superanimal_inference_runners(
        superanimal_name=SUPER_ANIMAL,
        model_name=MODEL_NAME,
        detector_name=DETECTOR_NAME,
        max_individuals=max_individuals,
        batch_size=BATCH_SIZE,
        detector_batch_size=DETECTOR_BATCH_SIZE,
        customized_model_config=customized_model_config,
        customized_pose_checkpoint=customized_pose_checkpoint,
        customized_detector_checkpoint=customized_detector_checkpoint,
    )
    if detector_runner is None:
        raise SystemExit(f"no detector built for {DETECTOR_NAME}")

    paths = [str(images_dir / name) for name in image_names]
    print(f"  detecting over {len(paths)} images")
    boxes = detector_runner.inference(images=paths)
    print("  estimating pose")
    poses = pose_runner.inference(list(zip(paths, boxes)))

    return dict(zip(image_names, poses))


def add_heldout_split(run_dir: Path, dataset_dir: Path | None = None) -> Path:
    """Write the manifest's ``test`` split into the project, and link its images.

    ``dataset_dir`` says which export outright, for a project that was never
    trained in and so has no name or manifest to recover it from.
    """
    dataset = mouse_dataset()
    manifest = dataset.load_manifest((dataset_dir or dataset_for_run(run_dir)).name)
    to_deeplabcut(dataset, manifest, out_dir=run_dir, splits={HELDOUT: "test"})
    return run_dir / "annotations" / f"{HELDOUT}.json"


def keypoint_sigmas(annotations: Path) -> np.ndarray:
    """``OKS_SIGMA`` per keypoint, sized from the annotations' own category."""
    categories = json.loads(annotations.read_text())["categories"]
    if len(categories) != 1:
        raise SystemExit(f"expected one category, got {len(categories)}")
    return np.full(len(categories[0]["keypoints"]), OKS_SIGMA)


def main(project_dir: Path, dataset_dir: Path | None = None) -> dict[str, float]:
    """Predict and score one project, start to finish inside DeepLabCut.

    ``project_dir`` is the run itself for an adapted fit, and a fold's project
    under ``ZERO_SHOT_DIR`` zero-shot; ``dataset_dir``, given, is the export to
    score the vendor SuperAnimal weights over without any fitting.
    """
    run_dir = project_dir.resolve()
    zero_shot = dataset_dir is not None
    print(f"\nevaluating {dataset_dir.name + ' (zero-shot)' if zero_shot else run_dir}")
    annotations = add_heldout_split(run_dir, dataset_dir)

    checkpoints = run_dir / "checkpoints"
    # Zero-shot has no project to read a count from, so the fold's own held-out
    # frames set it: a fixed cap either drops animals or invites spurious ones.
    fold_individuals = instances_per_image(run_dir, HELDOUT) if zero_shot else None
    config = (
        superanimal_config(run_dir, fold_individuals)
        if zero_shot
        else checkpoints / "pytorch_config.yaml"
    )
    loader = COCOLoader(
        project_root=run_dir,
        model_config=config,
        train_json_filename=f"{HELDOUT}.json" if zero_shot else "train.json",
        test_json_filename=f"{HELDOUT}.json",
    )
    # Reads the images off disk and rewrites ``file_name`` to an absolute path,
    # which is the key ``predictions_to_coco`` looks predictions up by.
    data = loader.load_data("test")
    paths = {Path(image["file_name"]).name: image["file_name"] for image in data["images"]}
    individuals = fold_individuals or loader.get_dataset_parameters().max_num_animals
    print(f"  {len(paths)} images   max individuals {individuals}")

    predictions = predict_split(
        customized_model_config=config,
        customized_detector_checkpoint=(
            None if zero_shot else best_snapshot(checkpoints, detector=True)
        ),
        customized_pose_checkpoint=None if zero_shot else best_snapshot(checkpoints),
        images_dir=run_dir / "images",
        image_names=list(paths),
        max_individuals=individuals,
    )
    results = loader.predictions_to_coco(
        {paths[name]: prediction for name, prediction in predictions.items()},
        mode="test",
    )
    path = run_dir / f"{HELDOUT}_predictions.json"
    path.write_text(json.dumps(results), encoding="utf-8")
    print(f"  {len(results)} detections -> {path}")

    metrics = evaluate(annotations, results, keypoint_sigmas(annotations))
    report_metrics(
        metrics,
        ZERO_SHOT_DIR if zero_shot else run_dir,
        HELDOUT,
        f"_{fold_key(dataset_dir)}" if zero_shot else "",
    )
    return metrics


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != ZERO_SHOT_FLAG]

    if ZERO_SHOT_FLAG in sys.argv[1:]:
        splits = [split_dir(a) for a in args] or zero_shot_splits()
        scored = {
            fold_key(split): main(zero_shot_project(split), split) for split in splits
        }
    else:
        runs = [Path(a) for a in args] or dlc_runs()
        scored = {run.name: main(run) for run in runs}

    if len(scored) > 1:
        print(f"\n{'run':<52} {'box AP':>8} {'kp AP':>8} {'kp AP50':>8}")
        for name, m in scored.items():
            print(
                f"  {name:<50} {m['bbox/AP']:>8.4f} "
                f"{m['keypoints/AP']:>8.4f} {m['keypoints/AP50']:>8.4f}",
            )
