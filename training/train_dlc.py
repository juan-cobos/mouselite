import json
from pathlib import Path

from deeplabcut.pose_estimation_pytorch.config import PoseConfig
from deeplabcut.pose_estimation_pytorch.modelzoo.train_from_coco import adaptation_train
from deeplabcut.pose_estimation_pytorch.modelzoo.utils import (
    get_super_animal_snapshot_path,
)
from src.data import (
    EVERY,
    build_leave_one_out,
    build_loo_override,
    build_pooled,
    dlc_run_dir,
    mouse_dataset,
)

PROTOCOL = "loo"
FOLDS: int | None = None  # ``loo`` only; None fits all nine
VALID_MODE = "out_domain"  # ``loo`` only

#: ``train.py``'s constant, and set together with it -- a fold the two families
#: split differently is not a comparison.
OVERRIDE_FOLD: str | None = None

SUPER_ANIMAL = "superanimal_topviewmouse"
MODEL_NAME = "hrnet_w32"  # or "resnet_50"
DETECTOR_NAME = "fasterrcnn_resnet50_fpn_v2"  # or "fasterrcnn_mobilenet_v3_large_fpn"

EPOCHS = 60
SAVE_EPOCHS = 5
BATCH_SIZE = 8
DETECTOR_EPOCHS = 25
DETECTOR_SAVE_EPOCHS = 5
#: Well below the pose batch above, and not a typo. The pose model is top-down, so
#: it trains on per-animal crops; the detector trains on whole 1280x720 frames --
#: ``ResizeFromDataSizeCollate`` scales the short side to at most 1152 and does not
#: square them -- which is what runs a 12 GB card out of memory at 8.
DETECTOR_BATCH_SIZE = 2
EVAL_INTERVAL: int | None = 5
MAX_INDIVIDUALS = 4
DEVICE: str | None = None  # None -> auto-detected


def drop_unlabelled_instances(project_root: Path) -> None:
    """Rewrite the project's annotations without the instances DeepLabCut drops."""
    for split in ("train", "test"):
        path = project_root / "annotations" / f"{split}.json"
        coco = json.loads(path.read_text())
        kept = [
            ann
            for ann in coco["annotations"]
            if any(v > 0 for v in ann["keypoints"][2::3])
        ]
        dropped = len(coco["annotations"]) - len(kept)
        if dropped:
            path.write_text(json.dumps({**coco, "annotations": kept}))
        print(f"  {split:<5} kept {len(kept)} instances, dropped {dropped} unlabelled")


def instances_per_image(project_root: Path, split: str) -> int:
    """Most annotations on any one image of a split."""
    coco = json.loads((project_root / "annotations" / f"{split}.json").read_text())
    counts: dict[int, int] = {}
    for ann in coco["annotations"]:
        counts[ann["image_id"]] = counts.get(ann["image_id"], 0) + 1
    return max(counts.values(), default=0)


def untrainable(project_root: Path) -> bool:
    train = instances_per_image(project_root, "train")
    # ``mtmb`` maps the project's ``test.json`` to the valid split.
    valid = instances_per_image(project_root, "test")
    return valid > train


def train_dlc(project_root: Path, model_folder: Path | None = None) -> Path:
    """Adapt the SuperAnimal snapshots to one DeepLabCut COCO project."""
    model_folder = model_folder or project_root / "checkpoints"
    model_folder.mkdir(parents=True, exist_ok=True)
    individuals = MAX_INDIVIDUALS

    # Downloads from HuggingFace on first use.
    pose_snapshot = get_super_animal_snapshot_path(SUPER_ANIMAL, MODEL_NAME)
    detector_snapshot = get_super_animal_snapshot_path(SUPER_ANIMAL, DETECTOR_NAME)

    config = PoseConfig.build_for_superanimal_inference(
        super_animal=SUPER_ANIMAL,
        model_name=MODEL_NAME,
        detector_name=DETECTOR_NAME,
        max_individuals=individuals,
        device=DEVICE,
    )
    model_config_path = model_folder / "pytorch_config.yaml"
    config.to_yaml(model_config_path, overwrite=True)

    print(f"\nfine-tuning {SUPER_ANIMAL}_{MODEL_NAME} on {project_root}")
    print(f"  max individuals per image: {individuals}")
    print(f"  pose snapshot:     {pose_snapshot}")
    print(f"  detector snapshot: {detector_snapshot}")
    print(f"  model config:      {model_config_path}")
    print(f"  snapshots ->       {model_folder}")

    adaptation_train(
        project_root=project_root,
        model_folder=model_folder,
        train_file="train.json",
        test_file="test.json",
        model_config_path=model_config_path,
        device=DEVICE,
        epochs=EPOCHS,
        save_epochs=SAVE_EPOCHS,
        detector_epochs=DETECTOR_EPOCHS,
        detector_save_epochs=DETECTOR_SAVE_EPOCHS,
        snapshot_path=pose_snapshot,
        detector_path=detector_snapshot,
        batch_size=BATCH_SIZE,
        detector_batch_size=DETECTOR_BATCH_SIZE,
        eval_interval=EVAL_INTERVAL,
        skip_detector=DETECTOR_EPOCHS == 0,
    )
    return model_folder


def splits_to_fit() -> list[Path]:
    """The exports this configuration asks for, built and ready to train on."""
    if PROTOCOL != "loo":
        return [build_pooled(EVERY)]
    if OVERRIDE_FOLD:
        return [build_loo_override(OVERRIDE_FOLD, EVERY, VALID_MODE)]
    return build_leave_one_out(EVERY, folds=FOLDS, valid_mode=VALID_MODE)


if __name__ == "__main__":
    splits = splits_to_fit()
    dataset = mouse_dataset()
    for i, dataset_dir in enumerate(splits, start=1):
        print(f"\n=== {i}/{len(splits)}  {dataset_dir.name} ===")
        run_dir = dlc_run_dir(dataset_dir)
        if (run_dir / "checkpoints" / f"snapshot-{EPOCHS:03d}.pt").exists():
            print(f"  already trained: {run_dir}")
            continue

        project_root = dataset.to_deeplabcut(dataset_dir.name, out_dir=run_dir)
        drop_unlabelled_instances(project_root)

        # Checked after the pruning above, which is what decides the final counts.
        if untrainable(project_root):
            print(
                f"  skipping {dataset_dir.name} cause number of instances in valid > train",
            )
            continue
        train_dlc(project_root)
