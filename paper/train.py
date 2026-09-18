"""Fine-tune the RF-DETR keypoint preview model on the pooled mtmb split."""

from pathlib import Path

from rfdetr.config import KeypointTrainConfig
from rfdetr.datasets import infer_coco_keypoint_schema
from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer

from src.builder import RESOLUTION, make_rfdetr_keypoints
from src.data import (
    EVERY,
    build_leave_one_out,
    build_loo_override,
    build_pooled,
    oks_sigmas,
    run_dir,
)
from src.utils import last_checkpoint, report_run

#: ``pooled`` is one fit on the image-level split, which leaks across train/test
#: and reads as a ceiling. ``loo`` is one fit per held-out assay -- a transfer
#: estimate, and nine times the wall clock.
PROTOCOL = "loo"
FOLDS: int | None = None  # ``loo`` only; None fits all nine

#: ``loo`` only. ``in_domain`` slices valid out of the training tasks and keeps
#: all eight in train; ``out_domain`` gives valid a task of its own, which costs a
#: training task but stops checkpoint selection from being blind to transfer.
#: Exports and run directories carry the mode, so the two sweeps sit side by side
#: rather than one landing on top of the other.
VALID_MODE = "out_domain"

#: ``loo``/``out_domain`` only. A held-out task to refit alone under
#: ``LOO_VALID_OVERRIDE``'s valid assay instead of the rotation's; the export is
#: named for that assay, so this lands beside the sweep's fold rather than on it.
OVERRIDE_FOLD: str | None = None

EPOCHS = 60
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
NUM_WORKERS = 2

LR = 1e-4
LR_ENCODER = 1e-4  # cookbook value; RF-DETR's own default is 1.5e-4

#: RF-DETR's default is ``step`` with ``lr_drop=100``, which never fires below 100
#: epochs -- the LR would stay flat for the whole run. Cosine anneals to
#: ``LR * LR_MIN_FACTOR`` over ``EPOCHS``, so an early stop leaves the schedule
#: unfinished at a higher LR than a completed run would end on.
LR_SCHEDULER = "cosine"
LR_MIN_FACTOR = 0.0
WARMUP_EPOCHS = 1.0

USE_EMA = True
MULTI_SCALE = True
EXPANDED_SCALES = False
COMPUTE_TRAIN_METRICS = True

#: Stops on ``val/keypoint_map_50_95`` -- ``max(regular, ema)``, or the EMA metric
#: alone under ``EARLY_STOPPING_USE_EMA``. ``SKIP_BEST_EPOCHS`` is the grace period
#: both this and best-checkpoint selection ignore: the 27-point head is built from
#: scratch over COCO's 17-point one, so keypoint mAP sits at 0 for the first few
#: epochs and would otherwise burn the patience budget before anything is learnt.
EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 0.001
EARLY_STOPPING_USE_EMA = False
SKIP_BEST_EPOCHS = 10

SEED = 0
CHECKPOINT_INTERVAL = 50

#: Continue the run in ``RUNS_DIR`` from its ``last.ckpt`` rather than starting
#: over. Deliberately explicit: resuming a finished fit is a no-op that looks like
#: a run, so this is not inferred from the checkpoint merely being there.
RESUME = False


def train(
    dataset_dir: Path,
    output_dir: Path | None = None,
    resolution: int = RESOLUTION,
    epochs: int = EPOCHS,
    resume: Path | None = None,
    **overrides,
) -> Path:
    """Fine-tune one model on a built split."""
    output_dir = output_dir or run_dir(dataset_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\ntraining on {dataset_dir} -> {output_dir}")
    schema = infer_coco_keypoint_schema(
        str(dataset_dir / "train" / "_annotations.coco.json"),
    )

    config = KeypointTrainConfig(
        dataset_file="roboflow",  # the `<split>/_annotations.coco.json` layout
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        num_workers=NUM_WORKERS,
        lr=LR,
        lr_encoder=LR_ENCODER,
        lr_scheduler=LR_SCHEDULER,
        lr_scheduler_kwargs={"min_factor": LR_MIN_FACTOR},
        warmup_epochs=WARMUP_EPOCHS,
        use_ema=USE_EMA,
        run_test=False,  # the test split is scored by eval.py, not by the fit
        compute_train_metrics=COMPUTE_TRAIN_METRICS,
        compute_val_loss=True,
        multi_scale=MULTI_SCALE,
        expanded_scales=EXPANDED_SCALES,
        class_names=schema.class_names,
        keypoint_flip_pairs=schema.keypoint_flip_pairs,
        keypoint_oks_sigmas=oks_sigmas(schema),
        early_stopping=EARLY_STOPPING,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_min_delta=EARLY_STOPPING_MIN_DELTA,
        early_stopping_use_ema=EARLY_STOPPING_USE_EMA,
        skip_best_epochs=SKIP_BEST_EPOCHS,
        progress_bar="tqdm",
        seed=SEED,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        resume=str(resume) if resume else None,
        tensorboard=False,
        **overrides,
    )
    model = make_rfdetr_keypoints(schema, resolution)

    module = RFDETRModelModule(model.model_config, config)
    datamodule = RFDETRDataModule(model.model_config, config)
    trainer = build_trainer(config, model.model_config)
    trainer.fit(module, datamodule=datamodule, ckpt_path=config.resume or None)
    return output_dir


def fit_all(dataset_dirs: list[Path]) -> list[Path]:
    """Fit each split in turn, reporting as each one finishes."""
    single = len(dataset_dirs) == 1
    output_dirs = []
    for i, dataset_dir in enumerate(dataset_dirs, start=1):
        print(f"\n=== {i}/{len(dataset_dirs)}  {dataset_dir.name} ===")
        resume = last_checkpoint(dataset_dir, required=single) if RESUME else None
        output_dir = train(dataset_dir, resume=resume)
        report_run(output_dir)
        output_dirs.append(output_dir)
    return output_dirs


def splits_to_fit() -> list[Path]:
    """Export the configuration built and ready to train on."""
    if PROTOCOL != "loo":
        return [build_pooled(EVERY)]
    if OVERRIDE_FOLD:
        return [build_loo_override(OVERRIDE_FOLD, EVERY, VALID_MODE)]
    return build_leave_one_out(EVERY, folds=FOLDS, valid_mode=VALID_MODE)


if __name__ == "__main__":
    fit_all(splits_to_fit())
