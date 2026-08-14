"""Fine-tune every size of one RF-DETR kind on the pooled split."""

from pathlib import Path

from rfdetr import (
    RFDETRKeypointPreview,
    RFDETRLarge,
    RFDETRMedium,
    RFDETRNano,
    RFDETRSegLarge,
    RFDETRSegMedium,
    RFDETRSegNano,
    RFDETRSegSmall,
    RFDETRSmall,
)
from src.data import EVERY, RUNS_DIR, build_pooled
from src.utils import report_run

KIND = "segmentation"  # detection | segmentation

MODELS: dict[str, dict[str, type]] = {
    "detection": {
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "medium": RFDETRMedium,
        "large": RFDETRLarge,
    },
    "segmentation": {
        "nano": RFDETRSegNano,
        "small": RFDETRSegSmall,
        "medium": RFDETRSegMedium,
        "large": RFDETRSegLarge,
    },
    "keypoints": {"preview": RFDETRKeypointPreview},
}

EPOCHS = 100
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
NUM_WORKERS = 2

LR = 1e-4
LR_ENCODER = 1e-4
LR_SCHEDULER = "cosine"
LR_MIN_FACTOR = 0.0
WARMUP_EPOCHS = 1.0

USE_EMA = True
MULTI_SCALE = True
EXPANDED_SCALES = False

EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 0.001
EARLY_STOPPING_USE_EMA = False
SKIP_BEST_EPOCHS = 10

SEED = 0
CHECKPOINT_INTERVAL = 50


def variants(kind: str) -> dict[str, type]:
    """The sizes ``kind`` ships, or a listing of the kinds that exist."""
    if kind not in MODELS:
        raise SystemExit(f"unknown kind {kind!r}; expected one of {list(MODELS)}")
    return MODELS[kind]


def run_dir(kind: str, size: str, dataset_dir: Path) -> Path:
    """Where a ``kind``/``size`` run over ``dataset_dir`` writes its weights."""
    parts = [kind, size if len(variants(kind)) > 1 else None, dataset_dir.name]
    return RUNS_DIR / "_".join(part for part in parts if part)


def make_model(kind: str, size: str, **overrides):
    """Instantiate the RF-DETR class for ``kind``/``size``."""
    available = variants(kind)
    if size not in available:
        raise SystemExit(
            f"{kind} has no size {size!r}; expected one of {list(available)}",
        )

    cls = available[size]
    model = cls(**overrides)
    print(f"  model {cls.__name__} (resolution {model.model_config.resolution})")
    return model


def train(
    kind: str,
    size: str,
    dataset_dir: Path,
    output_dir: Path | None = None,
    epochs: int = EPOCHS,
    **train_overrides,
) -> Path:
    """Fine-tune one size on a built split."""
    output_dir = output_dir or run_dir(kind, size, dataset_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\ntraining on {dataset_dir} -> {output_dir}")

    model = make_model(kind, size)
    model.train(
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
        multi_scale=MULTI_SCALE,
        expanded_scales=EXPANDED_SCALES,
        run_test=False,  # the test split is scored by the eval scripts, not the fit
        early_stopping=EARLY_STOPPING,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_min_delta=EARLY_STOPPING_MIN_DELTA,
        early_stopping_use_ema=EARLY_STOPPING_USE_EMA,
        skip_best_epochs=SKIP_BEST_EPOCHS,
        seed=SEED,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        progress_bar="tqdm",
        tensorboard=False,
        **train_overrides,
    )
    return output_dir


def train_all(kind: str, dataset_dir: Path) -> list[Path]:
    """Fit every size of ``kind`` in turn, reporting as each one finishes."""
    sizes = list(variants(kind))
    output_dirs = []
    for i, size in enumerate(sizes, start=1):
        print(f"\n=== {i}/{len(sizes)}  {kind} {size} ===")
        output_dir = run_dir(kind, size, dataset_dir)
        if (output_dir / "checkpoint_best_total.pth").exists():
            print(f"  already trained: {output_dir}")
            continue
        train(kind, size, dataset_dir)
        report_run(output_dir)
        output_dirs.append(output_dir)
    return output_dirs


if __name__ == "__main__":
    train_all(KIND, build_pooled(EVERY))
