from pathlib import Path

from mtmb.dataset import MouseDataset

ROOT = Path.cwd()
RUNS_DIR = ROOT / "runs"
RUN_PREFIX = "keypoint_preview"
DLC_SUFFIX = "_dlc"

EVERY = 10  # keep every Nth frame; 1 uses all ~14k
OKS_SIGMA = 0.1

POOLED_NAME = "pooled"
POOLED_RATIOS = (0.7, 0.15, 0.15)

LOO_NAME = "loo"
#: Slice of the training tasks' frames held back to validate on under
#: ``in_domain``; unused under ``out_domain``, where valid is a whole task.
LOO_VALID_FRACTION = 0.1
#: ``in_domain`` cuts valid out of the training tasks, so it measures fit and
#: keeps all eight remaining assays in train. ``out_domain`` gives valid a task of
#: its own -- closer to what test asks, at the cost of a training task and of
#: selecting checkpoints on the very thing being measured.
LOO_VALID_MODE = "in_domain"
#: Draws the split, not the fit -- a run may be reseeded without rebuilding the
#: export, and rebuilding it under a new seed would silently move the boundary
#: between train and test.
SPLIT_SEED = 0

#: Held-out (test) task -> the valid assay replacing the rotation's; test is untouched.
LOO_VALID_OVERRIDE = {"direct_interaction": "nort"}

MTMB_DATASET_DIR = ROOT / "mtmb_dataset"
DATASETS_DIR = ROOT / "datasets"


def mouse_dataset() -> MouseDataset:
    """The downloaded tasks, exporting into this project's ``datasets/``."""
    return MouseDataset(path=MTMB_DATASET_DIR, build_root=DATASETS_DIR)


def downloaded_dataset() -> MouseDataset:
    """``mouse_dataset``, fetched from the Hub first; files already current are skipped."""
    dataset = mouse_dataset()
    dataset.download()
    return dataset


def pooled_split_name(every: int) -> str:
    """``every`` goes in the name: ``build`` refuses to overwrite a different split."""
    return POOLED_NAME if every == 1 else f"{POOLED_NAME}_every{every}"


def build_pooled(every: int, rebuild: bool = False) -> Path:
    """Pool all nine tasks, split at the image level, materialise the export.

    Neighbouring frames are near-duplicates at ~30 fps, so this split leaks across
    train/test -- a ceiling, not a generalisation estimate. A matching export is
    reused; ``rebuild`` relinks it.
    """
    dataset = downloaded_dataset()
    manifest = dataset.split_random(
        ratios=POOLED_RATIOS,
        name=pooled_split_name(every),
        seed=SPLIT_SEED,
        every=every,
    )
    print(f"\n{manifest.describe()}")
    return dataset.build(manifest, overwrite=rebuild)


def loo_split_name(every: int, valid_mode: str = LOO_VALID_MODE) -> str:
    """Base name for the fold exports; ``mtmb`` appends ``_fold_<n>_<task>``.

    ``valid_mode`` is part of the name because it changes what train and valid
    hold for the same held-out task. Only ``out_domain`` is suffixed, so the
    ``in_domain`` exports and runs already on disk keep their names.
    """
    name = LOO_NAME if every == 1 else f"{LOO_NAME}_every{every}"
    return name if valid_mode == "in_domain" else f"{name}_{valid_mode}"


def build_leave_one_out(
    every: int,
    folds: int | None = None,
    valid_mode: str = LOO_VALID_MODE,
    rebuild: bool = False,
) -> list[Path]:
    """One export per fold: that task is test, the other eight are train.

    The transfer counterpart to ``build_pooled``. Test is a whole assay the fit
    never saw, so nothing leaks across the boundary and the scores answer "does
    this hold up on an arena it was not trained on" rather than "did it memorise
    these frames". Expect them well below the pooled split's.

    ``folds`` builds only that many, the held-out tasks drawn under ``SPLIT_SEED``
    -- nine folds is nine fits. Each is still trained on every task but its own.
    """
    dataset = downloaded_dataset()
    manifests = dataset.split_leave_one_out(
        name=loo_split_name(every, valid_mode),
        valid_fraction=LOO_VALID_FRACTION,
        seed=SPLIT_SEED,
        every=every,
        folds=folds,
        valid_mode=valid_mode,
    )

    built = []
    for manifest in manifests:
        print(f"\n{manifest.describe()}")
        built.append(dataset.build(manifest, overwrite=rebuild))
    return built


def build_loo_override(
    held_out: str,
    every: int,
    valid_mode: str = "out_domain",
    rebuild: bool = False,
) -> Path:
    """Rebuild one leave-one-out fold with its valid assay taken from the override.

    Fold number, held-out task and seed are the sweep's, so the fold lines up with
    the one it replaces; only which assay validates it moves. That assay goes in
    the export's name, so the two sit side by side and both can be reported.
    """
    from mtmb.manifest import Manifest, Split

    if valid_mode != "out_domain":
        # ``in_domain`` validates on a slice of the training tasks: no valid assay
        # to move, nothing for the override to say.
        raise SystemExit(
            f"the valid override is an out_domain rule, not {valid_mode!r}",
        )

    dataset = downloaded_dataset()
    tasks = [dataset.task(t).value for t in dataset.tasks]
    held_out = dataset.task(held_out).value
    if held_out not in LOO_VALID_OVERRIDE:
        raise SystemExit(
            f"no valid override for {held_out!r}; have {sorted(LOO_VALID_OVERRIDE)}",
        )

    valid_task = dataset.task(LOO_VALID_OVERRIDE[held_out]).value
    train_tasks = [t for t in tasks if t not in (held_out, valid_task)]
    image_ids = {t: dataset.image_ids(t, every) for t in tasks}
    fold = tasks.index(held_out)  # the sweep enumerates the tasks in this order

    splits = {
        "train": Split(train_tasks, [i for t in train_tasks for i in image_ids[t]]),
        "valid": Split([valid_task], image_ids[valid_task]),
        "test": Split([held_out], image_ids[held_out]),
    }
    manifest = Manifest.make(
        dataset,
        f"{loo_split_name(every, valid_mode)}_fold_{fold:02d}_{held_out}"
        f"_valid_{valid_task}",
        f"{LOO_NAME}/{valid_mode}_valid",  # still the sweep's protocol
        splits,
        SPLIT_SEED + fold,
        every,
    )
    print(f"\n{manifest.describe()}")
    return dataset.build(manifest, overwrite=rebuild)


def run_dir(dataset_dir: Path) -> Path:
    """Where a run over ``dataset_dir`` writes its weights and metrics."""
    return RUNS_DIR / f"{RUN_PREFIX}_{dataset_dir.name}"


def dlc_run_dir(dataset_dir: Path) -> Path:
    """The DeepLabCut counterpart to ``run_dir``."""
    return RUNS_DIR / f"{dataset_dir.name}{DLC_SUFFIX}"


def dataset_for_run(output_dir: Path) -> Path:
    """The split a run was fitted on -- the inverse of ``run_dir``.

    Scoring starts from a run directory, so the export has to be recovered from
    its name. Reading it back beats rebuilding a default one: a fold scored
    against the pooled split's annotations would report on frames its fit was
    trained on, and nothing about the numbers would look wrong.
    """
    name = output_dir.name.removeprefix(f"{RUN_PREFIX}_")
    dataset_dir = DATASETS_DIR / name
    if not dataset_dir.is_dir():
        raise SystemExit(f"no split at {dataset_dir} for run {output_dir.name}")
    return dataset_dir


def oks_sigmas(schema) -> list[float]:
    """``OKS_SIGMA`` per keypoint, sized from the schema rather than hardcoded."""
    return [OKS_SIGMA] * sum(schema.num_keypoints_per_class)
