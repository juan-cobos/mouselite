"""Save training runs' info and checkpoints, and find them again."""

import contextlib
import json
import shutil
from pathlib import Path

from mouselite.models import MODELS

RUN_INFO = "checkpoints.json"
BEST_CHECKPOINTS = (
    "checkpoint_best_total.pth",
    "checkpoint_best_ema.pth",
    "checkpoint_best_regular.pth",
)
CACHE_DIR = Path.home() / ".cache" / "mouselite"
OUTPUT_DIR = Path("output")


def write_run_info(run_dir: str | Path, info: dict) -> None:
    with open(Path(run_dir) / RUN_INFO, "w") as f:
        json.dump(info, f, indent=2, default=str)  # default=str: Paths in kwargs


def read_run_info(run_dir: str | Path) -> dict:
    """The run's info, or an empty dict for runs trained before it was saved."""
    path = Path(run_dir) / RUN_INFO
    return json.loads(path.read_text()) if path.exists() else {}


def _model_of(checkpoint: Path) -> dict:
    """The kind and size of the rfdetr model a checkpoint was trained with, if known."""
    import torch

    try:  # mmap and weights_only: reads the metadata without loading or unpickling weights
        name = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        name = name.get("model_name")
    except Exception:
        return {}
    for kind, classes in MODELS.items():
        sizes = classes if isinstance(classes, dict) else {None: classes}
        for size, cls in sizes.items():
            if cls == name:
                return {"kind": kind, "size": size}
    return {}


def _backfill_run_info(run_dir: Path, checkpoint: Path) -> dict:
    """Write the info of a run trained before it was saved, from its checkpoint."""
    info = _model_of(checkpoint)
    if info:  # in a read-only folder, still list the run, just don't save it
        with contextlib.suppress(OSError):
            write_run_info(run_dir, info)
    return info


def save_run(run_dir: str | Path, cache_dir: str | Path | None = None) -> Path:
    """Copy the run's info and best checkpoints to the cache, returning the copy."""
    run_dir = Path(run_dir)
    target = Path(cache_dir or CACHE_DIR) / run_dir.name
    target.mkdir(parents=True, exist_ok=True)
    for name in (RUN_INFO, *BEST_CHECKPOINTS):
        if (run_dir / name).exists():
            shutil.copy2(run_dir / name, target / name)
    return target


def find_runs(roots: tuple[str | Path, ...] | None = None) -> list[dict]:
    """The info, folder and best checkpoints of every run found under the given roots."""
    roots = roots if roots is not None else (OUTPUT_DIR, CACHE_DIR)
    run_dirs = dict.fromkeys(  # in order of the roots, sorted within each
        run_dir
        for root in map(Path, roots)
        if root.is_dir()
        for run_dir in sorted(
            {path.parent for name in BEST_CHECKPOINTS for path in root.rglob(name)}
        )
    )
    runs = []
    for run_dir in run_dirs:
        checkpoints = [
            run_dir / name for name in BEST_CHECKPOINTS if (run_dir / name).exists()
        ]
        info = read_run_info(run_dir) or _backfill_run_info(run_dir, checkpoints[0])
        runs.append({**info, "dir": run_dir, "checkpoints": checkpoints})
    return runs
