"""Save training runs' info and checkpoints, and find them again."""

import json
import shutil
from pathlib import Path

RUN_INFO = "checkpoints.json"
BEST_CHECKPOINTS = (  # in order of preference
    "checkpoint_best_total.pth",
    "checkpoint_best_ema.pth",
    "checkpoint_best_regular.pth",
)
CACHE_DIR = Path.home() / ".cache" / "mouselite"
OUTPUT_DIR = Path("output")


def write_run_info(run_dir: str | Path, info: dict) -> None:
    """Save a run's info in its folder."""
    with open(Path(run_dir) / RUN_INFO, "w") as f:
        json.dump(info, f, indent=2, default=str)  # default=str: Paths in kwargs


def read_run_info(run_dir: str | Path) -> dict:
    """Return a run's info, or an empty dict if it has none."""
    path = Path(run_dir) / RUN_INFO
    return json.loads(path.read_text()) if path.exists() else {}


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
    """Return the info, folder and best checkpoint of every run found."""
    roots = roots if roots is not None else (OUTPUT_DIR, CACHE_DIR)
    run_dirs = dict.fromkeys(  # in order of the roots, sorted within each
        run_dir
        for root in map(Path, roots)
        if root.is_dir()
        for run_dir in sorted(
            {path.parent for name in BEST_CHECKPOINTS for path in root.rglob(name)},
        )
    )
    runs = []
    for run_dir in run_dirs:
        checkpoint = next(
            run_dir / name for name in BEST_CHECKPOINTS if (run_dir / name).exists()
        )
        info = read_run_info(run_dir)
        runs.append({**info, "dir": run_dir, "checkpoint": checkpoint})
    return runs
