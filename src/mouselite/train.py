"""Fine-tune an rfdetr model on a COCO dataset. Needs `pip install mouselite[train]`."""

from datetime import datetime
from pathlib import Path

from mouselite.models import BASE, MOUSELITE, build_model
from mouselite.runs import OUTPUT_DIR, save_run, write_run_info


def default_output_dir() -> Path:
    """`output/<timestamp>`, so each run gets its own folder."""
    return OUTPUT_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def train(
    dataset_dir: str | Path,
    kind: str,
    size: str = "medium",
    weights: str | Path = BASE,
    output_dir: str | Path | None = None,
    epochs: int = 10,
    batch_size: int = 4,
    from_format: str | None = None,
    symlink: bool = True,
    **kwargs,
) -> Path:
    """Train `kind`/`size` starting from `weights` (BASE, MOUSELITE or a checkpoint path).

    Extra kwargs are forwarded to `rfdetr`'s `train` (e.g. `lr`, `resolution`, `device`).
    Saves a `metrics.png` next to the checkpoints once training finishes. `output_dir`
    defaults to `default_output_dir()`.
    """
    from rfdetr.visualize import plot_metrics

    from mouselite.format import prepare_dataset

    output_dir = Path(output_dir) if output_dir is not None else default_output_dir()
    dataset = prepare_dataset(dataset_dir, output_dir, kind, from_format, symlink=symlink)
    if from_format is not None:
        print(f"converted {from_format} dataset to {dataset}")
    output_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "kind": kind,
        "size": None if kind == "keypoints" else size,  # keypoints has a single size
        "weights": weights if weights in (BASE, MOUSELITE) else Path(weights).resolve(),
        "dataset_dir": Path(dataset_dir).resolve(),
        "from_format": from_format,
        "epochs": epochs,
        "batch_size": batch_size,
        "train_kwargs": kwargs,
        "status": "running",
        "started": _now(),
        "finished": None,
    }
    write_run_info(output_dir, info)

    try:
        model = build_model(kind, size, weights=weights)
        model.train(
            dataset_dir=str(dataset),
            output_dir=str(output_dir),
            epochs=epochs,
            batch_size=batch_size,
            **kwargs,
        )
    except BaseException as exc:
        # Lightning turns Ctrl-C into sys.exit(1) after shutting down gracefully
        interrupted = isinstance(exc, KeyboardInterrupt | SystemExit)
        info["status"] = "interrupted" if interrupted else "failed"
        info["finished"] = _now()
        write_run_info(output_dir, info)
        raise

    info["status"], info["finished"] = "finished", _now()
    write_run_info(output_dir, info)
    save_run(output_dir)
    plot_metrics(
        str(output_dir / "metrics.csv"),
        output_path=str(output_dir / "metrics.png"),
    )
    return output_dir
