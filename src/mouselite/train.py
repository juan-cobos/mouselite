"""Fine-tune an rfdetr model on a COCO dataset. Needs `pip install mouselite[train]`."""

from datetime import datetime
from pathlib import Path

from mouselite.models import BASE, build_model


def default_output_dir() -> Path:
    """`output/<timestamp>`, so each run gets its own folder."""
    return Path("output") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def train(
    dataset_dir: str | Path,
    kind: str,
    size: str = "medium",
    weights: str | Path = BASE,
    output_dir: str | Path | None = None,
    epochs: int = 10,
    batch_size: int = 4,
    **kwargs,
) -> None:
    """Train `kind`/`size` starting from `weights` (BASE, MOUSELITE or a checkpoint path).

    Extra kwargs are forwarded to `rfdetr`'s `train` (e.g. `lr`, `resolution`, `device`).
    Saves a `metrics.png` next to the checkpoints once training finishes. `output_dir`
    defaults to `default_output_dir()`.
    """
    from rfdetr.visualize import plot_metrics

    output_dir = Path(output_dir) if output_dir is not None else default_output_dir()
    model = build_model(kind, size, weights=weights)
    model.train(
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
        epochs=epochs,
        batch_size=batch_size,
        **kwargs,
    )
    plot_metrics(
        str(output_dir / "metrics.csv"),
        output_path=str(output_dir / "metrics.png"),
    )
