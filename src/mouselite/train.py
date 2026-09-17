"""Fine-tune an rfdetr model on a COCO dataset. Needs `pip install mouselite[train]`."""

from pathlib import Path

from mouselite.models import BASE, build_model


def train(
    dataset_dir: str | Path,
    kind: str,
    size: str = "medium",
    weights: str | Path = BASE,
    output_dir: str | Path = Path("output/train"),
    epochs: int = 10,
    batch_size: int = 4,
    **kwargs,
) -> None:
    """Train `kind`/`size` starting from `weights` (BASE, MOUSELITE or a checkpoint path).

    Extra kwargs are forwarded to `rfdetr`'s `train` (e.g. `lr`, `resolution`, `device`).
    Saves a `metrics.png` next to the checkpoints once training finishes.
    """
    from rfdetr.visualize import plot_metrics

    model = build_model(kind, size, weights=weights)
    model.train(
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
        epochs=epochs,
        batch_size=batch_size,
        **kwargs,
    )
    output_dir = Path(output_dir)
    plot_metrics(
        str(output_dir / "metrics.csv"),
        output_path=str(output_dir / "metrics.png"),
    )
