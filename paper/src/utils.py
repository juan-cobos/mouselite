import json
from pathlib import Path

from src.data import run_dir


def report_run(output_dir: Path) -> None:
    """Report rfdetr training run."""
    from rfdetr.visualize import plot_loss_metrics, plot_map_metrics

    print(f"\ndone: {output_dir}")
    weights = sorted(output_dir.glob("*.ckpt")) + sorted(output_dir.glob("*.pth"))
    for w in weights:
        print(f"  {w.name}  {w.stat().st_size / 1e6:.0f} MB")
    if not weights:
        print("  no weights written")

    if not (metrics_csv := output_dir / "metrics.csv").exists():
        print("  no metrics.csv to plot")
        return

    plots = {
        "_loss.png": plot_loss_metrics,
        "_map.png": plot_map_metrics,
    }
    for name, plot in plots.items():
        try:
            plot(str(metrics_csv), output_path=str(output_dir / name))
        except ValueError as exc:  # no columns of that family in the CSV
            print(f"  {name}: {exc}")
        else:
            print(f"  {name}")


def report_metrics(
    metrics: dict[str, float],
    output_dir: Path,
    split: str,
    suffix: str = "",
) -> Path:
    """Print COCO summary rows and write them beside the weights they scored."""
    print()
    for key, value in metrics.items():
        print(f"  {key:<20} {value:.4f}")

    path = output_dir / f"{split}_metrics{suffix}.json"
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\n  {path}")
    return path


def last_checkpoint(dataset_dir: Path, required: bool = True) -> Path | None:
    """The run's resumable checkpoint, rewritten every epoch by the trainer."""
    last = run_dir(dataset_dir) / "last.ckpt"
    if last.exists():
        print(f"  resuming from {last}")
        return last
    if required:
        raise SystemExit(f"RESUME: no checkpoint at {last}")
    return None
