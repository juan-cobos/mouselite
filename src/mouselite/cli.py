from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True)

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _video_files(paths: list[Path]) -> list[Path]:
    """Expand directories to the video files inside them (not recursive), in order."""
    videos = []
    for path in paths:
        if path.is_dir():
            found = sorted(p for p in path.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)
            if not found:
                raise typer.BadParameter(f"No video files in {path}")
            videos.extend(found)
        else:
            videos.append(path)
    return videos


@app.command()
def run(
    video_paths: Annotated[
        list[Path],
        typer.Argument(help="Video files, or directories of videos, to process in turn."),
    ],
    kind: Annotated[str, typer.Option(help="One of the kinds from list-models.")],
    size: str = "medium",
    checkpoint: Path | None = None,
    tracker: str = "ocsort",
    threshold: float = 0.5,
    nms_threshold: float = 0.5,
    top_k: int | None = None,
    every: int = 1,
    output_dir: Path = Path("output"),
    show: bool = False,
    hud: bool = False,
    dtype: str = "float32",
    batch_size: int = 1,
    compile: bool = False,
    show_progress: bool = True,
) -> None:
    """Run inference on each video, writing an annotated video and a COCO export."""
    from mouselite.models import get_model
    from mouselite.pipeline import Pipeline
    from mouselite.tracker import get_tracker

    videos = _video_files(video_paths)

    model = get_model(
        kind,
        size=size,
        checkpoint=checkpoint,
        dtype=dtype,
        batch_size=batch_size,
        compile=compile,
    )
    tracker_instance = get_tracker(tracker)
    pipeline = Pipeline(
        model,
        tracker_instance,
        threshold=threshold,
        nms_threshold=nms_threshold,
        top_k=top_k,
        every=every,
    )
    for video_path in videos:
        output = pipeline.run(
            video_path,
            output_dir=output_dir,
            show=show,
            show_progress=show_progress,
            hud=hud,
        )
        typer.echo(f"wrote {output}")


@app.command()
def retrack(
    annotations_path: Path,
    tracker: Annotated[str, typer.Option(help="One of the trackers from list-trackers.")],
    output_dir: Path = Path("output"),
    lost_track_buffer: Annotated[
        int | None,
        typer.Option(help="Frames a track survives without a match."),
    ] = None,
    minimum_iou_threshold: Annotated[
        float | None,
        typer.Option(help="Minimum IoU to match a detection to a track."),
    ] = None,
    show_progress: bool = True,
) -> None:
    """Re-run tracking on a previously exported COCO dataset, without running inference."""
    from mouselite.tracker import retrack as retrack_video

    tracker_kwargs = {
        "lost_track_buffer": lost_track_buffer,
        "minimum_iou_threshold": minimum_iou_threshold,
    }
    output = retrack_video(
        annotations_path,
        tracker,
        output_dir=output_dir,
        show_progress=show_progress,
        **{k: v for k, v in tracker_kwargs.items() if v is not None},
    )
    typer.echo(f"wrote {output}")


@app.command()
def analyze(
    annotations_path: Path,
    fps: Annotated[
        float | None,
        typer.Option(help="Source frame rate, if the export did not record it."),
    ] = None,
    scale: Annotated[
        float,
        typer.Option(help="Units per pixel (e.g. cm/px) for distances and speeds."),
    ] = 1.0,
    immobile_below: Annotated[
        float | None,
        typer.Option(help="Speed under which a frame counts as immobile, in output units."),
    ] = None,
    min_frames: Annotated[
        int,
        typer.Option(help="Drop tracks seen on fewer frames than this."),
    ] = 1,
    max_gap: Annotated[
        int | None,
        typer.Option(help="Fill missing detections over gaps of up to this many frames."),
    ] = None,
    smooth_window: Annotated[
        int | None,
        typer.Option(help="Rolling-median window, in frames, applied to the trajectories."),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(help="Defaults to the folder holding the annotations."),
    ] = None,
) -> None:
    """Summarise a COCO export into per-track statistics and trajectory tables.

    Writes `summary.csv` and `trajectories.csv` next to the annotations.
    """
    from mouselite import analysis

    tracks = analysis.Tracks.from_coco(annotations_path, fps=fps, min_frames=min_frames)
    if max_gap is not None:
        tracks.xyxy = analysis.interpolate(tracks.xyxy, max_gap=max_gap)
        if tracks.keypoints is not None:
            tracks.keypoints = analysis.interpolate(tracks.keypoints, max_gap=max_gap)
    if smooth_window is not None:
        tracks.xyxy = analysis.smooth(tracks.xyxy, window=smooth_window)
        if tracks.keypoints is not None:
            tracks.keypoints = analysis.smooth(tracks.keypoints, window=smooth_window)

    output_dir = output_dir or annotations_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    table = tracks.summary(scale=scale, immobile_below=immobile_below)
    table.to_csv(output_dir / "summary.csv", index=False)
    tracks.to_dataframe().to_csv(output_dir / "trajectories.csv", index=False)

    units = "px" if scale == 1.0 else "units"
    per = "s" if tracks.fps else "frame"
    typer.echo(f"{tracks.n_tracks} tracks over {tracks.n_frames} frames ({units}/{per})")
    typer.echo(table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    typer.echo(f"wrote {output_dir / 'summary.csv'}")


@app.command()
def train(
    dataset_dir: Path,
    kind: Annotated[str, typer.Option(help="One of the kinds from list-models.")],
    size: str = "medium",
    weights: Annotated[
        str,
        typer.Option(help="`base` (rfdetr pretrained), `mouselite`, or a checkpoint path."),
    ] = "base",
    output_dir: Path = Path("output/train"),
    epochs: int = 10,
    batch_size: int = 4,
    lr: float | None = None,
    resolution: int | None = None,
    device: str | None = None,
    from_format: Annotated[
        str | None,
        typer.Option(
            "--from",
            help="Convert `dataset_dir` from this format first (see list-formats).",
        ),
    ] = None,
    symlink: Annotated[
        bool,
        typer.Option(help="With --from: symlink the images instead of copying them."),
    ] = True,
) -> None:
    r"""Fine-tune on a COCO dataset. Needs the extra: `pip install mouselite\[train]`."""
    try:
        from mouselite.format import convert
        from mouselite.train import train as train_model
    except ImportError as exc:  # the train extra (which includes convert) is optional
        typer.echo(
            f"{exc}\nTraining needs extras: install with `pip install mouselite[train]`.",
            err=True,
        )
        raise typer.Exit(1) from exc

    if from_format is not None:
        if kind == "segmentation":
            typer.echo(
                f"--from {from_format} is not supported for --kind segmentation: "
                "pose formats carry keypoints, not masks.",
                err=True,
            )
            raise typer.Exit(1)
        dataset_dir = convert(
            dataset_dir, output_dir / "dataset", from_format, symlink=symlink
        )
        typer.echo(f"converted {from_format} dataset to {dataset_dir}")

    extra = {"lr": lr, "resolution": resolution, "device": device}
    train_model(
        dataset_dir,
        kind,
        size=size,
        weights=weights,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        **{k: v for k, v in extra.items() if v is not None},
    )
    typer.echo(f"wrote checkpoints and metrics.png to {output_dir}")


@app.command("app")
def app_command(
    share: bool = True,
    host: str | None = None,
    port: int | None = None,
) -> None:
    r"""Launch the Gradio demo. Needs the extra: `pip install mouselite\[app]`."""
    try:
        from mouselite.app import main as launch_app
    except ImportError as exc:  # gradio is an optional dependency
        typer.echo(
            f"{exc}\nThe demo needs gradio: install with `pip install mouselite[app]`.",
            err=True,
        )
        raise typer.Exit(1) from exc
    launch_app(share=share, host=host, port=port)


@app.command("list-models")
def list_models() -> None:
    """Print the available --kind values, and --size options for sized kinds."""
    from mouselite.models import MODELS

    for kind, sizes in MODELS.items():
        typer.echo(f"{kind}: {', '.join(sizes)}" if isinstance(sizes, dict) else kind)


@app.command("list-trackers")
def list_trackers() -> None:
    """Print the available --tracker values."""
    from mouselite.tracker import TRACKERS

    for name in TRACKERS:
        typer.echo(name)


@app.command("list-formats")
def list_formats() -> None:
    r"""Print the dataset formats `train --from` accepts. Needs `mouselite\[convert]`."""
    try:
        from mouselite.format import FORMATS
    except ImportError as exc:  # pandas/tables are optional
        typer.echo(
            f"{exc}\nInstall the extra with `pip install mouselite[convert]`.",
            err=True,
        )
        raise typer.Exit(1) from exc

    for name in FORMATS:
        typer.echo(name)


def main() -> None:
    app()
