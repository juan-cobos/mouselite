from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True)


@app.command()
def run(
    video_path: Path,
    kind: Annotated[str, typer.Option(help="One of the kinds from list-models.")],
    size: str = "medium",
    checkpoint: Path | None = None,
    tracker: str = "bytetrack",
    threshold: float = 0.5,
    nms_threshold: float = 0.5,
    top_k: int | None = None,
    every: int = 1,
    output_dir: Path = Path("output"),
    save_path: Path | None = None,
    show: bool = False,
    hud: bool = False,
    dtype: str = "float32",
    batch_size: int = 1,
    compile: bool = False,
    show_progress: bool = True,
) -> None:
    """Run inference on `video_path`, writing an annotated video and a COCO export."""
    import supervision as sv

    from mouselite.models import get_model
    from mouselite.pipeline import Pipeline
    from mouselite.tracker import get_tracker

    model = get_model(
        kind,
        size=size,
        checkpoint=checkpoint,
        dtype=dtype,
        batch_size=batch_size,
        compile=compile,
    )
    fps = sv.VideoInfo.from_video_path(str(video_path)).fps
    tracker_instance = get_tracker(tracker, frame_rate=fps)
    pipeline = Pipeline(
        model,
        tracker_instance,
        threshold=threshold,
        nms_threshold=nms_threshold,
        top_k=top_k,
        every=every,
    )
    output = pipeline.run(
        video_path,
        output_dir=output_dir,
        show=show,
        show_progress=show_progress,
        hud=hud,
        save_path=save_path,
    )
    typer.echo(f"wrote {output}")


@app.command()
def retrack(
    annotations_path: Path,
    tracker: Annotated[str, typer.Option(help="One of the trackers from list-trackers.")],
    output_dir: Path = Path("output"),
    fps: Annotated[
        float | None,
        typer.Option(help="Defaults to the frame rate recorded in the export, else 30."),
    ] = None,
    lost_track_buffer: Annotated[
        int | None,
        typer.Option(help="Frames a track survives without a match (at 30 fps)."),
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
        fps=fps,
        show_progress=show_progress,
        **{k: v for k, v in tracker_kwargs.items() if v is not None},
    )
    typer.echo(f"wrote {output}")


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


def main() -> None:
    app()
