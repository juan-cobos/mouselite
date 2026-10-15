"""Gradio demo wrapping the mouselite Pipeline, and fine-tuning in its own process.

Install the extra and run with:
    uv run --extra app mouselite app
"""

import atexit
import multiprocessing
import os
import re
import signal
import tempfile
from functools import lru_cache
from pathlib import Path

import gradio as gr
import pandas as pd
import supervision as sv
from huggingface_hub import snapshot_download

from mouselite.format import FORMATS, prepare_dataset
from mouselite.models import BASE, MODELS, MOUSELITE, get_model
from mouselite.pipeline import Pipeline
from mouselite.runs import CACHE_DIR, OUTPUT_DIR, find_runs, read_run_info
from mouselite.tracker import TRACKERS, get_tracker, retrack
from mouselite.train import default_output_dir, train_logged

ASSETS = Path(__file__).parent / "assets"
LOGO, ICON = ASSETS / "logo.svg", ASSETS / "icon.svg"
LOGO_HEIGHT = 60
REPO = "https://github.com/juan-cobos/mouselite"
THEME = gr.themes.Cyberpunk(
    primary_hue="violet",
    secondary_hue="teal",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),  # the font the docs use
)
KINDS = list(MODELS)
SIZES = list(MODELS["detection"])
EXPORTS = ("annotations.json", "trajectories.csv", "summary.csv")  # each <stem>_-prefixed
EXAMPLES_REPO = "juancobos/MultiTaskMouseBehaviour"  # one 10 s clip per behavioural task
COCO = "coco"  # the dataset format rfdetr trains on, needing no conversion
INFERENCE_WEIGHTS = (("MouseLite (released)", MOUSELITE),)
TRAIN_WEIGHTS = (("RF-DETR pretrained", BASE), ("MouseLite (released)", MOUSELITE))
LOG = "train.log"  # the training process's output, next to its checkpoints


def _header_html(height: int = LOGO_HEIGHT) -> str:
    """Logo and tagline on one line, drawn in the app's own colours."""
    logo = re.sub(r"<style>.*?</style>", "", LOGO.read_text(), flags=re.S)
    logo = logo.replace(
        "<svg ", f'<svg style="height:{height}px;width:auto;fill:currentColor" ', 1
    )
    return (
        '<div style="display:flex;align-items:center;gap:14px;padding:4px 0 10px;'
        'border-bottom:1px solid var(--border-color-primary)">'
        f'{logo}<span style="color:var(--body-text-color-subdued)">'
        "Run detection, segmentation or pose inference on mice videos.</span></div>"
    )


@lru_cache(maxsize=2)
def _load_model(kind: str, size: str, checkpoint: str):
    """Cache weights so repeated runs don't re-download and re-init the model."""
    return get_model(kind, size=size, checkpoint=checkpoint)


def _example_videos() -> list[Path]:
    """The dataset's example clips, cached after the first download; none when offline."""
    try:
        root = snapshot_download(
            EXAMPLES_REPO, repo_type="dataset", allow_patterns="assets/examples/*.mp4"
        )
    except Exception as exc:
        print(f"Skipping the example videos, could not download them: {exc}")
        return []
    return sorted(Path(root, "assets", "examples").glob("*.mp4"))


def _tmp_dir(prefix: str = "mouselite_") -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix))


def _exports(results_dir: str | Path, video_path: str | Path):
    """Value for the downloads component: the COCO export and CSV tables for a run."""
    stem = Path(video_path).stem
    files = [str(Path(results_dir) / f"{stem}_{name}") for name in EXPORTS]
    return gr.update(value=files, visible=True)


def _checkpoint_choices(*fixed: tuple[str, str]) -> list[tuple[str, str]]:
    """Dropdown choices: the fixed ones, then the checkpoint of every run found."""
    choices = list(fixed)
    for run in find_runs():
        model = " ".join(filter(None, (run.get("kind"), run.get("size")))) or "unknown"
        cached = " · cached" if run["dir"].is_relative_to(CACHE_DIR) else ""
        label = f"{run['dir'].name} · {model}{cached}"
        choices.append((label, str(run["checkpoint"])))
    return choices


def _checkpoint_info(choice: str | None) -> dict:
    """The run info of a chosen checkpoint; empty for released weights or unknown runs."""
    if not choice or choice in (BASE, MOUSELITE):
        return {}
    return read_run_info(Path(choice).parent)


def _toggle_size(kind, choice):
    """Keypoints ships a single checkpoint, and a trained checkpoint fixes its size."""
    return gr.update(
        interactive=kind != "keypoints" and "kind" not in _checkpoint_info(choice)
    )


def _lock_model(choice, kind):
    """Set kind and size from a trained checkpoint, which fixes its architecture."""
    info = _checkpoint_info(choice)
    kind = info.get("kind", kind)
    size = {"value": info["size"]} if info.get("size") else {}
    return (
        gr.update(value=kind, interactive="kind" not in info),
        gr.update(**size, interactive=kind != "keypoints" and "kind" not in info),
    )


def run_inference(
    video_path,
    checkpoint,
    kind,
    model_size,
    tracker_type,
    threshold,
    nms_threshold,
    top_k,
    every,
    progress=gr.Progress(track_tqdm=True),  # noqa: B008 — Gradio injects via this default
):
    """Run inference on the input video, returning the annotated video and its exports."""
    if video_path is None:
        gr.Warning("Please upload a video or record from webcam first.")
        return None, gr.update(visible=False), None

    progress(0, desc=f"Loading {kind} model...")
    model = _load_model(kind, model_size, checkpoint or MOUSELITE)

    fps = sv.VideoInfo.from_video_path(str(video_path)).fps
    tracker = get_tracker(tracker_type, frame_rate=fps)
    pipeline = Pipeline(
        model,
        tracker,
        threshold=threshold,
        nms_threshold=nms_threshold,
        top_k=int(top_k),
        every=int(every),
    )

    out_dir = _tmp_dir()
    progress(0, desc="Running inference...")
    video_out_path = pipeline.run(video_path, output_dir=out_dir)

    # Pipeline writes the COCO export and CSV tables next to the annotated video.
    results_dir = video_out_path.parent
    state = {"results_dir": str(results_dir), "video_path": str(video_path)}
    return str(video_out_path), _exports(results_dir, video_path), state


def run_retrack(
    state,
    tracker_type,
    progress=gr.Progress(track_tqdm=True),  # noqa: B008 — Gradio injects via this default
):
    """Re-run tracking on the last run's predictions, without running inference again."""
    if not state:
        gr.Warning("Run inference once before retracking.")
        return None, gr.update()

    progress(0, desc=f"Retracking with {tracker_type}...")
    stem = Path(state["video_path"]).stem
    target = retrack(
        Path(state["results_dir"]) / f"{stem}_annotations.json",
        state["video_path"],
        tracker_type,
        output_dir=_tmp_dir("mouselite_retrack_"),
    )
    return str(target), _exports(state["results_dir"], state["video_path"])


# The one training process the app drives, shared by every browser tab since they share
# the GPU: its Process, output_dir and epochs.
_job: dict = {}


def _training() -> bool:
    return bool(_job) and _job["process"].is_alive()


def _stop_training() -> None:
    """Interrupt the training process group, so train() records it as interrupted."""
    if _training():
        try:
            os.killpg(_job["process"].pid, signal.SIGINT)
        except ProcessLookupError:  # not yet in its own process group
            os.kill(_job["process"].pid, signal.SIGINT)


atexit.register(_stop_training)  # don't leave a trainer holding the GPU after the app


def train_model(
    dataset_dir,
    dataset_format,
    kind,
    size,
    weights,
    run_name,
    epochs,
    batch_size,
    progress=gr.Progress(),  # noqa: B008 — Gradio injects via this default
):
    """Prepare the dataset here, then train in a separate process."""
    if _training():
        gr.Warning("A model is already training; stop it first.")
        return gr.update(), gr.update()
    if not dataset_dir or not Path(dataset_dir).exists():
        gr.Warning("Point the dataset field at an existing folder.")
        return gr.update(), gr.update()
    output_dir = OUTPUT_DIR / run_name.strip() if run_name.strip() else default_output_dir()
    if output_dir.exists() and any(output_dir.iterdir()):
        gr.Warning(f"{output_dir} already exists; pick another run name.")
        return gr.update(), gr.update()

    progress(0, desc="Preparing the dataset...")
    from_format = None if dataset_format == COCO else dataset_format
    try:
        dataset = prepare_dataset(dataset_dir, output_dir, kind, from_format=from_format)
    except (ValueError, FileNotFoundError, AssertionError, KeyError) as exc:
        gr.Warning(f"Could not read the dataset: {exc}")
        return gr.update(), gr.update()
    missing = [
        split
        for split in ("train", "valid")
        if not (dataset / split / "_annotations.coco.json").exists()
    ]
    if missing:
        gr.Warning(f"No {' or '.join(missing)}/_annotations.coco.json in {dataset}.")
        return gr.update(), gr.update()

    output_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "dataset_dir": dataset,  # already converted
        "kind": kind,
        "size": size,
        "weights": weights,
        "output_dir": output_dir,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
    }
    # spawn, not fork: this process may hold CUDA state and runs Gradio's threads
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=train_logged, args=(str(output_dir / LOG), kwargs))
    process.start()
    _job.update(process=process, output_dir=output_dir, epochs=int(epochs))
    return _banner(), gr.Timer(active=True)


def stop_training():
    _stop_training()
    return gr.update(value="Stopping training...", visible=True)


def _per_epoch(output_dir: Path) -> pd.DataFrame | None:
    """The run's metrics, one row per epoch; None until the first ones are logged."""
    try:
        metrics = pd.read_csv(output_dir / "metrics.csv")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return None
    return metrics.groupby("epoch").last()  # last logged value of each metric


def _plot_data(per_epoch: pd.DataFrame | None, pattern: str) -> pd.DataFrame | None:
    """Long-form `epoch, metric, value` rows of the columns matching `pattern`."""
    if per_epoch is None:
        return None
    columns = [c for c in per_epoch.columns if re.fullmatch(pattern, c)]
    data = per_epoch[columns].reset_index().melt(id_vars="epoch", var_name="metric")
    data = data.dropna()
    return data if len(data) else None  # an empty frame stops the plot drawing later rows


def _log_tail(path: Path, lines: int = 25) -> str:
    """The end of the training log, uncoloured, keeping progress bars' last state."""
    try:
        with open(path, "rb") as f:
            f.seek(max(0, f.seek(0, os.SEEK_END) - 32_000))
            text = f.read().decode(errors="replace")
    except FileNotFoundError:
        return ""
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)  # rich's colours
    rows = [row.rsplit("\r", 1)[-1] for row in text.split("\n")]
    return "\n".join([row for row in rows if row.strip()][-lines:])


def _banner():
    """The training status shown above both tabs."""
    if not _job:
        return gr.update(visible=False)
    name = _job["output_dir"].name
    if _training():
        per_epoch = _per_epoch(_job["output_dir"])
        done = 0  # epochs validated so far
        if per_epoch is not None:
            done = int(per_epoch.filter(like="val/").notna().any(axis=1).sum())
        text = (
            f"Training **{name}**: epoch {done}/{_job['epochs']}. "
            "Inference shares the GPU meanwhile."
        )
    else:
        status = read_run_info(_job["output_dir"]).get("status", "failed")
        text = {
            "finished": f"Training **{name}** finished; pick it in the Model list.",
            "interrupted": f"Training **{name}** was stopped.",
        }.get(status, f"Training **{name}** failed; see the log in the Train tab.")
    return gr.update(value=text, visible=True)


def poll_training():
    """Timer tick: refresh the banner, curves and log, and wrap up a finished run."""
    if not _job:
        return (gr.update(),) * 5 + (gr.Timer(active=False), gr.update(), gr.update())
    per_epoch = _per_epoch(_job["output_dir"])
    updates = (
        _banner(),
        _plot_data(per_epoch, r"(train|val)/loss"),
        _plot_data(per_epoch, r"val/(ema_)?(keypoint_)?(map|mAP)_50_95"),
        _log_tail(_job["output_dir"] / LOG),
    )
    if _training():
        return (*updates, gr.update(), gr.Timer(active=True), gr.update(), gr.update())
    finished = read_run_info(_job["output_dir"]).get("status") == "finished"
    return (
        *updates,
        gr.update(visible=finished),
        gr.Timer(active=False),
        gr.update(choices=_checkpoint_choices(*INFERENCE_WEIGHTS)),
        gr.update(choices=_checkpoint_choices(*TRAIN_WEIGHTS)),
    )


def use_trained_model():
    """Select the last run's best checkpoint for inference and switch to that tab."""
    [run] = find_runs((_job["output_dir"],))
    return gr.update(value=str(run["checkpoint"])), gr.Tabs(selected="inference")


def _on_load():
    """Fresh checkpoint lists, and reattach to a run started before a page reload."""
    return (
        gr.update(choices=_checkpoint_choices(*INFERENCE_WEIGHTS)),
        gr.update(choices=_checkpoint_choices(*TRAIN_WEIGHTS)),
        _banner(),
        gr.Timer(active=bool(_job)),
    )


def build_demo(example_videos: list[Path]) -> gr.Blocks:
    """Build the demo's interface."""
    with gr.Blocks(title="MouseLite") as demo:
        gr.HTML(_header_html(), padding=False, container=False)
        banner = gr.Markdown(visible=False)
        timer = gr.Timer(3, active=False)

        with gr.Tabs() as tabs:
            with gr.Tab("Inference", id="inference"):
                last_run = gr.State()

                with gr.Row():
                    video_in = gr.Video(
                        sources=["upload"],
                        label="Upload a video",
                        height=420,
                    )
                    video_out = gr.Video(
                        label="Annotated output", interactive=False, height=420
                    )

                if example_videos:
                    with gr.Accordion("Examples", open=True):
                        examples = gr.Examples(
                            [[str(video)] for video in example_videos],
                            inputs=video_in,
                        )
                        examples.dataset.show_label = (
                            False  # the accordion carries the title
                        )

                with gr.Row():
                    run_btn = gr.Button("Run", variant="primary", size="lg")
                    retrack_btn = gr.Button("Retrack", size="lg")
                exports_out = gr.File(
                    label="Results",
                    file_count="multiple",
                    interactive=False,
                    visible=False,
                )

                with gr.Accordion("Configuration", open=True), gr.Row():
                    with gr.Column():
                        with gr.Row(equal_height=True):
                            checkpoint = gr.Dropdown(
                                choices=list(INFERENCE_WEIGHTS),
                                value=MOUSELITE,
                                label="Model",
                                info="The released weights, or a checkpoint you trained. "
                                "Paste a path to use one from elsewhere.",
                                allow_custom_value=True,
                                scale=5,
                            )
                            refresh_btn = gr.Button("Refresh", size="sm", scale=1)
                        kind = gr.Dropdown(
                            choices=KINDS,
                            value="keypoints",
                            label="Kind",
                            info="Which model to run: pose, detection or segmentation.",
                        )
                        model_size = gr.Dropdown(
                            choices=SIZES,
                            value="medium",
                            label="Size",
                            info="Larger models are more accurate but slower. "
                            "Not used by keypoints.",
                            interactive=False,
                        )
                        tracker_type = gr.Dropdown(
                            choices=list(TRACKERS),
                            value="bytetrack",
                            label="Tracker",
                            info="Multi-object tracking algorithm.",
                        )

                    with gr.Column():
                        threshold = gr.Slider(
                            minimum=0.05,
                            maximum=1.0,
                            step=0.05,
                            value=0.5,
                            label="Threshold",
                            info="Minimum confidence for a prediction to be kept.",
                        )
                        nms_threshold = gr.Slider(
                            minimum=0.05,
                            maximum=1.0,
                            step=0.05,
                            value=0.5,
                            label="NMS threshold",
                            info="Drop the lower-scoring of two predictions overlapping "
                            "above this.",
                        )
                        top_k = gr.Slider(
                            minimum=1,
                            maximum=10,
                            step=1,
                            value=2,
                            label="Max animals",
                            info="Keep only the highest-scoring predictions per frame.",
                        )
                        every = gr.Slider(
                            minimum=1,
                            maximum=10,
                            step=1,
                            value=1,
                            label="Inference stride",
                            info="Run inference on 1 of every N frames.",
                        )

            with gr.Tab("Train", id="train"):
                with gr.Row():
                    with gr.Column():
                        dataset_dir = gr.Textbox(
                            label="Dataset",
                            placeholder="/path/to/dataset",
                            info="A COCO dataset (train/ and valid/ folders), or a "
                            "DeepLabCut / Lightning Pose project.",
                        )
                        dataset_format = gr.Dropdown(
                            choices=[COCO, *FORMATS],
                            value=COCO,
                            label="Format",
                            info="Projects in other formats are converted into the run "
                            "folder.",
                        )
                        run_name = gr.Textbox(
                            label="Run name",
                            placeholder="defaults to the date and time",
                            info=f"Checkpoints are written to {OUTPUT_DIR}/<run name>.",
                        )
                    with gr.Column():
                        train_weights = gr.Dropdown(
                            choices=list(TRAIN_WEIGHTS),
                            value=MOUSELITE,
                            label="Start from",
                            info="Released weights are usually the best start for mice; "
                            "or continue from a checkpoint you trained.",
                            allow_custom_value=True,
                        )
                        train_kind = gr.Dropdown(
                            choices=KINDS, value="keypoints", label="Kind"
                        )
                        train_size = gr.Dropdown(
                            choices=SIZES, value="medium", label="Size", interactive=False
                        )
                    with gr.Column():
                        epochs = gr.Slider(
                            minimum=1, maximum=200, step=1, value=30, label="Epochs"
                        )
                        batch_size = gr.Slider(
                            minimum=1,
                            maximum=32,
                            step=1,
                            value=4,
                            label="Batch size",
                            info="Lower it if the GPU runs out of memory.",
                        )
                with gr.Row():
                    start_btn = gr.Button(
                        "Start training",
                        variant="primary",
                        size="lg",
                    )
                    stop_btn = gr.Button("Stop", size="lg")
                    use_btn = gr.Button("Use for inference", size="lg", visible=False)
                with gr.Row():
                    loss_plot = gr.LinePlot(
                        x="epoch", y="value", color="metric", label="Loss"
                    )
                    map_plot = gr.LinePlot(
                        x="epoch", y="value", color="metric", label="Validation mAP 50:95"
                    )
                log_box = gr.Textbox(label="Log", lines=12, max_lines=12, autoscroll=True)

        gr.HTML(
            '<div style="text-align:center;padding:10px 0;'
            "border-top:1px solid var(--border-color-primary);"
            'color:var(--body-text-color-subdued)">'
            f'<a href="{REPO}" target="_blank" rel="noopener" style="color:inherit">'
            "MouseLite on GitHub</a></div>",
            padding=False,
            container=False,
        )

        kind.change(_toggle_size, inputs=[kind, checkpoint], outputs=model_size)
        checkpoint.change(
            _lock_model, inputs=[checkpoint, kind], outputs=[kind, model_size]
        )
        refresh_btn.click(
            lambda: gr.update(choices=_checkpoint_choices(*INFERENCE_WEIGHTS)),
            outputs=checkpoint,
        )

        run_btn.click(
            lambda: gr.update(visible=False),
            outputs=exports_out,
        ).then(
            run_inference,
            inputs=[
                video_in,
                checkpoint,
                kind,
                model_size,
                tracker_type,
                threshold,
                nms_threshold,
                top_k,
                every,
            ],
            outputs=[video_out, exports_out, last_run],
        )

        retrack_btn.click(
            run_retrack,
            inputs=[last_run, tracker_type],
            outputs=[video_out, exports_out],
        )

        train_kind.change(
            _toggle_size, inputs=[train_kind, train_weights], outputs=train_size
        )
        train_weights.change(
            _lock_model,
            inputs=[train_weights, train_kind],
            outputs=[train_kind, train_size],
        )
        start_btn.click(
            lambda: gr.update(visible=False),
            outputs=use_btn,
        ).then(
            train_model,
            inputs=[
                dataset_dir,
                dataset_format,
                train_kind,
                train_size,
                train_weights,
                run_name,
                epochs,
                batch_size,
            ],
            outputs=[banner, timer],
        )
        stop_btn.click(stop_training, outputs=banner)
        timer.tick(
            poll_training,
            outputs=[
                banner,
                loss_plot,
                map_plot,
                log_box,
                use_btn,
                timer,
                checkpoint,
                train_weights,
            ],
        )
        use_btn.click(use_trained_model, outputs=[checkpoint, tabs])
        demo.load(_on_load, outputs=[checkpoint, train_weights, banner, timer])
    return demo


def main(
    share: bool = False,
    host: str | None = None,
    port: int | None = None,
    inbrowser: bool = True,
) -> None:
    """Serve the demo. Called by `mouselite app`."""
    example_videos = _example_videos()
    build_demo(example_videos).launch(
        theme=THEME,
        favicon_path=str(ICON),
        share=share,
        server_name=host,
        server_port=port,
        footer_links=[],
        inbrowser=inbrowser,
        # the clips live in the HF cache, outside the cwd and temp dirs gradio serves
        allowed_paths=[str(path) for path in example_videos],
    )


if __name__ == "__main__":
    main()
