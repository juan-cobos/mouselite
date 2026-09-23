"""Gradio demo wrapping the mouselite Pipeline.

Install the extra and run with:
    uv run --extra app mouselite app
"""

import re
import tempfile
from functools import lru_cache
from pathlib import Path

import gradio as gr
import supervision as sv

from mouselite.models import MODELS, get_model
from mouselite.pipeline import Pipeline
from mouselite.tracker import TRACKERS, get_tracker, retrack

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
EXPORTS = ("annotations.json", "trajectories.csv", "summary.csv")


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
def _load_model(kind: str, size: str):
    """Cache weights so repeated runs don't re-download and re-init the model."""
    return get_model(kind, size=size)


def _tmp_dir(prefix: str = "mouselite_") -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix))


def _exports(results_dir: str | Path):
    """Value for the downloads component: the COCO export and CSV tables in `results_dir`."""
    files = [str(Path(results_dir) / name) for name in EXPORTS]
    return gr.update(value=files, visible=True)


def run_inference(
    video_path,
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
    model = _load_model(kind, model_size)

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
    return str(video_out_path), _exports(results_dir), state


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
    target = retrack(
        Path(state["results_dir"]) / "annotations.json",
        state["video_path"],
        tracker_type,
        output_dir=_tmp_dir("mouselite_retrack_"),
    )
    return str(target), _exports(state["results_dir"])


def _toggle_size(kind):
    """Keypoints ships a single checkpoint, so size does not apply to it."""
    return gr.update(interactive=kind != "keypoints")


with gr.Blocks(title="MouseLite") as demo:
    gr.HTML(_header_html(), padding=False, container=False)

    last_run = gr.State()

    with gr.Row():
        video_in = gr.Video(
            sources=["upload"],
            label="Upload a video",
            height=420,
        )
        video_out = gr.Video(label="Annotated output", interactive=False, height=420)

    with gr.Row():
        run_btn = gr.Button("Run", variant="primary", size="lg")
        retrack_btn = gr.Button("Retrack", size="lg")
    exports_out = gr.File(
        label="Results: COCO export, trajectories and summary",
        file_count="multiple",
        interactive=False,
        visible=False,
    )

    with gr.Accordion("Configuration", open=True), gr.Row():
        with gr.Column():
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
                info="Larger models are more accurate but slower. Not used by keypoints.",
                interactive=False,
            )
            tracker_type = gr.Dropdown(
                choices=list(TRACKERS),
                value="ocsort",
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
                info="Drop the lower-scoring of two predictions overlapping above this.",
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
    gr.HTML(
        '<div style="text-align:center;padding:10px 0;'
        "border-top:1px solid var(--border-color-primary);"
        'color:var(--body-text-color-subdued)">'
        f'<a href="{REPO}" target="_blank" rel="noopener" style="color:inherit">'
        "MouseLite on GitHub</a></div>",
        padding=False,
        container=False,
    )

    kind.change(_toggle_size, inputs=kind, outputs=model_size)

    run_btn.click(
        lambda: gr.update(visible=False),
        outputs=exports_out,
    ).then(
        run_inference,
        inputs=[
            video_in,
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


def main(
    share: bool = False,
    host: str | None = None,
    port: int | None = None,
    inbrowser: bool = True,
) -> None:
    """Serve the demo. Called by `mouselite app`."""
    demo.launch(
        theme=THEME,
        favicon_path=str(ICON),
        share=share,
        server_name=host,
        server_port=port,
        footer_links=[],
        inbrowser=inbrowser,
    )


if __name__ == "__main__":
    main()
