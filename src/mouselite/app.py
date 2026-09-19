"""Gradio demo wrapping the mouselite Pipeline.

Install the extra and run with:
    uv run --extra app mouselite app
"""

import tempfile
from functools import lru_cache
from pathlib import Path

import gradio as gr
import supervision as sv

from mouselite.models import MODELS, get_model
from mouselite.pipeline import Pipeline
from mouselite.tracker import TRACKERS, get_tracker, retrack

KINDS = list(MODELS)
SIZES = list(MODELS["detection"])


@lru_cache(maxsize=2)
def _load_model(kind: str, size: str):
    """Cache weights so repeated runs don't re-download and re-init the model."""
    return get_model(kind, size=size)


def _tmp_dir(prefix: str = "mouselite_") -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix))


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
    """Run inference on the input video, returning the annotated video and COCO export."""
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

    # Pipeline writes the COCO export next to the annotated video.
    ann_out = video_out_path.parent / "annotations.json"
    state = {"annotations": str(ann_out)}
    return str(video_out_path), gr.update(value=str(ann_out), visible=True), state


def run_retrack(
    state,
    tracker_type,
    progress=gr.Progress(track_tqdm=True),  # noqa: B008 — Gradio injects via this default
):
    """Re-run tracking on the last run's predictions, without running inference again."""
    if not state:
        gr.Warning("Run inference once before retracking.")
        return None

    progress(0, desc=f"Retracking with {tracker_type}...")
    target = retrack(
        state["annotations"],
        tracker_type,
        output_dir=_tmp_dir("mouselite_retrack_"),
    )
    return str(target)


def _toggle_size(kind):
    """Keypoints ships a single checkpoint, so size does not apply to it."""
    return gr.update(interactive=kind != "keypoints")


with gr.Blocks(title="MouseLite") as demo:
    gr.Markdown(
        "# MouseLite\nRun detection, segmentation or pose inference on mice videos.",
    )

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
    annotations_out = gr.File(label="Annotations file", interactive=False, visible=False)

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

    kind.change(_toggle_size, inputs=kind, outputs=model_size)

    run_btn.click(
        lambda: gr.update(visible=False),
        outputs=annotations_out,
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
        outputs=[video_out, annotations_out, last_run],
    )

    retrack_btn.click(
        run_retrack,
        inputs=[last_run, tracker_type],
        outputs=video_out,
    )


def main(
    share: bool = False,
    host: str | None = None,
    port: int | None = None,
    inbrowser: bool = True,
) -> None:
    """Serve the demo. Called by `mouselite app`."""
    demo.launch(
        theme=gr.themes.Monochrome(),
        share=share,
        server_name=host,
        server_port=port,
        footer_links=[],
        inbrowser=inbrowser,
    )


if __name__ == "__main__":
    main()
