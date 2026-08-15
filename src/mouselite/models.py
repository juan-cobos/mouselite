from pathlib import Path

from huggingface_hub import hf_hub_download
from rfdetr import (
    RFDETRKeypointPreview,
    RFDETRLarge,
    RFDETRMedium,
    RFDETRNano,
    RFDETRSegLarge,
    RFDETRSegMedium,
    RFDETRSegNano,
    RFDETRSegSmall,
    RFDETRSmall,
)

MODELS: dict[str, dict[str, type]] = {
    "detection": {
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "medium": RFDETRMedium,
        "large": RFDETRLarge,
    },
    "segmentation": {
        "nano": RFDETRSegNano,
        "small": RFDETRSegSmall,
        "medium": RFDETRSegMedium,
        "large": RFDETRSegLarge,
    },
    "keypoints": RFDETRKeypointPreview,
}

HF_REPO_ID = "mouselite/mouselite"  # TODO: placeholder for HF model weights repo
WEIGHTS = "mouselite-{kind}-{size}.pt"


def get_model(
    kind: str,
    size: str = "medium",
    checkpoint: str | Path | None = None,
    dtype: str = "float32",
    batch_size: int = 1,
    compile: bool = False,
):
    cls = MODELS[kind] if kind == "keypoints" else MODELS[kind][size]
    if checkpoint is not None:
        pretrain_weights = str(checkpoint)
    else:
        if kind == "keypoints":
            filename = f"mouselite-{kind}.pt"
        else:
            filename = WEIGHTS.format(kind=kind, size=size)
        pretrain_weights = hf_hub_download(repo_id=HF_REPO_ID, filename=filename)
    model = cls(pretrain_weights=pretrain_weights)
    model.inference(compile=compile, batch_size=batch_size, dtype=dtype)
    return model
