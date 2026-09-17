from pathlib import Path

from huggingface_hub import hf_hub_download

MODELS = {
    "detection": {
        "nano": "RFDETRNano",
        "small": "RFDETRSmall",
        "medium": "RFDETRMedium",
        "large": "RFDETRLarge",
    },
    "segmentation": {
        "nano": "RFDETRSegNano",
        "small": "RFDETRSegSmall",
        "medium": "RFDETRSegMedium",
        "large": "RFDETRSegLarge",
    },
    "keypoints": "RFDETRKeypointPreview",
}

HF_REPO_ID = "mouselite/mouselite"  # TODO: placeholder for HF model weights repo
WEIGHTS = "mouselite-{kind}-{size}.pt"

BASE = "base"  # rfdetr's own pretrained weights, the fine-tuning starting point
MOUSELITE = "mouselite"  # our fine-tuned weights from HF_REPO_ID


def build_model(kind: str, size: str = "medium", weights: str | Path = MOUSELITE):
    """Construct the rfdetr model with `weights`: BASE, MOUSELITE or a local checkpoint."""
    import rfdetr

    cls = getattr(rfdetr, MODELS[kind] if kind == "keypoints" else MODELS[kind][size])
    if weights == BASE:
        return cls()  # rfdetr downloads its default pretrain_weights itself
    if weights == MOUSELITE:
        if kind == "keypoints":
            filename = f"mouselite-{kind}.pt"
        else:
            filename = WEIGHTS.format(kind=kind, size=size)
        weights = hf_hub_download(repo_id=HF_REPO_ID, filename=filename)
    return cls(pretrain_weights=str(weights))


def get_model(
    kind: str,
    size: str = "medium",
    checkpoint: str | Path | None = None,
    dtype: str = "float32",
    batch_size: int = 1,
    compile: bool = False,
):
    """Inference-ready model, using the mouselite weights unless `checkpoint` is given."""
    model = build_model(kind, size, weights=checkpoint or MOUSELITE)
    model.inference(compile=compile, batch_size=batch_size, dtype=dtype)
    return model
