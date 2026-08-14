from pathlib import Path

from rfdetr import RFDETRKeypointPreview
from rfdetr.config import RFDETRKeypointPreviewConfig

RESOLUTION = RFDETRKeypointPreviewConfig.model_fields["resolution"].default


def make_rfdetr_keypoints(
    schema,
    resolution: int,
    weights: Path | None = None,
) -> RFDETRKeypointPreview:
    """Instantiate the preview keypoint model against a dataset's schema."""
    fields = RFDETRKeypointPreviewConfig.model_fields
    patch_size = fields["patch_size"].default
    block = patch_size * fields["num_windows"].default
    if resolution <= 0 or resolution % block:
        raise ValueError(f"resolution must be a positive multiple of {block}")

    model = RFDETRKeypointPreview(
        num_classes=len(schema.class_names),
        num_keypoints_per_class=schema.num_keypoints_per_class,
        resolution=resolution,
        # This variant derives its PE from the resolution, so a non-default
        # resolution has to carry the PE with it or the backbone mismatches.
        positional_encoding_size=resolution // patch_size,
        **(
            {"pretrain_weights": str(weights), "trust_checkpoint": True}
            if weights
            else {}
        ),
    )
    model.model_config.model_name = type(model).__name__
    print(f"  model {type(model).__name__} (resolution {resolution})")
    return model
