import argparse
from pathlib import Path

import torch

RUNS = Path(__file__).resolve().parents[1] / "runs"

POSE_NAME = "hrnet_w32"
DETECTOR_NAME = "fasterrcnn_resnet50_fpn_v2"
BUFFER_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")
# The order eval.py scores them in; any of them holds the same architecture, and
# the count is a property of that rather than of which weights won.
RFDETR_CHECKPOINTS = (
    "checkpoint_best_ema.pth",
    "checkpoint_best_regular.pth",
    "checkpoint_best_total.pth",
)


def rfdetr_checkpoint(run_dir: Path) -> Path | None:
    """Best weights a keypoint run wrote, or None if it wrote none."""
    for name in RFDETR_CHECKPOINTS:
        if (path := run_dir / name).exists():
            return path
    return None


def default_rfdetr_run() -> Path:
    """First finished keypoint run on disk."""
    for run in sorted(RUNS.glob("keypoint_preview_*")):
        if rfdetr_checkpoint(run):
            return run
    raise SystemExit(f"no trained RF-DETR keypoint run under {RUNS}")


def default_dlc_run() -> Path:
    """First DeepLabCut run carrying both stages."""
    for run in sorted(RUNS.glob("*_dlc")):
        snapshots = [path.name for path in run.glob("checkpoints/snapshot-*.pt")]
        if any("detector" in name for name in snapshots) and any(
            "detector" not in name for name in snapshots
        ):
            return run
    raise SystemExit(f"no DeepLabCut run with both stages under {RUNS}")


def best_snapshot(checkpoints: Path, detector: bool = False) -> Path:
    """DeepLabCut's own best snapshot, or the last numbered epoch without one."""
    snapshots = [
        path
        for path in checkpoints.glob("snapshot-*.pt")
        if ("detector" in path.name) == detector
    ]
    kind = "detector" if detector else "pose"
    if best := sorted(path for path in snapshots if "best" in path.name):
        chosen = best[-1]
    elif numbered := sorted(path for path in snapshots if "best" not in path.name):
        chosen = numbered[-1]  # zero-padded, so lexical order is epoch order
    else:
        raise SystemExit(f"no {kind} snapshot in {checkpoints}")

    print(f"  {kind + ' snapshot':<17} {chosen.name}")
    return chosen


def checkpoint_params(path: Path) -> int:
    """Parameter count of a checkpoint, from its state dict alone.

    Both families store their weights under ``model``. Buffers ride in the same
    dict and are not parameters: DeepLabCut's are batch-norm statistics, named
    by suffix, and RF-DETR's are the boolean keypoint masks, which the dtype
    test catches.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)["model"]
    return sum(
        tensor.numel()
        for key, tensor in state.items()
        if tensor.is_floating_point() and not key.endswith(BUFFER_SUFFIXES)
    )


def main(rfdetr_run: Path, dlc_run: Path) -> None:
    checkpoints = dlc_run if dlc_run.name == "checkpoints" else dlc_run / "checkpoints"
    if not checkpoints.is_dir():
        raise SystemExit(f"no checkpoints at {checkpoints}")
    if (weights := rfdetr_checkpoint(rfdetr_run)) is None:
        raise SystemExit(
            f"no checkpoint in {rfdetr_run}; looked for {', '.join(RFDETR_CHECKPOINTS)}",
        )

    print(f"\ncounting {rfdetr_run}")
    print(f"  {'checkpoint':<17} {weights.name}")
    rfdetr = checkpoint_params(weights)

    print(f"\ncounting {checkpoints}")
    pose = checkpoint_params(best_snapshot(checkpoints))
    detector = checkpoint_params(best_snapshot(checkpoints, detector=True))

    rows = [
        ("RF-DETR keypoint preview", rfdetr),
        (f"DLC pose ({POSE_NAME})", pose),
        (f"DLC detector ({DETECTOR_NAME})", detector),
        ("DLC total (pose + detector)", pose + detector),
    ]
    print(f"\n  {'model':<44} {'params':>14} {'M':>8}")
    for name, count in rows:
        print(f"  {name:<44} {count:>14,} {count / 1e6:>8.1f}")

    print(f"\n  RF-DETR / DLC total: {rfdetr / (pose + detector):.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rfdetr", type=Path, default=None, help="keypoint run dir")
    parser.add_argument("--dlc", type=Path, default=None, help="DeepLabCut run dir")
    args = parser.parse_args()
    main(args.rfdetr or default_rfdetr_run(), args.dlc or default_dlc_run())
