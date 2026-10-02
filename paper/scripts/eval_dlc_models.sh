#!/usr/bin/env bash
# Score every SuperAnimal pose/detector pair zero-shot over the leave-one-out folds.
#
#     scripts/eval_dlc_models.sh [split ...]
#
# With no splits, eval_dlc.py builds and scores all folds. Each pair collects under
# runs/zeroshot/<pose>_<detector>/. Run from paper/.
set -euo pipefail

cd "$(dirname "$0")/.."

MODELS=(hrnet_w32 resnet_50)
DETECTORS=(fasterrcnn_resnet50_fpn_v2 fasterrcnn_mobilenet_v3_large_fpn)

for model in "${MODELS[@]}"; do
    for detector in "${DETECTORS[@]}"; do
        echo "=== ${model} + ${detector} ==="
        uv run --project dlc python eval_dlc.py --zero-shot \
            --model="${model}" --detector="${detector}" "$@"
    done
done
