#!/bin/bash
# Shell wrapper to evaluate check-pointed model performance

set -e
cd "$(dirname "$0")/.."

echo "Evaluating the model against the test set..."
python evaluate.py \
    --train_config ./configs/training.yaml \
    --dataset_config ./configs/dataset.yaml \
    --inference_config ./configs/inference.yaml \
    --output_report ./outputs/evaluation_report.md
