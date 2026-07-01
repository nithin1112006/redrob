#!/bin/bash
# Shell wrapper to benchmark model inference metrics

set -e
cd "$(dirname "$0")/.."

echo "Benchmarking model generation parameters..."
python benchmark.py \
    --train_config ./configs/training.yaml \
    --num_runs 5
