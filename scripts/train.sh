#!/bin/bash
# Shell wrapper to execute Phase 2 model training using HF Accelerate config

set -e

# Change directory to project root if run from subfolders
cd "$(dirname "$0")/.."

echo "Initializing training directories..."
mkdir -p checkpoints outputs/logs

echo "Starting training via Accelerate..."
accelerate launch --config_file ./configs/accelerate.yaml train.py \
    --train_config ./configs/training.yaml \
    --lora_config ./configs/lora.yaml \
    --dataset_config ./configs/dataset.yaml
