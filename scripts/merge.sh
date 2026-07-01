#!/bin/bash
# Shell wrapper to merge LoRA weights

set -e
cd "$(dirname "$0")/.."

echo "Merging LoRA adapter weights into base model weights..."
python merge_lora.py \
    --train_config ./configs/training.yaml \
    --output_dir ./merged_model
