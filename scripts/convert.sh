#!/bin/bash
# Shell wrapper to generate GGUF convert setups

set -e
cd "$(dirname "$0")/.."

echo "Generating Modelfile and GGUF steps..."
python convert_to_gguf.py \
    --model_dir ./merged_model \
    --output_file ./Modelfile
