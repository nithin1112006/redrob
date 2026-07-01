# Shared constants for Redrob Phase 2 pipeline

import os

# Base paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

# Config File Paths
TRAINING_CONFIG_PATH = os.path.join(CONFIG_DIR, "training.yaml")
LORA_CONFIG_PATH = os.path.join(CONFIG_DIR, "lora.yaml")
DATASET_CONFIG_PATH = os.path.join(CONFIG_DIR, "dataset.yaml")
INFERENCE_CONFIG_PATH = os.path.join(CONFIG_DIR, "inference.yaml")

# Default Styles
STYLES = ["Professional", "Recruiter", "Concise", "Technical"]

# Role markers
SYSTEM_ROLE = "system"
USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"
