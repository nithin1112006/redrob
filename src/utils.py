# Helper utilities for config loading, system checks, and logging

import os
import yaml
import random
import numpy as np
import logging
from typing import Dict, Any


def load_yaml(file_path: str) -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found at: {file_path}")
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging(log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """Sets up the global logging configuration."""
    logger = logging.getLogger("Phase2Training")
    logger.setLevel(level)
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d]: %(message)s"
    )
    
    # Stream Handler
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    # File Handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger

def set_seed(seed: int = 42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def check_bf16_support() -> bool:
    """Checks if bfloat16 is natively supported by the GPU."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        major, _ = torch.cuda.get_device_capability()
        return major >= 8
    except ImportError:
        return False


def init_directories():
    """Initializes output and checkpoints directories."""
    from src.constants import DATA_DIR, OUTPUT_DIR, CHECKPOINT_DIR
    for d in [DATA_DIR, OUTPUT_DIR, CHECKPOINT_DIR, os.path.join(DATA_DIR, "synthetic")]:
        os.makedirs(d, exist_ok=True)
