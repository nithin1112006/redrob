# Splitter and preparer script for SFT training

import os
import json
import argparse
import random
from src.utils import setup_logging, load_yaml, set_seed
from src.constants import DATASET_CONFIG_PATH

logger = setup_logging()

def main():
    parser = argparse.ArgumentParser(description="Split raw synthetic dataset into train, validation, and test sets")
    parser.add_argument("--config", type=str, default=DATASET_CONFIG_PATH, help="Path to dataset config YAML")
    args = parser.parse_args()
    
    config = load_yaml(args.config)
    seed = config["generation"].get("seed", 42)
    set_seed(seed)
    
    raw_file = os.path.join(config["generation"]["output_dir"], "raw_synthetic_dataset.jsonl")
    if not os.path.exists(raw_file):
        raise FileNotFoundError(
            f"Raw dataset file not found at: {raw_file}. Please run python generate_synthetic_dataset.py first."
        )
        
    logger.info(f"Loading raw dataset from {raw_file}...")
    with open(raw_file, 'r') as f:
        records = [json.loads(line) for line in f]
        
    logger.info(f"Total records loaded: {len(records)}")
    
    # Shuffle dataset
    random.shuffle(records)
    
    total = len(records)
    train_ratio = config["splits"]["train_ratio"]
    val_ratio = config["splits"]["validation_ratio"]
    
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_set = records[:train_end]
    val_set = records[train_end:val_end]
    test_set = records[val_end:]
    
    logger.info(f"Splits sizes - Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
    
    # Save files
    os.makedirs(os.path.dirname(config["splits"]["train_path"]), exist_ok=True)
    
    for split_data, path in [
        (train_set, config["splits"]["train_path"]),
        (val_set, config["splits"]["validation_path"]),
        (test_set, config["splits"]["test_path"])
    ]:
        logger.info(f"Saving split to {path}...")
        with open(path, 'w') as f:
            for rec in split_data:
                f.write(json.dumps(rec) + "\n")
                
    logger.info("Dataset preparation complete.")

if __name__ == "__main__":
    main()
