import os
import argparse
from src.utils import setup_logging, load_yaml, init_directories
from src.dataset import load_explanation_dataset
from src.trainer import ExplanationTrainer
from src.constants import TRAINING_CONFIG_PATH, LORA_CONFIG_PATH, DATASET_CONFIG_PATH
from src.tokenizer import get_tokenizer

logger = setup_logging()

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-4B-Instruct using QLoRA")
    parser.add_argument("--train_config", type=str, default=TRAINING_CONFIG_PATH, help="Path to training config YAML")
    parser.add_argument("--lora_config", type=str, default=LORA_CONFIG_PATH, help="Path to lora config YAML")
    parser.add_argument("--dataset_config", type=str, default=DATASET_CONFIG_PATH, help="Path to dataset config YAML")
    args = parser.parse_args()
    
    # Initialize dirs
    init_directories()
    
    # Load configs
    train_conf = load_yaml(args.train_config)
    lora_conf = load_yaml(args.lora_config)
    dataset_conf = load_yaml(args.dataset_config)
    
    # Pre-load tokenizer to format dataset correctly
    logger.info("Initializing tokenizer for dataset mapping...")
    tokenizer = get_tokenizer(
        train_conf["model"]["base_model_name"],
        trust_remote_code=train_conf["model"].get("trust_remote_code", True)
    )
    
    # Load and map splits
    logger.info("Loading training and validation datasets...")
    train_path = dataset_conf["splits"]["train_path"]
    val_path = dataset_conf["splits"]["validation_path"]
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(
            f"Dataset files not found at {train_path} or {val_path}. Please run prepare_dataset.py first."
        )
        
    train_dataset = load_explanation_dataset(train_path, tokenizer=tokenizer)
    val_dataset = load_explanation_dataset(val_path, tokenizer=tokenizer)
    
    logger.info(f"Loaded {len(train_dataset)} training examples.")
    logger.info(f"Loaded {len(val_dataset)} validation examples.")
    
    # Start fine-tuning
    trainer_engine = ExplanationTrainer(train_conf, lora_conf)
    trainer_engine.train(train_dataset, val_dataset)

if __name__ == "__main__":
    main()
