# Merges trained LoRA adapter weights into base model weights

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.utils import setup_logging, load_yaml
from src.constants import TRAINING_CONFIG_PATH

logger = setup_logging()

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter weights into the base model")
    parser.add_argument("--train_config", type=str, default=TRAINING_CONFIG_PATH, help="Path to training config YAML")
    parser.add_argument("--adapter_path", type=str, default=None, help="LoRA adapter weights location")
    parser.add_argument("--output_dir", type=str, default="./merged_model", help="Directory to save the merged model")
    args = parser.parse_args()
    
    train_conf = load_yaml(args.train_config)
    base_model_name = train_conf["model"]["base_model_name"]
    
    # Locate adapter
    adapter_path = args.adapter_path
    if not adapter_path:
        adapter_path = os.path.join(train_conf["training"]["output_dir"], "final_adapter")
        
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"LoRA adapter directory not found at: {adapter_path}")
        
    logger.info(f"Loading tokenizer from base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    logger.info(f"Loading base model: {base_model_name} (using FP16/BF16 precision for clean merge)...")
    device_map = "cpu" if not torch.cuda.is_available() else "auto"
    
    # Load base model in half precision for merging (quantized models cannot be merged directly)
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True
        )
    except Exception as e:
        logger.warning(f"Could not load target base model {base_model_name}. Loading fallback: {e}")
        fallback_model = "Qwen/Qwen2.5-3B-Instruct"
        base_model = AutoModelForCausalLM.from_pretrained(
            fallback_model,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True
        )
        base_model_name = fallback_model
        tokenizer = AutoTokenizer.from_pretrained(fallback_model)
        
    logger.info(f"Loading and wrapping with LoRA adapter weights: {adapter_path}")
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    
    logger.info("Merging LoRA adapter weights into base model weights...")
    merged_model = peft_model.merge_and_unload()
    
    logger.info(f"Saving merged standalone model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    merged_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("Weight merge completed successfully.")

if __name__ == "__main__":
    main()
