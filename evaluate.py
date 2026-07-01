# Core evaluation entry point

import os
import json
import argparse
import torch
from tqdm import tqdm
from src.utils import setup_logging, load_yaml
from src.constants import TRAINING_CONFIG_PATH, DATASET_CONFIG_PATH, INFERENCE_CONFIG_PATH
from src.inference_engine import InferenceEngine
from src.evaluator import ModelEvaluator

logger = setup_logging()

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Qwen model checkpoints")
    parser.add_argument("--train_config", type=str, default=TRAINING_CONFIG_PATH, help="Path to training config YAML")
    parser.add_argument("--dataset_config", type=str, default=DATASET_CONFIG_PATH, help="Path to dataset config YAML")
    parser.add_argument("--inference_config", type=str, default=INFERENCE_CONFIG_PATH, help="Path to inference config YAML")
    parser.add_argument("--adapter_path", type=str, default=None, help="Specific adapter path (defaults to final_adapter)")
    parser.add_argument("--output_report", type=str, default="./outputs/evaluation_report.md", help="Path to write the evaluation report")
    args = parser.parse_args()
    
    train_conf = load_yaml(args.train_config)
    dataset_conf = load_yaml(args.dataset_config)
    inf_conf = load_yaml(args.inference_config)
    
    # Locate checkpoint adapter
    adapter_path = args.adapter_path
    if not adapter_path:
        adapter_path = os.path.join(train_conf["training"]["output_dir"], "final_adapter")
        if not os.path.exists(adapter_path):
            logger.warning(f"Final adapter path not found at {adapter_path}. Using base model configuration without adapters.")
            adapter_path = None
            
    # Initialize inference engine
    base_model = train_conf["model"]["base_model_name"]
    logger.info(f"Initializing inference engine with base: {base_model} and adapter: {adapter_path}")
    
    engine = InferenceEngine(
        base_model_path=base_model,
        adapter_path=adapter_path,
        inference_config=inf_conf
    )
    
    # Load test dataset
    test_path = dataset_conf["splits"]["test_path"]
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test dataset split not found at: {test_path}. Please run prepare_dataset.py first.")
        
    logger.info(f"Loading test records from {test_path}...")
    cips = []
    references = []
    styles = []
    
    with open(test_path, 'r') as f:
        for line in f:
            rec = json.loads(line)
            cips.append(rec["candidate_package"])
            references.append(rec["explanation"])
            styles.append(rec.get("style", "Recruiter"))
            
    logger.info(f"Loaded {len(cips)} test cases. Starting batch generation...")
    
    # Run predictions (cap to first 100 for evaluation efficiency if testing set is huge, otherwise run full)
    predictions = []
    eval_cips = cips[:100]  # Let's evaluate first 100 to avoid high latency during local test runs
    eval_references = references[:100]
    eval_styles = styles[:100]
    
    for idx, (cip, style) in enumerate(tqdm(zip(eval_cips, eval_styles), total=len(eval_cips), desc="Evaluating")):
        explanation, _ = engine.generate_explanation(cip, style=style, max_retries=2)
        predictions.append(explanation)
        
    # Evaluate predictions using the Evaluator
    evaluator = ModelEvaluator()
    evaluator.evaluate_predictions(
        cips=eval_cips,
        predictions=predictions,
        references=eval_references,
        output_report_path=args.output_report
    )

if __name__ == "__main__":
    main()
