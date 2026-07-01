# Interactive / script inference runner

import os
import json
import argparse
import sys
from src.utils import setup_logging, load_yaml
from src.constants import TRAINING_CONFIG_PATH, INFERENCE_CONFIG_PATH
from src.inference_engine import InferenceEngine

logger = setup_logging()

def main():
    parser = argparse.ArgumentParser(description="Generate grounded recruiter explanation from Candidate Intelligence Package")
    parser.add_argument("--json_path", type=str, default=None, help="Path to input candidate package JSON file")
    parser.add_argument("--json_str", type=str, default=None, help="Raw candidate package JSON string")
    parser.add_argument("--style", type=str, default="Recruiter", help="Writing style: Professional, Recruiter, Concise, Technical")
    parser.add_argument("--adapter_path", type=str, default=None, help="LoRA adapter weights location")
    parser.add_argument("--train_config", type=str, default=TRAINING_CONFIG_PATH, help="Path to training config YAML")
    parser.add_argument("--inference_config", type=str, default=INFERENCE_CONFIG_PATH, help="Path to inference config YAML")
    args = parser.parse_args()
    
    # Load configuration files
    train_conf = load_yaml(args.train_config)
    inf_conf = load_yaml(args.inference_config)
    
    # 1. Handle offline mode environment variables
    if inf_conf.get("offline", False):
        logger.info("Setting offline environment switches (HF_HUB_OFFLINE=1)...")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    # 2. Parse Input
    cip = None
    if args.json_path:
        with open(args.json_path, 'r') as f:
            cip = json.load(f)
    elif args.json_str:
        cip = json.loads(args.json_str)
    else:
        # Default mock CIP for testing
        logger.info("No inputs provided. Using fallback Candidate Intelligence Package...")
        cip = {
            "candidate_id": "C123",
            "rank": 7,
            "score": 0.923,
            "technical_fit": 0.95,
            "career_evidence": 0.91,
            "behavior": 0.88,
            "integrity": 1.0,
            "confidence": 0.97,
            "positive_signals": ["Production Retrieval", "Python", "Leadership", "Vector Database"],
            "negative_signals": ["60 Day Notice"],
            "jd_coverage": {
                "must": ["Python", "Retrieval"],
                "preferred": ["Embeddings", "Ranking"],
                "missing": ["Evaluation Metrics"]
            },
            "feature_contributions": {
                "career": 42,
                "technical": 31,
                "behavior": 17,
                "integrity": 10
            }
        }
        
    # Find adapter path
    adapter_path = args.adapter_path
    if not adapter_path:
        adapter_path = os.path.join(train_conf["training"]["output_dir"], "final_adapter")
        if not os.path.exists(adapter_path):
            logger.warning(f"No fine-tuned adapter found at {adapter_path}. Running base model inference.")
            adapter_path = None
            
    base_model = train_conf["model"]["base_model_name"]
    
    # Run Inference
    engine = InferenceEngine(
        base_model_path=base_model,
        adapter_path=adapter_path,
        inference_config=inf_conf
    )
    
    explanation, is_valid = engine.generate_explanation(cip, style=args.style)
    
    print("\n" + "="*50)
    print("CANDIDATE INTELLIGENCE PACKAGE:")
    print(json.dumps(cip, indent=2))
    print("-"*50)
    print(f"GENERATED RECRUITER EXPLANATION ({args.style} Style):")
    print(explanation)
    print("-"*50)
    print(f"Validation Status: {'PASSED (Zero Hallucinations)' if is_valid else 'FAILED (Hallucination Detected/Rejected)'}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
