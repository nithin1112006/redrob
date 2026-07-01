# Dataset parsing and template application module

import json
from datasets import Dataset, load_dataset
from src.prompts import create_messages
from typing import Dict, Any

def format_dataset_example(example: Dict[str, Any], tokenizer: Any = None) -> Dict[str, Any]:
    """Formats an example candidate profile into chat template messages."""
    cip = example.get("candidate_package", {})
    if not cip and "candidate_id" in example:
        # Reconstruct cip if stored flat
        cip = {
            "candidate_id": example.get("candidate_id"),
            "rank": example.get("rank"),
            "score": example.get("score"),
            "technical_fit": example.get("technical_fit"),
            "career_evidence": example.get("career_evidence"),
            "behavior": example.get("behavior"),
            "integrity": example.get("integrity"),
            "confidence": example.get("confidence"),
            "positive_signals": example.get("positive_signals", []),
            "negative_signals": example.get("negative_signals", []),
            "jd_coverage": example.get("jd_coverage", {}),
            "feature_contributions": example.get("feature_contributions", {})
        }
    
    style = example.get("style", "Recruiter")
    explanation = example.get("explanation", "")
    
    messages = create_messages(cip, style=style, response=explanation)
    
    formatted = {"messages": messages}
    if tokenizer is not None:
        formatted["text"] = tokenizer.apply_chat_template(messages, tokenize=False)
        
    return formatted

def load_explanation_dataset(file_path: str, tokenizer: Any = None) -> Dataset:
    """Loads a JSONL dataset file and maps it to the format required by the TRL SFTTrainer."""
    dataset = load_dataset("json", data_files=file_path, split="train")
    
    # Apply format conversion
    mapped_dataset = dataset.map(
        lambda x: format_dataset_example(x, tokenizer),
        desc="Formatting dataset to chat templates"
    )
    
    return mapped_dataset
