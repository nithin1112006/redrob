# Tokenizer setup and formatting utilities

from transformers import AutoTokenizer
from typing import Any

def get_tokenizer(model_name: str, trust_remote_code: bool = True, local_files_only: bool = False) -> Any:
    """Instantiates the tokenizer with proper padding token settings and offline capabilities."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        use_fast=True,
        local_files_only=local_files_only
    )
    
    # Handle padding token for causal LM
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = "<|endoftext|>"
            
    tokenizer.padding_side = "right"  # Standard for training, can toggle to left for batch inference
    return tokenizer

def apply_chat_template(tokenizer: Any, messages: list, tokenize: bool = False, add_generation_prompt: bool = False) -> str:
    """Applies the default Qwen/Hugging Face chat template to messages."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=tokenize,
        add_generation_prompt=add_generation_prompt
    )
