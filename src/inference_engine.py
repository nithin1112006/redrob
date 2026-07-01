# Inference pipeline and candidate fact generation validator

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.prompts import create_messages
from src.validator import GroundednessValidator
from src.utils import setup_logging, check_bf16_support
from typing import Dict, Any, Tuple

logger = setup_logging()

class InferenceEngine:
    """Orchestrates model generation, chat formatting, and fact verification/regeneration."""
    
    def __init__(self, 
                 base_model_path: str, 
                 adapter_path: str = None, 
                 inference_config: dict = None,
                 trust_remote_code: bool = True):
        self.config = inference_config or {}
        self.gen_config = self.config.get("generation", {})
        self.val_config = self.config.get("validation", {})
        
        # Determine device target (CPU or CUDA)
        config_device = self.config.get("device", "cpu").lower()
        if config_device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
            
        # Determine offline mode
        self.offline = self.config.get("offline", False)
        
        logger.info(f"Targeting inference device: {self.device} (Offline mode: {self.offline})")
        
        # Load tokenizer offline-safe
        logger.info(f"Loading tokenizer from {base_model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=trust_remote_code,
            local_files_only=self.offline
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Determine model datatype
        if self.device == "cuda":
            bf16_supported = check_bf16_support()
            compute_dtype = torch.bfloat16 if bf16_supported else torch.float16
        else:
            # Standard CPU type (bfloat16 can run on modern CPUs, fallback to float32 if needed)
            compute_dtype = torch.float32  # Standard for clean CPU computation
            
        logger.info(f"Loading model {base_model_path} with dtype {compute_dtype} and low_cpu_mem_usage=True...")
        
        # Load base model safely
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=compute_dtype,
                device_map={"": self.device},
                trust_remote_code=trust_remote_code,
                local_files_only=self.offline,
                low_cpu_mem_usage=True
            )
        except Exception as e:
            logger.warning(f"Could not load base model {base_model_path}. Loading Qwen/Qwen2.5-3B-Instruct as fallback: {e}")
            fallback_model = "Qwen/Qwen2.5-3B-Instruct"
            self.model = AutoModelForCausalLM.from_pretrained(
                fallback_model,
                torch_dtype=compute_dtype,
                device_map={"": self.device},
                trust_remote_code=trust_remote_code,
                local_files_only=self.offline,
                low_cpu_mem_usage=True
            )
            # Re-load tokenizer for fallback
            self.tokenizer = AutoTokenizer.from_pretrained(
                fallback_model,
                local_files_only=self.offline
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load LoRA adapter if provided
        if adapter_path:
            logger.info(f"Loading LoRA adapter weights from {adapter_path}...")
            # Quantized base models cannot load adapters easily on CPU without PEFT mapping helper
            self.model = PeftModel.from_pretrained(
                self.model, 
                adapter_path, 
                local_files_only=self.offline
            )
            
        self.model.eval()

    def generate_explanation(self, 
                             cip: Dict[str, Any], 
                             style: str = "Recruiter",
                             max_retries: int = 3) -> Tuple[str, bool]:
        """Generates recruiter explanation, verifying it against the GroundednessValidator.
        
        If violations are found and strict validation is on, it will retry.
        
        Returns:
            Tuple[str, bool]: (generated_text, was_valid)
        """
        messages = create_messages(cip, style=style)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generation args
        temperature = self.gen_config.get("temperature", 0.2)
        top_p = self.gen_config.get("top_p", 0.9)
        max_new_tokens = self.gen_config.get("max_new_tokens", 512)
        repetition_penalty = self.gen_config.get("repetition_penalty", 1.15)
        pad_token_id = self.tokenizer.pad_token_id
        
        explanation = ""
        is_valid = False
        
        for attempt in range(1, max_retries + 1):
            logger.info(f"Generating explanation on {self.device} (Attempt {attempt}/{max_retries})...")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if attempt == 1 else temperature + 0.1 * attempt,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    do_sample=True,
                    pad_token_id=pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Extract response
            input_length = inputs.input_ids.shape[1]
            generated_tokens = outputs[0][input_length:]
            explanation = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            # Run groundedness verification
            is_valid, violations = GroundednessValidator.validate(cip, explanation)
            
            if is_valid:
                logger.info("Explanation verified successfully. Zero hallucinations detected.")
                return explanation, True
            else:
                logger.warning(f"Validation failed on attempt {attempt}. Violations: {violations}")
                
        # If we reach here, we exceeded retries without a valid output
        if self.val_config.get("strict_skills_match", True):
            logger.error("Strict validation mode is enabled. Rejecting explanation due to persistent hallucinations.")
            return explanation, False
            
        logger.warning("Returning invalid explanation because strict mode is disabled.")
        return explanation, False
