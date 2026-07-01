# Model training management and SFTTrainer orchestration

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from src.tokenizer import get_tokenizer
from src.utils import setup_logging, check_bf16_support

logger = setup_logging()

class ExplanationTrainer:
    """Trainer class overseeing quantization, LoRA instantiation, and SFT training arguments."""
    
    def __init__(self, training_config: dict, lora_config: dict):
        self.training_config = training_config
        self.lora_config = lora_config
        
        # Decide compute dtype based on hardware support
        self.bf16_supported = check_bf16_support()
        self.compute_dtype = torch.bfloat16 if self.bf16_supported else torch.float16
        
        logger.info(f"BFloat16 support detected: {self.bf16_supported}")
        logger.info(f"Using compute dtype: {self.compute_dtype}")

    def prepare_model_and_tokenizer(self):
        """Loads model with 4-bit quantization config and PEFT configurations."""
        model_name = self.training_config["model"]["base_model_name"]
        trust_remote_code = self.training_config["model"].get("trust_remote_code", True)
        
        # 1. Setup tokenizer
        logger.info(f"Loading tokenizer for {model_name}...")
        tokenizer = get_tokenizer(model_name, trust_remote_code)
        
        # 2. Setup Quantization Configuration
        q_conf = self.lora_config.get("quantization", {})
        logger.info("Initializing BitsAndBytes 4-bit configuration...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=q_conf.get("load_in_4bit", True),
            bnb_4bit_quant_type=q_conf.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=q_conf.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_compute_dtype=self.compute_dtype
        )
        
        # 3. Load Base Model
        logger.info(f"Loading base model: {model_name}...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=trust_remote_code,
                use_cache=self.training_config["model"].get("use_cache", False)
            )
        except Exception as e:
            logger.warning(f"Could not load target model {model_name}. Fallback to Qwen/Qwen2.5-3B-Instruct for training: {e}")
            fallback_model = "Qwen/Qwen2.5-3B-Instruct"
            model = AutoModelForCausalLM.from_pretrained(
                fallback_model,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=trust_remote_code,
                use_cache=self.training_config["model"].get("use_cache", False)
            )
            model_name = fallback_model
            
        # 4. Prepare for kbit training
        if self.training_config["training"].get("gradient_checkpointing", True):
            model.gradient_checkpointing_enable()
            
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        
        # 5. Initialize PEFT/LoRA Config
        l_conf = self.lora_config.get("lora", {})
        peft_config = LoraConfig(
            r=l_conf.get("r", 64),
            lora_alpha=l_conf.get("lora_alpha", 128),
            target_modules=l_conf.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
            lora_dropout=l_conf.get("lora_dropout", 0.05),
            bias=l_conf.get("bias", "none"),
            task_type=l_conf.get("task_type", "CAUSAL_LM")
        )
        
        model = get_peft_model(model, peft_config)
        logger.info("LoRA Adapter weights configured successfully.")
        model.print_trainable_parameters()
        
        return model, tokenizer, peft_config

    def train(self, train_dataset, val_dataset):
        """Sets up and runs the training loop using SFTTrainer."""
        model, tokenizer, peft_config = self.prepare_model_and_tokenizer()
        
        t_args = self.training_config["training"]
        d_args = self.training_config["dataset"]
        
        # Setup TrainingArguments
        training_args = TrainingArguments(
            output_dir=t_args.get("output_dir", "./checkpoints"),
            num_train_epochs=t_args.get("num_train_epochs", 3),
            per_device_train_batch_size=t_args.get("per_device_train_batch_size", 4),
            per_device_eval_batch_size=t_args.get("per_device_eval_batch_size", 4),
            gradient_accumulation_steps=t_args.get("gradient_accumulation_steps", 4),
            learning_rate=float(t_args.get("learning_rate", 2e-4)),
            weight_decay=t_args.get("weight_decay", 0.01),
            adam_beta1=t_args.get("adam_beta1", 0.9),
            adam_beta2=t_args.get("adam_beta2", 0.999),
            adam_epsilon=float(t_args.get("adam_epsilon", 1e-8)),
            max_grad_norm=t_args.get("max_grad_norm", 1.0),
            lr_scheduler_type=t_args.get("lr_scheduler_type", "cosine"),
            warmup_ratio=t_args.get("warmup_ratio", 0.03),
            logging_dir=t_args.get("logging_dir", "./outputs/logs"),
            logging_steps=t_args.get("logging_steps", 10),
            save_strategy=t_args.get("save_strategy", "epoch"),
            evaluation_strategy=t_args.get("evaluation_strategy", "epoch"),
            save_total_limit=t_args.get("save_total_limit", 3),
            seed=t_args.get("seed", 42),
            bf16=self.bf16_supported,
            fp16=not self.bf16_supported,
            gradient_checkpointing=t_args.get("gradient_checkpointing", True),
            ddp_find_unused_parameters=t_args.get("ddp_find_unused_parameters", False),
            optim=t_args.get("optim", "paged_adamw_8bit"),
            report_to="tensorboard"
        )
        
        # Setup completion data collator if packing is disabled
        packing = d_args.get("packing", True)
        data_collator = None
        if not packing:
            # Mask user instructions during loss calculations for ChatML
            response_template = "<|im_start|>assistant\n"
            data_collator = DataCollatorForCompletionOnlyLM(
                response_template=response_template, 
                tokenizer=tokenizer
            )
            logger.info("Completion-only loss collator integrated successfully (packing disabled).")

        # Instantiate SFTTrainer
        logger.info("Initializing SFTTrainer...")
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=d_args.get("max_seq_length", 4096),
            tokenizer=tokenizer,
            args=training_args,
            packing=packing,
            data_collator=data_collator
        )
        
        # Check for checkpoint to resume from
        resume_checkpoint = None
        if os.path.exists(t_args.get("output_dir", "./checkpoints")):
            checkpoints = [
                os.path.join(t_args.get("output_dir", "./checkpoints"), d)
                for d in os.listdir(t_args.get("output_dir", "./checkpoints"))
                if d.startswith("checkpoint-")
            ]
            if checkpoints:
                resume_checkpoint = max(checkpoints, key=os.path.getmtime)
                logger.info(f"Found checkpoint: {resume_checkpoint}. Resuming training...")
                
        # Start training
        logger.info("Starting training loop...")
        trainer.train(resume_from_checkpoint=resume_checkpoint)
        
        # Save final adapter weights
        final_adapter_dir = os.path.join(t_args.get("output_dir", "./checkpoints"), "final_adapter")
        logger.info(f"Saving final adapter model to {final_adapter_dir}...")
        trainer.model.save_pretrained(final_adapter_dir)
        tokenizer.save_pretrained(final_adapter_dir)
        logger.info("Training complete.")
        
        return trainer
