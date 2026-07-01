# Performance profiling and hardware benchmarking script

import os
import time
import json
import torch
import psutil
import argparse
from tabulate import tabulate
from src.utils import setup_logging, load_yaml, check_bf16_support
from src.constants import TRAINING_CONFIG_PATH
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = setup_logging()

def get_memory_usage():
    """Returns CPU memory usage in MB and GPU memory usage if available."""
    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / (1024 * 1024) # MB
    
    gpu_mem = 0
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / (1024 * 1024) # MB
        
    return cpu_mem, gpu_mem

def main():
    parser = argparse.ArgumentParser(description="Profile model latency, throughput, and hardware memory consumption")
    parser.add_argument("--model_path", type=str, default=None, help="Path to model directory")
    parser.add_argument("--train_config", type=str, default=TRAINING_CONFIG_PATH, help="Path to training config YAML")
    parser.add_argument("--num_runs", type=int, default=5, help="Number of times to run benchmark for average metrics")
    args = parser.parse_args()
    
    # 1. Choose Model Path
    model_path = args.model_path
    if not model_path:
        train_conf = load_yaml(args.train_config)
        model_path = train_conf["model"]["base_model_name"]
        
    logger.info(f"Setting up benchmark for: {model_path}...")
    
    # Check compute capability
    bf16_supported = check_bf16_support()
    compute_dtype = torch.bfloat16 if bf16_supported else torch.float16
    
    # Track model load memory
    cpu_before, gpu_before = get_memory_usage()
    start_load = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=compute_dtype,
            device_map="auto"
        )
    except Exception as e:
        logger.warning(f"Could not load {model_path}. Benchmarking fallback model: {e}")
        fallback_model = "Qwen/Qwen2.5-3B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(
            fallback_model,
            torch_dtype=compute_dtype,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(fallback_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
    load_time = time.time() - start_load
    cpu_after, gpu_after = get_memory_usage()
    
    logger.info(f"Model loaded in {load_time:.2f} seconds.")
    logger.info(f"GPU RAM growth: {gpu_after - gpu_before:.1f} MB | CPU RAM growth: {cpu_after - cpu_before:.1f} MB")
    
    # Set mock candidate input
    cip = {
        "candidate_id": "C123",
        "rank": 7,
        "score": 0.923,
        "technical_fit": 0.95,
        "career_evidence": 0.91,
        "behavior": 0.88,
        "integrity": 1.0,
        "confidence": 0.97,
        "positive_signals": ["Production Retrieval", "Python", "Vector Database"],
        "negative_signals": [],
        "jd_coverage": {
            "must": ["Python", "Retrieval"],
            "preferred": ["Embeddings"],
            "missing": ["Evaluation Metrics"]
        },
        "feature_contributions": {"career": 50, "technical": 50}
    }
    
    messages = [
        {"role": "system", "content": "You are an experienced senior recruiter. Only explain supplied evidence."},
        {"role": "user", "content": json.dumps(cip)}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Warmup
    logger.info("Running warmup run...")
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        
    latencies = []
    throughputs = []
    tokens_count_list = []
    ttfts = []  # Time to first token
    
    logger.info(f"Starting benchmark loop of {args.num_runs} runs...")
    
    for r in range(args.num_runs):
        start_run = time.time()
        
        # We simulate TTFT by intercepting logits or basic single token generation
        with torch.no_grad():
            ttft_start = time.time()
            _ = model(**{k: v[:, :1] for k, v in inputs.items()})
            ttft = time.time() - ttft_start
            ttfts.append(ttft)
            
            # Full generation
            gen_start = time.time()
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.2,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
            gen_duration = time.time() - gen_start
            
        total_duration = time.time() - start_run
        
        input_len = inputs.input_ids.shape[1]
        output_len = outputs[0].shape[0] - input_len
        tokens_count_list.append(output_len)
        
        t_sec = output_len / gen_duration if gen_duration > 0 else 0
        throughputs.append(t_sec)
        latencies.append(gen_duration)
        
        logger.info(f"Run {r+1}: Generated {output_len} tokens in {gen_duration:.2f}s ({t_sec:.1f} tok/sec) | TTFT: {ttft:.3f}s")
        
    cpu_end, gpu_end = get_memory_usage()
    
    # Compute summary tables
    summary_data = [
        ["Model Load Time", f"{load_time:.2f} seconds"],
        ["Base Model GPU RAM", f"{gpu_after:.1f} MB"],
        ["Inference Peak GPU RAM", f"{gpu_end:.1f} MB"],
        ["System CPU RAM", f"{cpu_end:.1f} MB"],
        ["Average Latency (Generation)", f"{sum(latencies)/len(latencies):.2f} seconds"],
        ["Average TTFT (First Token)", f"{sum(ttfts)/len(ttfts):.3f} seconds"],
        ["Average Throughput", f"{sum(throughputs)/len(throughputs):.1f} tokens/sec"],
        ["Average Output length", f"{sum(tokens_count_list)/len(tokens_count_list):.1f} tokens"]
    ]
    
    print("\n" + "="*50)
    print("BENCHMARK PERFORMANCE REPORT")
    print("="*50)
    print(tabulate(summary_data, headers=["Metric", "Result"], tablefmt="grid"))
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
