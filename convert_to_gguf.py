# GGUF converter helper and Ollama Modelfile builder

import os
import argparse
from src.utils import setup_logging

logger = setup_logging()

OLLAMA_MODELFILE_TEMPLATE = """# Ollama Modelfile for Phase 2 Explanation Model
FROM ./qwen3_phase2_q4_k_m.gguf

# Set generation hyper-parameters
PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER num_ctx 4096

# Set Recruiter System Prompt
SYSTEM \"\"\"
You are an experienced senior recruiter.
Never invent facts.
Never hallucinate.
Only explain supplied evidence.
Write the explanation in a recruiter style.
Adhere strictly to the provided Candidate Intelligence Package (CIP) facts.
\"\"\"
"""

def main():
    parser = argparse.ArgumentParser(description="Generate Ollama Modelfile and GGUF conversion instructions")
    parser.add_argument("--model_dir", type=str, default="./merged_model", help="Directory containing the merged model weights")
    parser.add_argument("--output_file", type=str, default="./Modelfile", help="Path to write the Ollama Modelfile")
    args = parser.parse_args()
    
    # 1. Write Ollama Modelfile
    logger.info(f"Generating Ollama Modelfile configuration at: {args.output_file}")
    with open(args.output_file, 'w') as f:
        f.write(OLLAMA_MODELFILE_TEMPLATE)
        
    # 2. Log out standard llama.cpp clone & convert commands
    logger.info("=== GGUF CONVERSION PROCESS STEPS ===")
    print("\nTo convert the merged PyTorch weights to GGUF, follow these steps:")
    print("1. Clone llama.cpp repository:")
    print("   git clone https://github.com/ggerganov/llama.cpp.git")
    print("2. Install requirements:")
    print("   pip install -r llama.cpp/requirements.txt")
    print("3. Run the convert script to output FP16 GGUF format:")
    print(f"   python llama.cpp/convert_hf_to_gguf.py {args.model_dir} --outfile ./qwen3_phase2_f16.gguf")
    print("4. Quantize the GGUF file to Q4_K_M (4-bit recommended):")
    print("   ./llama.cpp/llama-quantize ./qwen3_phase2_f16.gguf ./qwen3_phase2_q4_k_m.gguf Q4_K_M")
    print("5. Create Ollama Model:")
    print(f"   ollama create qwen3-phase2 -f {args.output_file}")
    print("6. Run the Ollama model:")
    print("   ollama run qwen3-phase2")
    print("======================================\n")

if __name__ == "__main__":
    main()
