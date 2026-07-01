# Phase 2 Qwen3 Explanation Training Repository

This repository hosts the production-ready machine learning training pipeline for fine-tuning `Qwen/Qwen3-4B-Instruct` (using QLoRA) for Phase 2 of the Redrob AI Hiring Challenge.

## Architecture & Design Goals

The primary goal of Phase 2 is to translate a structured **Candidate Intelligence Package (CIP)** (outputted from the Phase 1 ranking system) into clear, natural, and recruiter-quality candidate descriptions. 

### Core Constraints
- **No Ranking Changes**: This model does not re-rank candidates; it only explains the current rankings.
- **Zero Hallucination**: The model is strictly bound to the supplied CIP facts. It must not invent skills, technologies, metrics, or company experiences.
- **Tone Alignment**: Supports 4 unique communication styles (`Professional`, `Recruiter`, `Concise`, and `Technical`).

---

### Run Command

python batch_process.py     --input_file ./data/input_candidates.csv     --output_file ./outputs/reasoning_report.xlsx

## Folder Structure

```
phase2-qwen3-training/
├── README.md
├── requirements.txt
├── .gitignore
├── prepare_dataset.py
├── generate_synthetic_dataset.py
├── train.py
├── evaluate.py
├── inference.py
├── merge_lora.py
├── convert_to_gguf.py
├── benchmark.py
│
├── configs/
│   ├── training.yaml
│   ├── lora.yaml
│   ├── accelerate.yaml
│   ├── dataset.yaml
│   └── inference.yaml
│
├── src/
│   ├── dataset.py
│   ├── prompts.py
│   ├── trainer.py
│   ├── evaluator.py
│   ├── metrics.py
│   ├── validator.py
│   ├── tokenizer.py
│   ├── inference_engine.py
│   ├── utils.py
│   └── constants.py
│
├── data/
│   ├── train.jsonl
│   ├── validation.jsonl
│   ├── test.jsonl
│   └── synthetic/
│
├── scripts/
│   ├── train.sh
│   ├── merge.sh
│   ├── evaluate.sh
│   ├── benchmark.sh
│   └── convert.sh
│
├── outputs/
└── checkpoints/
```

---

## Hardware Requirements

- **GPU**: A single GPU with at least **12 GB VRAM** (e.g., RTX 3060/4060, T4, A10G) is required for 4-bit QLoRA training using sequence length 4096. An **Ampere/Ada/Hopper GPU** (e.g., RTX 3090, A100, H100) is highly recommended to enable native **BF16 mixed-precision training**.
- **System RAM**: At least **16 GB RAM** (32 GB recommended to handle large tokenized datasets and cache pools).

---

## Installation

Ensure you have Python 3.10+ installed. Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Step-by-Step Execution Guide

### 1. Data Generation & Preparation

Generate 10,000+ synthetic candidate evaluations matching the 4 recruiter styles:

```bash
python generate_synthetic_dataset.py
```

Split the generated candidate database into `train`, `validation`, and `test` splits:

```bash
python prepare_dataset.py
```

### 2. Fine-Tuning (SFT / QLoRA)

Run model training using the Accelerate orchestrator:

```bash
./scripts/train.sh
```
*Note: Training state will save checkpoints every epoch. Resuming training is supported automatically.*

### 3. Model Evaluation

Execute ROUGE, BLEU, BERTScore, and groundedness/violation checks on the test dataset split:

```bash
./scripts/evaluate.sh
```
*Outputs a detailed evaluation dashboard report in `outputs/evaluation_report.md`.*

### 4. Weights Merge

Combine the LoRA adapter weights with the base model weights to produce a standalone model directory:

```bash
./scripts/merge.sh
```

### 5. Benchmark Performance

Profile CPU/GPU memory, token latencies, and generation throughput speeds:

```bash
./scripts/benchmark.sh
```

### 6. GGUF Conversion

Prepare the model for deployment via Ollama/llama.cpp:

```bash
./scripts/convert.sh
```
*This writes a custom `Modelfile` and outputs command instructions to compile GGUF weights.*

### 7. Grounded Inference

To run inference interactively on a Candidate Intelligence Package, execute:

```bash
python inference.py --style "Recruiter"
```

---

## Fact Verification & Groundedness Validator

The `GroundednessValidator` class in `src/validator.py` strictly verifies generated summaries against the input Candidate Intelligence Package facts. 

If any of the following occur, the generator flags a validation error and attempts a fallback regeneration:
- Mentions a technology or skill not in the CIP.
- Mentions a company name not present in candidate records.
- Invented numerical metrics (e.g. hallucinating score ratios or notice timelines).
