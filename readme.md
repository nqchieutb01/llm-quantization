# LLM Quantization and Evaluation

A flexible system for quantizing and evaluating Large Language Models using GPTQ and AWQ methods.

## Features

- **Multiple Quantization Methods**: RTN, GPTQ, AWQ, GPTQ
- **Flexible Weight/Activation Bits**: W4A16, W8A8
- **Multiple Calibration Datasets**: GSM8K, LAMBADA, UltraChat
- **SmoothQuant Support**: Optional smoothing for better quantization
- **YAML Configuration**: Easy-to-manage configuration files
- **CLI Overrides**: Command-line arguments override config values
- **Automated Evaluation**: Built-in evaluation pipeline

## Installation
**Environment for RTN, AWQ, GPTQ**
```bash
pip install -r requirements.txt
pip install lm-eval[vllm]  # For evaluation
```
**Enviroment for GPTAQ**: Follow instruction in the original Repo https://github.com/ModelCloud/GPTQModel

## Quantized models
You can access and download all (almost) quantized models in Huggingface https://huggingface.co/collections/chieunq/qwen3-quantized-variants 

## System demo 
More details about system prompt and few-shot prompts can be seen in `demo.jsonl`
```
Input: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?

Output: The number of eggs Janet uses for breakfast is 3 * 1 = 3 eggs.\nThe number of eggs Janet uses for muffins is 4 * 1 = 4 eggs.\nThe total number of eggs used is 3 + 4 = 7 eggs.\nThe number of eggs left is 16 - 7 = 9 eggs.\nShe sells 9 eggs at the market for $2 each, so she makes 9 * 2 = <<9*2=18>>18 dollars.\n#### 18
```

## Quick Start

### 1. Quantization 

**For GPTQ, AWQ, RTN:**
```bash
python scripts/run_quantize_configurable.py --config config/config_quantize.yaml
```
**For GPTAQ**
```bash
python scripts/gptaq.py --model-id Qwen/Qwen3-1.7B --dataset-name openai/gsm8k --subset main --num-samples 2048
```

**For RTN**

```bash
python rtn_quantize.py
```

### 2. Evaluation

**Evaluate baseline model:**
```bash
python scripts/run_eval_configurable.py --baseline
```

**Evaluate specific quantized models:**
```bash
python scripts/run_eval_configurable.py --quantized ./model1 ./model2
```

**Evaluate all quantized models:**
```bash
python scripts/run_eval_configurable.py --all
```

## Configuration

### config/config_quantize.yaml Structure

```yaml
model:
  model_id: "Qwen/Qwen3-0.6B"
  torch_dtype: "auto"

quantization:
  method: "gptq"  # gptq or awq
  scheme: "W4A16"  # W4A16, W8A16, W4A8, W8A8
  group_size: 128
  targets: "Linear"
  ignore: ["lm_head"]
  use_smoothquant: false
  smoothing_strength: 0.8

dataset:
  calibration:
    name: "openai/gsm8k"
    subset: "main"
    num_samples: 1024
    max_seq_length: 2048
    seed: 42
  evaluation:
    name: "gsm8k"
    num_fewshot: 5
    limit: 250
    batch_size: 8

training:
  device: "cuda:0"
  gpu_memory_utilization: 0.8

output:
  save_dir: null  # auto-generated if null
  save_compressed: true
  output_path: "result"
  log_dir: "sparse_logs"
```

## Examples

### Example 1: GPTQ W4A16 on GSM8K

```bash
python scripts/run_quantize_configurable.py \
  --method gptq \
  --scheme W4A16 \
  --dataset "openai/gsm8k" \
  --num_samples 1024
```

### Example 2: AWQ W8A16 on UltraChat

```bash
python scripts/run_quantize_configurable.py \
  --method awq \
  --scheme W8A16 \
  --dataset "HuggingFaceH4/ultrachat_200k" \
  --num_samples 512
```


### Example 3: Evaluate Multiple Models

```bash
# Evaluate baseline
python scripts/run_eval_configurable.py --baseline

# Evaluate all quantized models
python scripts/run_eval_configurable.py --all

# Evaluate on LAMBADA instead of GSM8K
python scripts/run_eval_configurable.py --all --task lambada
```

## Command-Line Arguments

### run_quantize_configurable.py

- `--config`: Path to configuration file (default: config.yaml)
- `--model_id`: Model ID to quantize
- `--method`: Quantization method (gptq, awq)
- `--scheme`: Quantization scheme (W4A16, W8A16, W4A8, W8A8)
- `--group_size`: Group size for quantization
- `--dataset`: Calibration dataset name
- `--num_samples`: Number of calibration samples
- `--max_seq_length`: Maximum sequence length
- `--output_dir`: Output directory
- `--use_smoothquant`: Enable SmoothQuant

### run_eval_configurable.py

- `--config`: Path to configuration file (default: config.yaml)
- `--baseline`: Evaluate baseline model
- `--quantized`: Paths to quantized models to evaluate
- `--all`: Evaluate all quantized models in current directory
- `--device`: Device to use (e.g., cuda:0, cuda:1)
- `--task`: Evaluation task (overrides config)
- `--dry_run`: Print commands without executing

## Supported Datasets

### Calibration
- `openai/gsm8k`: Grade school math problems
- `lambada`: Language modeling dataset
- `HuggingFaceH4/ultrachat_200k`: Multi-turn conversations

### Evaluation
- `gsm8k`: Math reasoning
- `lambada`: Next word prediction

### Efficiency metrics
We borrow code from https://github.com/gjgjos/vllm_benchmark_serving to measure efficiency (time-to-first-token (TTFT), time-per-output-token (TPOT) and throughput).
```bash
vllm serve Qwen/Qwen3-0.6B --gpu-memory-utilization 0.8  # Deploy model
cd vllm_benchmark_serving
python3 run_sweep.py   # Run benchmark
python3 aggregate_result.py # Aggregate results
```