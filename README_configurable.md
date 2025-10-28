# Configurable LLM Quantization and Evaluation

A flexible system for quantizing and evaluating Large Language Models using GPTQ and AWQ methods.

## Features

- **Multiple Quantization Methods**: GPTQ, AWQ
- **Flexible Weight/Activation Bits**: W4A16, W8A16, W4A8, W8A8
- **Multiple Calibration Datasets**: GSM8K, LAMBADA, UltraChat
- **SmoothQuant Support**: Optional smoothing for better quantization
- **YAML Configuration**: Easy-to-manage configuration files
- **CLI Overrides**: Command-line arguments override config values
- **Automated Evaluation**: Built-in evaluation pipeline

## Installation

```bash
pip install llmcompressor transformers datasets pyyaml
pip install lm-eval[vllm]  # For evaluation
```

## Quick Start

### 1. Quantization

**Using custom config file:**
```bash
python run_quantize_configurable.py --config my_config.yaml
```

### 2. Evaluation

**Evaluate baseline model:**
```bash
python run_eval_configurable.py --baseline
```

**Evaluate specific quantized models:**
```bash
python run_eval_configurable.py --quantized ./model1 ./model2
```

**Evaluate all quantized models:**
```bash
python run_eval_configurable.py --all
```

**Evaluate with custom device:**
```bash
python run_eval_configurable.py --baseline --quantized ./model1 --device cuda:1
```

## Configuration

### config.yaml Structure

```yaml
model:
  model_id: "Qwen/Qwen2.5-0.5B-Instruct"
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
python run_quantize_configurable.py \
  --method gptq \
  --scheme W4A16 \
  --dataset "openai/gsm8k" \
  --num_samples 1024
```

### Example 2: AWQ W8A16 on UltraChat

```bash
python run_quantize_configurable.py \
  --method awq \
  --scheme W8A16 \
  --dataset "HuggingFaceH4/ultrachat_200k" \
  --num_samples 512
```

### Example 3: GPTQ with SmoothQuant

```bash
python run_quantize_configurable.py \
  --method gptq \
  --scheme W4A16 \
  --use_smoothquant
```

### Example 4: Evaluate Multiple Models

```bash
# Evaluate baseline
python run_eval_configurable.py --baseline

# Evaluate all quantized models
python run_eval_configurable.py --all

# Evaluate on LAMBADA instead of GSM8K
python run_eval_configurable.py --all --task lambada
```

### Example 5: Complete Pipeline

```bash
# 1. Quantize with GPTQ W4A16
python run_quantize_configurable.py \
  --model_id "Qwen/Qwen3-0.6B"\
  --method gptq \
  --scheme W4A16 \
  --dataset "openai/gsm8k" \
  --num_samples 2048

# 2. Quantize with AWQ W8A16
python run_quantize_configurable.py \
  --method awq \
  --scheme W8A16 \
  --dataset "openai/gsm8k" \
  --num_samples 1024

# 3. Evaluate all models
python run_eval_configurable.py --baseline --all
```

## Output Structure

```
.
├── config.yaml
├── run_quantize_configurable.py
├── run_eval_configurable.py
├── result/
│   ├── baseline_2025-10-10T10-00-00.json
│   ├── model1_2025-10-10T10-30-00.json
│   └── samples_*.jsonl
├── sparse_logs/
│   └── oneshot_*.log
└── Qwen2.5-0.5B-Instruct-GPTQ-W4A16-G128_gsm8k1024/
    ├── model.safetensors
    ├── config.json
    ├── tokenizer.json
    └── quantization_config.yaml
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

## License

See parent repository license.

