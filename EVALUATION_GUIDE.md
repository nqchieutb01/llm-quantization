# Evaluation Guide

## Overview

The evaluation system now uses only `experiments.yaml` for configuration, making it simpler and more streamlined.

## Configuration File: experiments.yaml

The `experiments.yaml` file contains:
- Baseline model configuration
- Quantization experiment parameters
- Evaluation settings (tasks, few-shot, batch size, device)
- Model discovery settings

## Quick Start

### 1. Evaluate Baseline Model

```bash
python run_eval_configurable.py --baseline
```

### 2. Evaluate All Quantized Models

```bash
python run_eval_configurable.py --all
```

The script will auto-discover models in the `models/` directory.

### 3. Evaluate Both Baseline and All Quantized Models

```bash
python run_eval_configurable.py --baseline --all
```

### 4. Evaluate Specific Models

```bash
python run_eval_configurable.py --models models/model1 models/model2
```

### 5. Evaluate on Different Task

```bash
python run_eval_configurable.py --all --task lambada
```

### 6. Use Different GPU

```bash
python run_eval_configurable.py --all --device cuda:1
```

### 7. Dry Run (Preview Commands)

```bash
python run_eval_configurable.py --baseline --all --dry_run
```

## Configuration Examples

### Basic experiments.yaml

```yaml
model:
  baseline: "Qwen/Qwen2.5-0.5B-Instruct"

evaluation:
  tasks: ["gsm8k", "lambada"]  # Multiple tasks
  num_fewshot: 5
  limit: 250
  batch_size: 8
  device: "cuda:0"
  gpu_memory_utilization: 0.8

paths:
  models_dir: "models"
  results_dir: "result"

quantized_models:
  auto_discover: true  # Auto-find models in models_dir
```

### Explicit Model Specification

If you want to specify models explicitly instead of auto-discovery:

```yaml
quantized_models:
  auto_discover: false
  explicit:
    - "models/Qwen2.5-0.5B-Instruct-GPTQ-W4A16-G128_gsm8k2048"
    - "models/Qwen2.5-0.5B-Instruct-GPTQ-W8A16-G128_gsm8k2048"
```

## Output Structure

### Results Files

Each evaluation creates:
- JSON result file: `result/modelname_task_timestamp.json`
- Sample outputs: `result/samples_task_timestamp.jsonl`

### Summary File

After evaluation completes, a summary is saved:
- `result/eval_summary_timestamp.yaml`

Contains:
- All evaluation results
- Metrics extracted from each evaluation
- Status of each evaluation
- Links to output files

## Features

### 1. Auto-Discovery
Automatically finds quantized models in the `models/` directory by pattern matching:
- `*-W4A16-*`, `*-W8A16-*`, `*-W4A8-*`, `*-W8A8-*`
- `*-GPTQ-*`, `*-AWQ-*`

### 2. Results Table
Displays a formatted table with:
- Model name
- Task
- Status (✓ success, ✗ failed)
- Accuracy metrics
- Output file names

### 3. Error Handling
- `--continue_on_error`: Continue evaluating other models if one fails
- Default: stops on first error

### 4. Multi-Task Support
Evaluate models on multiple tasks in a single run:

```yaml
evaluation:
  tasks: ["gsm8k", "lambada", "hellaswag"]
```

## Complete Workflow Example

### 1. Quantize Models

```bash
# Quantize with GPTQ W4A16
python run_quantize_configurable.py --config config.yaml

# Or run multiple experiments
python run_experiments.py --config config.yaml --experiments experiments.yaml
```

### 2. Configure Evaluation

Edit `experiments.yaml`:
```yaml
evaluation:
  tasks: ["gsm8k"]
  num_fewshot: 5
  limit: 250
  device: "cuda:0"
```

### 3. Run Evaluation

```bash
# Evaluate everything
python run_eval_configurable.py --baseline --all

# Or evaluate on multiple tasks
python run_eval_configurable.py --baseline --all --task gsm8k
python run_eval_configurable.py --all --task lambada
```

### 4. Review Results

Check the summary file:
```bash
cat result/eval_summary_*.yaml
```

Or individual result files:
```bash
ls -lt result/*.json | head
```

## Command-Line Options

| Option | Description |
|--------|-------------|
| `--config PATH` | Path to experiments.yaml (default: experiments.yaml) |
| `--baseline` | Evaluate baseline model |
| `--all` | Evaluate all quantized models |
| `--models PATH [PATH ...]` | Evaluate specific model paths |
| `--task TASK` | Override task from config |
| `--device DEVICE` | Override device (e.g., cuda:1) |
| `--dry_run` | Print commands without executing |
| `--continue_on_error` | Continue on failures |

## Tips

1. **Start with dry run**: Use `--dry_run` to preview what will be evaluated
2. **Use multiple GPUs**: If you have multiple GPUs, run evaluations in parallel:
   ```bash
   # Terminal 1
   python run_eval_configurable.py --all --device cuda:0 &
   
   # Terminal 2
   python run_eval_configurable.py --baseline --device cuda:1 &
   ```
3. **Monitor progress**: The script prints detailed progress for each evaluation
4. **Check summaries**: Review `eval_summary_*.yaml` for quick overview of all results
5. **Compare models**: All results use consistent naming for easy comparison

## Troubleshooting

**No models found:**
- Check that models are in the `models_dir` specified in experiments.yaml
- Ensure model directories contain model files (.safetensors or .bin)
- Use `--models` to specify paths explicitly

**OOM errors:**
- Reduce `batch_size` in experiments.yaml
- Lower `gpu_memory_utilization`
- Use different GPU with `--device cuda:1`

**Evaluation hangs:**
- Check GPU availability: `nvidia-smi`
- Ensure lm-eval and vllm are installed correctly
- Try with smaller `limit` first

## Advanced Usage

### Evaluate Subset of Models

```bash
# Only evaluate W4A16 models
python run_eval_configurable.py --models models/*W4A16*
```

### Custom Evaluation Config

Create a separate config for different evaluation scenarios:

```bash
# Light evaluation
python run_eval_configurable.py --config experiments_light.yaml --all

# Full evaluation
python run_eval_configurable.py --config experiments_full.yaml --all
```

### Batch Processing

```bash
# Evaluate all models on multiple tasks
for task in gsm8k lambada hellaswag; do
  python run_eval_configurable.py --all --task $task
done
```

