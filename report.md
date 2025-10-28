# Project Proposal Report: Quantization LLM Initiative

## Overview
This report summarizes the proposed direction for the Quantization LLM project, drawing on the current notes in `plan.md`. The effort focuses on systematically benchmarking GPTQ and AWQ methods across multiple precision settings and datasets to identify Pareto-optimal trade-offs between accuracy and efficiency for small-scale Qwen models.

## Objectives
- Establish reproducible baselines for Qwen models (0.5B, 0.6B, 1.7B) across full-precision and quantized variants.
- Quantize weights to both 4-bit and 8-bit formats while exploring 8-bit and 16-bit activations.
- Evaluate the impact of calibration dataset choice and sample budgets (512–4096) on downstream performance for reasoning (GSM8K), language modeling (LAMBADA, WikiText), and dialogue (UltraChat).
- Capture efficiency metrics—storage footprint, throughput, time-to-first-token, and peak GPU memory—to complement accuracy measurements.

## Current Findings (from `plan.md`)
- **GSM8K (0.5B model)**: Baseline accuracy 34.8; RTN drops to 5.6. GPTQ W4A16 ranges from 24.0 to 29.6 as calibration samples increase, while AWQ W4A16 at 2048 samples yields 16.8.
- **WikiText / UltraChat**: Baseline perplexity 18.11; RTN improves to 24.57 perplexity (lower is better). GPTQ (2048 samples) records 20.69, and AWQ (2048) achieves 21.58.
- **LAMBADA**: Baseline accuracy 52.4 (perplexity 12.57). RTN falls sharply to 34.0 accuracy (26.96 perplexity). GPTQ and AWQ recover to 44.4/46.4 accuracy (19.23/16.89 perplexity respectively).
- Mixed dataset experiments (W4A16, W4A4, W8A8 floating/int) are deprioritized for now.

These observations highlight the sensitivity of quantization quality to calibration sample size and method, with GPTQ consistently outperforming AWQ in reasoning tasks, while both methods narrow the gap on language modeling benchmarks.

## Proposed Workstream
1. **Experiment Expansion**
   - Extend `experiments.yaml` to sweep GPTQ and AWQ across precision schemes (W4A16, W4A8, W8A8) and calibration sample counts (512, 1024, 2048, 4096) for GSM8K, LAMBADA, and UltraChat.
   - Leverage `run_experiments.py` to batch these configurations, ensuring logs capture method, scheme, dataset, and sample budget metadata.

2. **Calibration Strategy Analysis**
   - Compare per-dataset calibrations and investigate hybrid datasets (e.g., mixing GSM8K reasoning with UltraChat dialogue) to test robustness beyond single-domain calibration.
   - Evaluate SmoothQuant integration for stability on lower-bit activations, especially when extending to W4A8.

3. **Evaluation Pipeline**
   - Use `run_eval_configurable.py` to benchmark baselines and quantized checkpoints on GSM8K (exact match), LAMBADA (accuracy/perplexity), WikiText (perplexity), and optionally UltraChat dialogue metrics.
   - Standardize few-shot settings, sample limits, and seeds to ensure comparability with current baselines.

4. **Efficiency Benchmarking**
   - After each quantization run, record storage size, vLLM throughput (tokens/s), time-to-first-token, and peak GPU memory using a consistent batch of prompts.
   - Store results in structured artifacts (JSON/YAML) inside the `result/` directory for aggregation.

5. **Reporting and Visualization**
   - Update `test.ipynb` or create a dedicated notebook to visualize accuracy vs. bit-width vs. calibration budget, highlighting best-performing configurations and regressions (e.g., RTN degradation).
   - Prepare final deliverables summarizing recommended quantization recipes per task and model scale, including efficiency trade-offs.

## Risks and Considerations
- Significant accuracy degradation observed with RTN suggests it should be avoided unless paired with additional calibration or smoothing techniques.
- Larger calibration datasets improve GPTQ performance on GSM8K but incur higher runtime; need to balance cost vs. gain.
- Evaluations require GPU availability; plan scheduling to prevent contention with ongoing quantization jobs.
- Comprehensive efficiency metrics may require custom benchmarking scripts beyond `lm-eval`; factor in development time.

## Next Steps
1. Finalize experiment matrix and begin batch quantization runs using the configurable scripts.
2. Automate evaluation passes for completed checkpoints and collect metrics into centralized summaries.
3. Draft comparative analysis highlighting optimal settings and outstanding gaps, setting the stage for follow-up experimentation (e.g., mixed-dataset calibration, SmoothQuant tuning).

This structured plan ensures coverage of the priority configurations outlined in `plan.md` while building the infrastructure for scalable evaluation and decision-making.
