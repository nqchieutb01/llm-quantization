#!/usr/bin/env python3
"""
Configurable LLM Evaluation Script
Evaluates baseline and quantized models using lm-evaluation-harness.
Uses only experiments.yaml for configuration.
"""

import argparse
import os
import yaml
import subprocess
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional


def load_config(config_path: str = "experiments.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def discover_quantized_models(models_dir: str) -> List[Path]:
    """Discover quantized models in the models directory."""
    models_path = Path(models_dir)
    if not models_path.exists():
        return []
    
    # Look for directories that might be quantized models
    model_patterns = ['*-W4A16-*', '*-W8A16-*', '*-W4A8-*', '*-W8A8-*', 
                     '*-GPTQ-*', '*-AWQ-*']
    quantized_dirs = []
    for pattern in model_patterns:
        quantized_dirs.extend(models_path.glob(pattern))
    
    # Filter to only directories that contain model files
    valid_models = []
    for model_dir in quantized_dirs:
        if model_dir.is_dir():
            # Check if it has model files
            has_model = any([
                (model_dir / "model.safetensors").exists(),
                (model_dir / "pytorch_model.bin").exists(),
                list(model_dir.glob("*.safetensors")),
            ])
            if has_model:
                valid_models.append(model_dir)
    
    return sorted(valid_models)


def build_eval_command(
    model_path: str,
    task: str,
    config: Dict[str, Any],
    output_name: str,
) -> tuple[str, str]:
    """Build lm_eval command based on configuration."""
    eval_config = config['evaluation']
    
    num_fewshot = eval_config['num_fewshot']
    limit = eval_config.get('limit', 250)
    batch_size = eval_config['batch_size']
    gpu_util = eval_config['gpu_memory_utilization']
    device = eval_config['device']
    
    results_dir = config['paths']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%d-%m-%M-%S")
    output_path = os.path.join(results_dir, f"{output_name}_{task}_{timestamp}.json")

    cmd_parts = [
        "lm_eval",
        "--model vllm",
        f"--model_args pretrained=\"{model_path}\",gpu_memory_utilization={gpu_util},add_bos_token=true",
        f"--tasks {task}",
        f"--num_fewshot {num_fewshot}",
        f"--batch_size {batch_size}",
        f"--device '{device}'",
        f"--output_path {output_path}",
        "--write_out",
        # "--log_samples",
    ]
    
    if limit:
        cmd_parts.insert(5, f"--limit {limit}")
    
    cmd = " ".join(cmd_parts)
    
    return cmd, output_path


def evaluate_model(
    model_path: str,
    model_name: str,
    task: str,
    config: Dict[str, Any],
    dry_run: bool = False
) -> Dict[str, Any]:
    """Evaluate a model and return results."""
    cmd, output_path = build_eval_command(model_path, task, config, model_name)
    
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_name}")
    print(f"Model Path: {model_path}")
    print(f"Task: {task}")
    print(f"{'='*80}")
    print(f"Command: {cmd}")
    print(f"Output: {output_path}")
    
    if dry_run:
        print("[DRY RUN] - Skipping execution\n")
        return {
            "model": model_path,
            "model_name": model_name,
            "task": task,
            "command": cmd,
            "output": output_path,
            "status": "dry_run"
        }
    
    try:
        print("\nExecuting evaluation...")
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        
        # Try to read and extract key metrics
        metrics = None
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r') as f:
                    eval_results = json.load(f)
                    # Extract main metric
                    if 'results' in eval_results and task in eval_results['results']:
                        metrics = eval_results['results'][task]
            except Exception as e:
                print(f"Warning: Could not parse results: {e}")
        
        return {
            "model": model_path,
            "model_name": model_name,
            "task": task,
            "output": output_path,
            "status": "success",
            "metrics": metrics,
        }
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error evaluating {model_name}:")
        print(e.stderr)
        return {
            "model": model_path,
            "model_name": model_name,
            "task": task,
            "output": output_path,
            "status": "failed",
            "error": str(e.stderr)
        }


def print_results_table(results: List[Dict[str, Any]]):
    """Print a formatted table of results."""
    if not results:
        return
    
    print("\n" + "="*120)
    print("EVALUATION RESULTS SUMMARY")
    print("="*120)
    
    # Group by task
    tasks = list(set(r['task'] for r in results))
    
    for task in tasks:
        print(f"\nTask: {task}")
        print("-"*120)
        print(f"{'Model':<60} {'Status':<15} {'Accuracy':<15} {'Output File':<30}")
        print("-"*120)
        
        task_results = [r for r in results if r['task'] == task]
        for result in task_results:
            model_name = result['model_name'][:58]
            status = result['status']
            
            # Extract accuracy metric if available
            accuracy = "N/A"
            if result.get('metrics'):
                metrics = result['metrics']
                # Try different metric keys
                for key in ['acc', 'acc_norm', 'exact_match', 'accuracy']:
                    if key in metrics:
                        accuracy = f"{metrics[key]*100:.2f}%" if metrics[key] < 1 else f"{metrics[key]:.2f}%"
                        break
            
            output_file = Path(result['output']).name if 'output' in result else ""
            
            status_symbol = "✓" if status == "success" else ("⊘" if status == "dry_run" else "✗")
            print(f"{model_name:<60} {status_symbol} {status:<13} {accuracy:<15} {output_file:<30}")
        
        print("-"*120)


def main():
    parser = argparse.ArgumentParser(
        description="Configurable LLM Evaluation (uses experiments.yaml)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate baseline model
  python run_eval_configurable.py --baseline
  
  # Evaluate all quantized models
  python run_eval_configurable.py --all
  
  # Evaluate baseline and all quantized models
  python run_eval_configurable.py --baseline --all
  
  # Evaluate specific models
  python run_eval_configurable.py --models models/model1 models/model2
  
  # Evaluate on specific task
  python run_eval_configurable.py --all --task lambada
  
  # Dry run (print commands without executing)
  python run_eval_configurable.py --all --dry_run
        """
    )
    
    parser.add_argument('--config', type=str, default='experiments.yaml',
                        help='Path to experiments configuration file (default: experiments.yaml)')
    parser.add_argument('--baseline', action='store_true',
                        help='Evaluate baseline model')
    parser.add_argument('--all', action='store_true',
                        help='Evaluate all quantized models in models directory')
    parser.add_argument('--models', type=str, nargs='+',
                        help='Specific model paths to evaluate')
    parser.add_argument('--task', type=str,
                        help='Specific task to evaluate (overrides config)')
    parser.add_argument('--device', type=str,
                        help='Device to use (overrides config, e.g., cuda:0, cuda:1)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print commands without executing')
    parser.add_argument('--continue_on_error', action='store_true',
                        help='Continue evaluating even if a model fails')
    parser.add_argument('--num_fewshot', type=int, default=5,
                        help='Number of few-shot examples to use')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    config['evaluation']['num_fewshot'] = args.num_fewshot
    
    # Override device if specified
    if args.device:
        config['evaluation']['device'] = args.device
    
    # Determine tasks to evaluate
    if args.task:
        tasks = [args.task]
    else:
        tasks = config['evaluation']['tasks']
    
    print("="*120)
    print("LLM EVALUATION CONFIGURATION")
    print("="*120)
    print(f"Config file: {args.config}")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Few-shot: {args.num_fewshot}")
    print(f"Limit: {config['evaluation'].get('limit', 'all samples')}")
    print(f"Device: {config['evaluation']['device']}")
    print(f"Batch size: {config['evaluation']['batch_size']}")
    print(f"Dry run: {args.dry_run}")
    print("="*120)
    
    # Collect models to evaluate
    models_to_eval = []
    
    # Add baseline if requested
    if args.baseline:
        baseline_model = config['model']['baseline']
        models_to_eval.append({
            'path': baseline_model,
            'name': 'baseline'
        })
        print(f"\n✓ Added baseline model: {baseline_model}")
    
    # Add specific models if provided
    if args.models:
        for model_path in args.models:
            model_name = Path(model_path).name
            models_to_eval.append({
                'path': model_path,
                'name': model_name
            })
        print(f"\n✓ Added {len(args.models)} specific model(s)")
    
    # Add all quantized models if requested
    if args.all:
        models_dir = config['paths']['models_dir']
        print(config)
        # import pdb; pdb.set_trace();
        # Check if should auto-discover or use explicit list
        if config.get('quantized_models', {}).get('auto_discover', True):
            discovered = discover_quantized_models(models_dir)
            if discovered:
                for model_path in discovered:
                    models_to_eval.append({
                        'path': str(model_path),
                        'name': model_path.name
                    })
                print(f"\n✓ Auto-discovered {len(discovered)} quantized model(s) in {models_dir}/")
                for m in discovered:
                    print(f"  - {m.name}")
            else:
                print(f"\n⚠ No quantized models found in {models_dir}/")
        elif 'explicit' in config.get('quantized_models', {}):
            explicit_models = config['quantized_models']['explicit']
            for model_path in explicit_models:
                model_name = Path(model_path).name
                models_to_eval.append({
                    'path': model_path,
                    'name': model_name
                })
            print(f"\n✓ Added {len(explicit_models)} explicit model(s) from config")
    
    # Check if any models to evaluate
    if not models_to_eval:
        print("\n⚠ No models specified for evaluation!")
        print("\nUsage examples:")
        print("  python run_eval_configurable.py --baseline")
        print("  python run_eval_configurable.py --all")
        print("  python run_eval_configurable.py --baseline --all")
        print("  python run_eval_configurable.py --models path/to/model1 path/to/model2")
        return
    
    print(f"\n{'='*120}")
    print(f"STARTING EVALUATION: {len(models_to_eval)} model(s) × {len(tasks)} task(s) = {len(models_to_eval) * len(tasks)} evaluation(s)")
    print(f"{'='*120}")
    
    # Run evaluations
    all_results = []
    
    for model_info in models_to_eval:
        for task in tasks:
            result = evaluate_model(
                model_path=model_info['path'],
                model_name=model_info['name'],
                task=task,
                config=config,
                dry_run=args.dry_run
            )
            all_results.append(result)
            
            # Stop if failed and not continue_on_error
            if result['status'] == 'failed' and not args.continue_on_error and not args.dry_run:
                print(f"\n⚠ Evaluation failed. Use --continue_on_error to continue despite failures.")
                break
        
        # Break outer loop too if failed
        if all_results and all_results[-1]['status'] == 'failed' and not args.continue_on_error and not args.dry_run:
            break
    
    # Print summary table
    print_results_table(all_results)
    
    # Save summary to file
    if not args.dry_run:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        summary_file = os.path.join(config['paths']['results_dir'], f"eval_summary_{timestamp}.yaml")
        
        # Prepare summary data
        summary_data = {
            'timestamp': timestamp,
            'config': args.config,
            'tasks': tasks,
            'num_models': len(models_to_eval),
            'num_evaluations': len(all_results),
            'results': [
                {
                    'model_name': r['model_name'],
                    'task': r['task'],
                    'status': r['status'],
                    'metrics': r.get('metrics'),
                    'output_file': r.get('output'),
                }
                for r in all_results
            ]
        }
        
        with open(summary_file, 'w') as f:
            yaml.dump(summary_data, f, default_flow_style=False, sort_keys=False)
        
        print(f"\n📊 Evaluation summary saved to: {summary_file}")
    
    # Final statistics
    success_count = sum(1 for r in all_results if r['status'] == 'success')
    failed_count = sum(1 for r in all_results if r['status'] == 'failed')
    
    print(f"\n{'='*120}")
    print("FINAL STATISTICS")
    print(f"{'='*120}")
    print(f"Total evaluations: {len(all_results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"{'='*120}\n")


if __name__ == "__main__":
    main()
