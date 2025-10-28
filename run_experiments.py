#!/usr/bin/env python3
"""
Batch Experiment Runner
Runs multiple quantization experiments with different configurations.
"""

import argparse
import os
import yaml
import subprocess
import itertools
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path


def create_experiment_configs(base_config: Dict[str, Any], experiments: Dict[str, List]) -> List[Dict[str, Any]]:
    """
    Create experiment configurations by combining different parameter values.
    
    Args:
        base_config: Base configuration to modify
        experiments: Dictionary with parameter names and their values to try
    
    Returns:
        List of experiment configurations
    """
    configs = []
    
    # Get all combinations
    keys = list(experiments.keys())
    values = list(experiments.values())
    
    for combination in itertools.product(*values):
        config = yaml.safe_load(yaml.dump(base_config))  # Deep copy
        
        for key, value in zip(keys, combination):
            # Handle nested keys with dot notation (e.g., 'quantization.method')
            parts = key.split('.')
            current = config
            for part in parts[:-1]:
                current = current[part]
            current[parts[-1]] = value
        
        configs.append(config)
    
    return configs


def run_quantization_experiment(config: Dict[str, Any], experiment_id: int, dry_run: bool = False) -> Dict[str, Any]:
    """Run a single quantization experiment."""
    # Save temporary config
    temp_config_path = f"temp_config_{experiment_id}.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    cmd = f"python run_quantize_configurable.py --config {temp_config_path}"
    
    method = config['quantization']['method']
    scheme = config['quantization']['scheme']
    dataset = config['dataset']['calibration']['name'].split('/')[-1]
    
    print(f"\n{'='*80}")
    print(f"Experiment {experiment_id}: {method.upper()} {scheme} on {dataset}")
    print(f"{'='*80}")
    
    if dry_run:
        print(f"[DRY RUN] Would execute: {cmd}")
        os.remove(temp_config_path)
        return {"experiment_id": experiment_id, "status": "dry_run", "config": config}
    
    try:
        print(f"Executing: {cmd}")
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        
        # Clean up temp config
        os.remove(temp_config_path)
        
        return {
            "experiment_id": experiment_id,
            "status": "success",
            "config": config,
            "stdout": result.stdout
        }
    except subprocess.CalledProcessError as e:
        print(f"Error in experiment {experiment_id}:")
        print(e.stderr)
        
        # Clean up temp config
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
        
        return {
            "experiment_id": experiment_id,
            "status": "failed",
            "config": config,
            "error": e.stderr
        }


def main():
    parser = argparse.ArgumentParser(description="Batch Experiment Runner for LLM Quantization")
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Base configuration file')
    parser.add_argument('--experiments', type=str, default='experiments.yaml',
                        help='Experiments specification file')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print experiment plan without executing')
    parser.add_argument('--continue_on_error', action='store_true',
                        help='Continue running experiments even if one fails')
    
    args = parser.parse_args()
    
    # Load base configuration
    print("Loading base configuration...")
    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Load experiment specifications
    print(f"Loading experiment specifications from {args.experiments}...")
    
    if not os.path.exists(args.experiments):
        print(f"Experiments file not found: {args.experiments}")
        print("Creating example experiments.yaml...")
        
        # Create example experiments file
        example_experiments = {
            "experiments": {
                "quantization.method": ["gptq", "awq"],
                "quantization.scheme": ["W4A16", "W8A16"],
                "dataset.calibration.name": ["openai/gsm8k"],
            }
        }
        with open('experiments.yaml', 'w') as f:
            yaml.dump(example_experiments, f, default_flow_style=False)
        
        print("\nExample experiments.yaml created. Please review and modify as needed.")
        print("\nExample content:")
        print(yaml.dump(example_experiments, default_flow_style=False))
        return
    
    with open(args.experiments, 'r') as f:
        experiments_spec = yaml.safe_load(f)
    
    # Create experiment configurations
    experiments = experiments_spec.get('experiments', {})
    configs = create_experiment_configs(base_config, experiments)
    
    print(f"\n{'='*80}")
    print(f"Experiment Plan: {len(configs)} experiments")
    print(f"{'='*80}")
    
    for i, config in enumerate(configs, 1):
        method = config['quantization']['method']
        scheme = config['quantization']['scheme']
        dataset = config['dataset']['calibration']['name'].split('/')[-1]
        num_samples = config['dataset']['calibration']['num_samples']
        print(f"{i}. {method.upper()} {scheme} - {dataset} ({num_samples} samples)")
    
    if args.dry_run:
        print("\n[DRY RUN MODE] - No experiments will be executed")
    
    # Run experiments
    results = []
    for i, config in enumerate(configs, 1):
        result = run_quantization_experiment(config, i, args.dry_run)
        results.append(result)
        
        if result['status'] == 'failed' and not args.continue_on_error:
            print(f"\nExperiment {i} failed. Stopping...")
            break
    
    # Print summary
    print(f"\n{'='*80}")
    print("Experiment Summary")
    print(f"{'='*80}")
    
    success_count = sum(1 for r in results if r['status'] == 'success')
    failed_count = sum(1 for r in results if r['status'] == 'failed')
    
    for result in results:
        status_symbol = "✓" if result['status'] == 'success' else "✗"
        config = result['config']
        method = config['quantization']['method']
        scheme = config['quantization']['scheme']
        dataset = config['dataset']['calibration']['name'].split('/')[-1]
        
        print(f"{status_symbol} Experiment {result['experiment_id']}: {method.upper()} {scheme} - {dataset}")
    
    print(f"\nTotal: {len(results)} | Success: {success_count} | Failed: {failed_count}")
    
    # Save experiment results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_file = f"experiment_results_{timestamp}.yaml"
    with open(results_file, 'w') as f:
        # Remove stdout/stderr for cleaner file
        clean_results = []
        for r in results:
            clean_r = {
                "experiment_id": r["experiment_id"],
                "status": r["status"],
                "config": r["config"]
            }
            if 'error' in r:
                clean_r['error'] = r['error'][:500]  # Truncate long errors
            clean_results.append(clean_r)
        
        yaml.dump({"results": clean_results}, f, default_flow_style=False)
    
    print(f"\nExperiment results saved to: {results_file}")


if __name__ == "__main__":
    main()

