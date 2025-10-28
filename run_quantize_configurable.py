#!/usr/bin/env python3
"""
Configurable LLM Quantization Script
Supports GPTQ and AWQ quantization methods with various weight/activation configurations.
"""

import argparse
import os
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def override_config(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Override config values with command-line arguments."""
    if args.model_id:
        config['model']['model_id'] = args.model_id
    if args.method:
        config['quantization']['method'] = args.method
    if args.scheme:
        config['quantization']['scheme'] = args.scheme
    if args.group_size:
        config['quantization']['group_size'] = args.group_size
    if args.dataset:
        config['dataset']['calibration']['name'] = args.dataset
    if args.num_samples:
        config['dataset']['calibration']['num_samples'] = args.num_samples
    if args.max_seq_length:
        config['dataset']['calibration']['max_seq_length'] = args.max_seq_length
    if args.output_dir:
        config['output']['save_dir'] = args.output_dir
    if args.use_smoothquant:
        config['quantization']['use_smoothquant'] = True
    
    return config


def prepare_dataset(config: Dict[str, Any], tokenizer) -> Any:
    """Prepare and preprocess the calibration dataset."""
    dataset_name = config['dataset']['calibration']['name']
    subset = config['dataset']['calibration'].get('subset', None)
    num_samples = config['dataset']['calibration']['num_samples']
    max_seq_length = config['dataset']['calibration']['max_seq_length']
    seed = config['dataset']['calibration']['seed']
    
    print(f"Loading dataset: {dataset_name}")
    
    # Load dataset
    if 'ultrachat' in dataset_name.lower():
            ds = load_dataset(dataset_name, split = "train_sft[:2048]")
    elif 'gsm8k' in dataset_name.lower():
            ds = load_dataset(dataset_name, "main")
    else:
        ds = load_dataset(dataset_name, split = subset)
    
    # Select training split and shuffle
    if 'train' in ds:
        ds = ds['train']
    elif isinstance(ds, dict):
        ds = list(ds.values())[0]
    
    ds = ds.shuffle(seed=seed).select(range(min(num_samples, len(ds))))
    
    # Preprocess based on dataset type
    def preprocess_gsm8k(example):
        messages = [
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["answer"]},
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}
    
    def preprocess_ultrachat(example):
        # ultrachat has 'messages' field already
        if 'messages' in example:
            return {"text": tokenizer.apply_chat_template(example['messages'], tokenize=False)}
        else:
            return {"text": example.get('text', '')}
    
    def preprocess_lambada(example):
        # LAMBADA has 'text' field
        return {"text": example['text']}
    
    # Apply appropriate preprocessing
    if 'gsm8k' in dataset_name.lower():
        ds = ds.map(preprocess_gsm8k)
    elif 'ultrachat' in dataset_name.lower():
        ds = ds.map(preprocess_ultrachat)
    elif 'lambada' in dataset_name.lower():
        ds = ds.map(preprocess_lambada)
    else:
        # Generic preprocessing - assume 'text' field exists
        print(f"Using generic preprocessing for {dataset_name}")
    
    # Tokenize
    def tokenize(sample):
        return tokenizer(
            sample["text"], 
            padding=False, 
            max_length=max_seq_length, 
            truncation=True, 
            add_special_tokens=False
        )
    
    ds = ds.map(tokenize, remove_columns=ds.column_names)
    
    return ds


def create_quantization_recipe(config: Dict[str, Any]) -> list:
    """Create quantization recipe based on configuration."""
    recipe = []
    
    # Add SmoothQuant if enabled
    if config['quantization'].get('use_smoothquant', False):
        smoothing_strength = config['quantization'].get('smoothing_strength', 0.8)
        recipe.append(SmoothQuantModifier(smoothing_strength=smoothing_strength))
        print(f"Added SmoothQuant with strength {smoothing_strength}")
    
    # Add quantization method
    method = config['quantization']['method'].lower()
    scheme = config['quantization']['scheme']
    targets = config['quantization']['targets']
    ignore = config['quantization'].get('ignore', None)
    
    quantization_kwargs = {
        'scheme': scheme,
        'targets': targets,
        'ignore': ignore,
        # 'offload_device': torch.device("cpu"),
    }
    
    # Add group_size if specified in scheme
    if 'group_size' in config['quantization']:
        quantization_kwargs['group_size'] = config['quantization']['group_size']
    
    if method == 'gptq':
        recipe.append(GPTQModifier(**quantization_kwargs))
        print(f"Added GPTQ quantization: {scheme}")
    elif method == 'awq':
        recipe.append(AWQModifier(**quantization_kwargs))
        print(f"Added AWQ quantization: {scheme}")
    else:
        raise ValueError(f"Unknown quantization method: {method}")
    
    return recipe


def generate_output_dir(config: Dict[str, Any]) -> str:
    """Generate output directory name based on configuration."""
    if config['output']['save_dir']:
        return config['output']['save_dir']
    
    model_name = config['model']['model_id'].rstrip("/").split("/")[-1]
    method = config['quantization']['method'].upper()
    scheme = config['quantization']['scheme']
    group_size = config['quantization'].get('group_size', '')
    dataset_name = config['dataset']['calibration']['name'].split("/")[-1]
    num_samples = config['dataset']['calibration']['num_samples']
    
    dir_name = f"{model_name}-{method}-{scheme}"
    if group_size:
        dir_name += f"-G{group_size}"
    dir_name += f"_{dataset_name}{num_samples}"
    
    return "models/" + dir_name


def main():
    parser = argparse.ArgumentParser(description="Configurable LLM Quantization")
    parser.add_argument('--config', type=str, default='config.yaml', 
                        help='Path to configuration file')
    parser.add_argument('--model_id', type=str, help='Model ID to quantize')
    parser.add_argument('--method', type=str, choices=['gptq', 'awq'], 
                        help='Quantization method')
    parser.add_argument('--scheme', type=str, 
                        help='Quantization scheme (e.g., W4A16, W8A16)')
    parser.add_argument('--group_size', type=int, help='Group size for quantization')
    parser.add_argument('--dataset', type=str, 
                        help='Calibration dataset name')
    parser.add_argument('--num_samples', type=int, 
                        help='Number of calibration samples')
    parser.add_argument('--max_seq_length', type=int, 
                        help='Maximum sequence length')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--use_smoothquant', action='store_true', 
                        help='Enable SmoothQuant')
    
    args = parser.parse_args()
    
    # Load and override configuration
    config = load_config(args.config)
    config = override_config(config, args)
    
    print("=" * 80)
    print("LLM Quantization Configuration")
    print("=" * 80)
    print(yaml.dump(config, default_flow_style=False))
    
    # Create output directories
    os.makedirs(config['output']['output_path'], exist_ok=True)
    os.makedirs(config['output']['log_dir'], exist_ok=True)
    
    # Load model and tokenizer
    print("\n" + "=" * 80)
    print("Loading Model and Tokenizer")
    print("=" * 80)
    model_id = config['model']['model_id']
    print(f"Model: {model_id}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=config['model']['torch_dtype']
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Prepare dataset
    print("\n" + "=" * 80)
    print("Preparing Calibration Dataset")
    print("=" * 80)
    ds = prepare_dataset(config, tokenizer)
    print(f"Dataset prepared with {len(ds)} samples")
    
    # Create quantization recipe
    print("\n" + "=" * 80)
    print("Creating Quantization Recipe")
    print("=" * 80)
    recipe = create_quantization_recipe(config)
    
    # Apply quantization
    print("\n" + "=" * 80)
    print("Applying Quantization")
    print("=" * 80)
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=config['dataset']['calibration']['max_seq_length'],
        num_calibration_samples=config['dataset']['calibration']['num_samples'],
    )
    
    # Save quantized model
    print("\n" + "=" * 80)
    print("Saving Quantized Model")
    print("=" * 80)
    save_dir = generate_output_dir(config)
    print(f"Output directory: {save_dir}")
    
    model.save_pretrained(
        save_dir, 
        save_compressed=config['output']['save_compressed']
    )
    tokenizer.save_pretrained(save_dir)
    
    # Save configuration used
    config_save_path = os.path.join(save_dir, 'quantization_config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Configuration saved to: {config_save_path}")
    
    print("\n" + "=" * 80)
    print("Quantization Complete!")
    print("=" * 80)
    print(f"Model saved to: {save_dir}")


if __name__ == "__main__":
    main()

