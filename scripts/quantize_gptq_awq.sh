echo "Quantizing GSM8K"
CUDA_VISIBLE_DEVICES="0" python3 ../run_quantize_configurable.py --config ../config/config_quantize.yaml --model_id Qwen/Qwen3-1.7B --dataset openai/gsm8k --method awq --scheme W4A16_ASYM --num_samples 128

CUDA_VISIBLE_DEVICES="0" python3 ../run_quantize_configurable.py --config ../config/config_quantize.yaml --model_id Qwen/Qwen3-1.7B --dataset openai/gsm8k --method awq --scheme W4A16_ASYM --num_samples 256

CUDA_VISIBLE_DEVICES="0" python3 ../run_quantize_configurable.py --config ../config/config_quantize.yaml --model_id Qwen/Qwen3-1.7B --dataset openai/gsm8k --method awq --scheme W4A16_ASYM --num_samples 512

CUDA_VISIBLE_DEVICES="0" python3 ../run_quantize_configurable.py --config ../config/config_quantize.yaml --model_id Qwen/Qwen3-1.7B --dataset openai/gsm8k --method awq --scheme W4A16_ASYM --num_samples 1024