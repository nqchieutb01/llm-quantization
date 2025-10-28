# echo "Quantizing Ultrachat"
# python3 run_quantize_configurable.py --config config.yaml --model_id Qwen/Qwen3-4B --dataset HuggingFaceH4/ultrachat_200k --method gptq --scheme W4A16

# echo "Quantizing GSM8K"
# python3 run_quantize_configurable.py --config config.yaml --model_id Qwen/Qwen3-4B --dataset openai/gsm8k --method gptq --scheme W4A16


# echo "Quantizing Ultrachat"
# python3 run_quantize_configurable.py --config config.yaml --model_id Qwen/Qwen3-4B --dataset HuggingFaceH4/ultrachat_200k --method awq --scheme W4A16_ASYM

echo "Quantizing GSM8K"
CUDA_VISIBLE_DEVICES="0" python3 run_quantize_configurable.py --config config.yaml --model_id Qwen/Qwen3-1.7B --dataset openai/gsm8k --method awq --scheme W4A16_ASYM --num_samples 128

CUDA_VISIBLE_DEVICES="0" python3 run_quantize_configurable.py --config config.yaml --model_id Qwen/Qwen3-1.7B --dataset openai/gsm8k --method awq --scheme W4A16_ASYM --num_samples 256

CUDA_VISIBLE_DEVICES="0" python3 run_quantize_configurable.py --config config.yaml --model_id Qwen/Qwen3-1.7B --dataset openai/gsm8k --method awq --scheme W4A16_ASYM --num_samples 512

CUDA_VISIBLE_DEVICES="0" python3 run_quantize_configurable.py --config config.yaml --model_id Qwen/Qwen3-1.7B --dataset openai/gsm8k --method awq --scheme W4A16_ASYM --num_samples 1024

# python3 run_quantize_configurable.py --config config.yaml --model_id Qwen/Qwen3-1.7B --dataset openai/gsm8k --method awq --scheme W4A16_ASYM --num_samples 2048
