# echo "Waiting for 20 minutes"
# sleep 20m


CUDA_VISIBLE_DEVICES=1 python ../gptq_v2.py --model-id Qwen/Qwen3-1.7B --dataset-name openai/gsm8k --subset main --num-samples 128

CUDA_VISIBLE_DEVICES=1 python ../gptq_v2.py --model-id Qwen/Qwen3-1.7B --dataset-name openai/gsm8k --subset main --num-samples 256

CUDA_VISIBLE_DEVICES=1 python ../gptq_v2.py --model-id Qwen/Qwen3-1.7B --dataset-name openai/gsm8k --subset main --num-samples 512

CUDA_VISIBLE_DEVICES=1 python ../gptq_v2.py --model-id Qwen/Qwen3-1.7B --dataset-name openai/gsm8k --subset main --num-samples 1024

# CUDA_VISIBLE_DEVICES=1 python ../gptq_v2.py --model-id Qwen/Qwen3-1.7B --dataset-name openai/gsm8k --subset main --num-samples 2048