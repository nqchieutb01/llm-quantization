
# Baseline
# lm_eval --model vllm --model_args pretrained="Qwen/Qwen2.5-0.5B-Instruct",gpu_memory_utilization=0.8,add_bos_token=true --tasks gsm8k --num_fewshot 5 --limit 250 --batch_size 8 --device 'cuda:1' --output_path result/original.json --write_out --log_samples

# # Quantize 
# lm_eval --model vllm --model_args pretrained="./Qwen2.5-0.5B-Instruct-W4A16-G128",gpu_memory_utilization=0.8,add_bos_token=true --tasks gsm8k --num_fewshot 5 --limit 250 --batch_size 8 --device 'cuda:1' --output_path result/W4A16-G128.json --write_out --log_samples

# 1.7B
# CUDA_VISIBLE_DEVICES="1" lm_eval --model vllm --model_args pretrained="models/1.7B/Qwen3-1.7B-gptq-v2-4bit-gsm8k2048",gpu_memory_utilization=0.8,add_bos_token=true --tasks gsm8k --num_fewshot 1 --limit 500 --batch_size 8

# CUDA_VISIBLE_DEVICES="1" lm_eval --model vllm --model_args pretrained="models/1.7B/Qwen3-1.7B-gptq-v2-4bit-gsm8k2048",gpu_memory_utilization=0.8,add_bos_token=true --tasks gsm8k --num_fewshot 2 --limit 500 --batch_size 8 

# CUDA_VISIBLE_DEVICES="1" lm_eval --model vllm --model_args pretrained="models/1.7B/Qwen3-1.7B-gptq-v2-4bit-gsm8k2048",gpu_memory_utilization=0.8,add_bos_token=true --tasks gsm8k --num_fewshot 10 --limit 500 --batch_size 8 

#GPTQ v2

# 0.6B
# CUDA_VISIBLE_DEVICES="1" lm_eval --model vllm --model_args pretrained="models/0.6B/Qwen3-0.6B-gptq-v2-4bit-gsm8k2048",gpu_memory_utilization=0.8,add_bos_token=true --tasks gsm8k --num_fewshot 1 --limit 500 --batch_size 8 

# CUDA_VISIBLE_DEVICES="1" lm_eval --model vllm --model_args pretrained="models/0.6B/Qwen3-0.6B-gptq-v2-4bit-gsm8k2048",gpu_memory_utilization=0.8,add_bos_token=true --tasks gsm8k --num_fewshot 2 --limit 500 --batch_size 8 

# CUDA_VISIBLE_DEVICES="1" lm_eval --model vllm --model_args pretrained="models/0.6B/Qwen3-0.6B-gptq-v2-4bit-gsm8k2048",gpu_memory_utilization=0.8,add_bos_token=true --tasks gsm8k --num_fewshot 10 --limit 500 --batch_size 8 

# 4B
# CUDA_VISIBLE_DEVICES="1" lm_eval --model vllm --model_args pretrained="models/4B/Qwen3-4B-gptq-v2-4bit-gsm8k2048",gpu_memory_utilization=0.8,add_bos_token=true --tasks gsm8k --num_fewshot 1 --limit 500 --batch_size 8 

# CUDA_VISIBLE_DEVICES="1" lm_eval --model vllm --model_args pretrained="models/4B/Qwen3-4B-gptq-v2-4bit-gsm8k2048",gpu_memory_utilization=0.8,add_bos_token=true --tasks gsm8k --num_fewshot 2 --limit 500 --batch_size 8 

# CUDA_VISIBLE_DEVICES="1" lm_eval --model vllm --model_args pretrained="models/4B/Qwen3-4B-gptq-v2-4bit-gsm8k2048",gpu_memory_utilization=0.8,add_bos_token=true --tasks gsm8k --num_fewshot 10 --limit 500 --batch_size 8 


# AWQ 
# 0.6B
# CUDA_VISIBLE_DEVICES="1" lm_eval --model vllm --model_args pretrained="models/0.6B/Qwen3-0.6B-AWQ-W4A16_ASYM_gsm8k2048",gpu_memory_utilization=0.8,add_bos_token=true --tasks gsm8k --num_fewshot 1 --limit 500 --batch_size 8 

# CUDA_VISIBLE_DEVICES="1" lm_eval --model vllm --model_args pretrained="models/0.6B/Qwen3-0.6B-AWQ-W4A16_ASYM_gsm8k2048",gpu_memory_utilization=0.8,add_bos_token=true --tasks gsm8k --num_fewshot 2 --limit 500 --batch_size 8 

# CUDA_VISIBLE_DEVICES="1" lm_eval --model vllm --model_args pretrained="models/0.6B/Qwen3-0.6B-AWQ-W4A16_ASYM_gsm8k2048",gpu_memory_utilization=0.8,add_bos_token=true --tasks gsm8k --num_fewshot 10 --limit 500 --batch_size 8 

# # 1.7B
# CUDA_VISIBLE_DEVICES="1" lm_eval --model vllm --model_args pretrained="models/1.7B/Qwen3-1.7B-AWQ-W4A16_ASYM_gsm8k2048",gpu_memory_utilization=0.8,add_bos_token=true --tasks gsm8k --num_fewshot 1 --limit 500 --batch_size 8 

# CUDA_VISIBLE_DEVICES="1" lm_eval --model vllm --model_args pretrained="models/1.7B/Qwen3-1.7B-AWQ-W4A16_ASYM_gsm8k2048",gpu_memory_utilization=0.8,add_bos_token=true --tasks gsm8k --num_fewshot 2 --limit 500 --batch_size 8 

# CUDA_VISIBLE_DEVICES="1" lm_eval --model vllm --model_args pretrained="models/1.7B/Qwen3-1.7B-AWQ-W4A16_ASYM_gsm8k2048",gpu_memory_utilization=0.8,add_bos_token=true --tasks gsm8k --num_fewshot 10 --limit 500 --batch_size 8 

# # 4B
# CUDA_VISIBLE_DEVICES="1" lm_eval --model vllm --model_args pretrained="models/4B/Qwen3-4B-AWQ-W4A16_ASYM_gsm8k2048",gpu_memory_utilization=0.8,add_bos_token=true --tasks gsm8k --num_fewshot 1 --limit 500 --batch_size 8 

# CUDA_VISIBLE_DEVICES="1" lm_eval --model vllm --model_args pretrained="models/4B/Qwen3-4B-AWQ-W4A16_ASYM_gsm8k2048",gpu_memory_utilization=0.8,add_bos_token=true --tasks gsm8k --num_fewshot 2 --limit 500 --batch_size 8 

# CUDA_VISIBLE_DEVICES="1" lm_eval --model vllm --model_args pretrained="models/4B/Qwen3-4B-AWQ-W4A16_ASYM_gsm8k2048",gpu_memory_utilization=0.8,add_bos_token=true --tasks gsm8k --num_fewshot 10 --limit 500 --batch_size 8 


# GPTQ
# 0.6B
# CUDA_VISIBLE_DEVICES="1" lm_eval --model vllm --model_args pretrained="models/0.6B/Qwen3-0.6B-GPTQ-W4A16_gsm8k2048",gpu_memory_utilization=0.8,add_bos_token=true --tasks gsm8k --num_fewshot 1 --limit 500 --batch_size 8 

# CUDA_VISIBLE_DEVICES="1" lm_eval --model vllm --model_args pretrained="models/0.6B/Qwen3-0.6B-GPTQ-W4A16_gsm8k2048",gpu_memory_utilization=0.8,add_bos_token=true --tasks gsm8k --num_fewshot 2 --limit 500 --batch_size 8 

# CUDA_VISIBLE_DEVICES="1" lm_eval --model vllm --model_args pretrained="models/0.6B/Qwen3-0.6B-GPTQ-W4A16_gsm8k2048",gpu_memory_utilization=0.8,add_bos_token=true --tasks gsm8k --num_fewshot 10 --limit 500 --batch_size 8 

# 1.7B
# CUDA_VISIBLE_DEVICES="1" lm_eval --model vllm --model_args pretrained="models/1.7B/Qwen3-1.7B-GPTQ-W4A16_gsm8k2048",gpu_memory_utilization=0.8,add_bos_token=true --tasks gsm8k --num_fewshot 1 --limit 500 --batch_size 8 

# CUDA_VISIBLE_DEVICES="1" lm_eval --model vllm --model_args pretrained="models/1.7B/Qwen3-1.7B-GPTQ-W4A16_gsm8k2048",gpu_memory_utilization=0.8,add_bos_token=true --tasks gsm8k --num_fewshot 2 --limit 500 --batch_size 8 

# CUDA_VISIBLE_DEVICES="1" lm_eval --model vllm --model_args pretrained="models/1.7B/Qwen3-1.7B-GPTQ-W4A16_gsm8k2048",gpu_memory_utilization=0.8,add_bos_token=true --tasks gsm8k --num_fewshot 10 --limit 500 --batch_size 8 

# 4B
CUDA_VISIBLE_DEVICES="1" lm_eval --model vllm --model_args pretrained="models/4B/Qwen3-4B-GPTQ-W4A16_gsm8k2048",gpu_memory_utilization=0.8,add_bos_token=true --tasks gsm8k --num_fewshot 1 --limit 500 --batch_size 8 

CUDA_VISIBLE_DEVICES="1" lm_eval --model vllm --model_args pretrained="models/4B/Qwen3-4B-GPTQ-W4A16_gsm8k2048",gpu_memory_utilization=0.8,add_bos_token=true --tasks gsm8k --num_fewshot 2 --limit 500 --batch_size 8 

CUDA_VISIBLE_DEVICES="1" lm_eval --model vllm --model_args pretrained="models/4B/Qwen3-4B-GPTQ-W4A16_gsm8k2048",gpu_memory_utilization=0.8,add_bos_token=true --tasks gsm8k --num_fewshot 10 --limit 500 --batch_size 8 