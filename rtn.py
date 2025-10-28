from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import BitsAndBytesConfig

model_id = "Qwen/Qwen3-4B"
save_dir = "models/4B/Qwen3-4B-RTN-INT4"

quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

# Load the model and apply quantization config
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=quantization_config
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Save the model
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)