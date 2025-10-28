from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3-4B"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

from datasets import load_dataset

NUM_CALIBRATION_SAMPLES=2048
MAX_SEQUENCE_LENGTH=2048

# Load dataset.
dataset_name = "openai/gsm8k"
ds = load_dataset(dataset_name, "main")
ds = ds.shuffle(seed=42)['train'].select(range(NUM_CALIBRATION_SAMPLES))

# Preprocess the data into the format the model is trained with.
def preprocess(example):
    messages = [
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}
ds = ds.map(preprocess)

# Tokenize the data (be careful with bos tokens - we need add_special_tokens=False since the chat_template already added it).
def tokenize(sample):
    return tokenizer(sample["text"], padding=False, max_length=MAX_SEQUENCE_LENGTH, truncation=True, add_special_tokens=False)

ds = ds.map(tokenize, remove_columns=ds.column_names)

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

recipe = [
    # SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(scheme="W4A16", targets=["Linear"], ignore=["lm_head", r"re:model\.layers\.(\d*?[02468])\.*"]),
]

# Apply quantization.
oneshot(
    model=model, 
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Save to disk compressed.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-W4A16-all" + "_" + dataset_name.split("/")[-1] + str(NUM_CALIBRATION_SAMPLES)
print("Saving model to", SAVE_DIR)
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)