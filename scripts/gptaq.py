#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Dict, Any, List

from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer

# External: requires your local package providing GPTQModel/QuantizeConfig
from gptqmodel import GPTQModel, QuantizeConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPTQ quantization data-prep + run")
    p.add_argument("--model-id", type=str, default="Qwen/Qwen3-0.6B",
                   help="HF model id to quantize.")
    p.add_argument("--dataset-name", type=str, default="HuggingFaceH4/ultrachat_200k",
                   help='Dataset path on Hub. e.g. "openai/gsm8k" or "HuggingFaceH4/ultrachat_200k"')
    p.add_argument("--subset", type=str, default="train_sft[:2048]",
                   help='Split/subset slice. For gsm8k use "main", for ultrachat use "train_sft[:2048]".')
    p.add_argument("--num-samples", type=int, default=2048,
                   help="Number of calibration samples.")
    p.add_argument("--max-seq-len", type=int, default=2048,
                   help="Max sequence length for tokenization/truncation.")
    p.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    p.add_argument("--bits", type=int, default=4, help="Quantization bits.")
    p.add_argument("--group-size", type=int, default=128, help="Quantization group size.")
    p.add_argument("--batch-size", type=int, default=2, help="Quantization batch size.")
    p.add_argument("--quant-path", type=str, default=None,
                   help="Output dir for quantized weights. If None, auto-generates from args.")
    p.add_argument("--no-shuffle", action="store_true",
                   help="Disable shuffling before selecting calibration samples.")
    p.add_argument("--bos-eos-policy", type=str, default="no_add",
                   choices=["no_add", "add_bos", "add_eos", "add_bos_eos"],
                   help="Whether to add BOS/EOS if your chat template does not.")
    return p.parse_args()


def safe_select(ds: Dataset, n: int, seed: int, do_shuffle: bool) -> Dataset:
    if do_shuffle:
        ds = ds.shuffle(seed=seed)
    n = min(n, len(ds))
    return ds.select(range(n))


def build_messages_for_gsm8k(example: Dict[str, Any]) -> List[Dict[str, str]]:
    # gsm8k schema: {"question": ..., "answer": ...}
    q = example.get("question", "")
    a = example.get("answer", "")
    return [
        {"role": "user", "content": q},
        {"role": "assistant", "content": a},
    ]


def apply_template(tokenizer: AutoTokenizer, messages_or_text, bos_eos_policy: str) -> str:
    """
    Convert either:
      - list[{"role":..., "content":...}] -> chat template
      - str -> returned as-is
    """
    if isinstance(messages_or_text, list):
        try:
            text = tokenizer.apply_chat_template(messages_or_text, tokenize=False)
        except Exception:
            # Fallback: naive concatenation if template missing
            text = ""
            for m in messages_or_text:
                role = m.get("role", "user")
                content = m.get("content", "")
                text += f"<|{role}|>\n{content}\n"
    else:
        text = str(messages_or_text)

    # Optionally add BOS/EOS if template doesn't
    if bos_eos_policy != "no_add":
        bos = tokenizer.bos_token or ""
        eos = tokenizer.eos_token or ""
        if bos_eos_policy in ("add_bos", "add_bos_eos") and bos and not text.startswith(bos):
            text = bos + text
        if bos_eos_policy in ("add_eos", "add_bos_eos") and eos and not text.endswith(eos):
            text = text + eos
    return text


def preprocess_example_ultrachat(ex: Dict[str, Any], tokenizer: AutoTokenizer, bos_eos_policy: str) -> Dict[str, str]:
    # Ultrachat examples usually contain a "messages" field (list of dicts with role/content)
    if "messages" in ex and isinstance(ex["messages"], list):
        txt = apply_template(tokenizer, ex["messages"], bos_eos_policy)
        return {"text": txt}
    # Fallback
    return {"text": ex.get("text", "")}


def preprocess_example_gsm8k(ex: Dict[str, Any], tokenizer: AutoTokenizer, bos_eos_policy: str) -> Dict[str, str]:
    messages = build_messages_for_gsm8k(ex)
    txt = apply_template(tokenizer, messages, bos_eos_policy)
    return {"text": txt}


def generic_preprocess(ex: Dict[str, Any]) -> Dict[str, str]:
    # Best effort if dataset isn't recognized
    if "text" in ex and isinstance(ex["text"], str):
        return {"text": ex["text"]}
    # Join all string fields
    joined = " ".join(str(v) for v in ex.values() if isinstance(v, str))
    return {"text": joined}


def main():
    args = parse_args()

    MODEL_ID = args.model_id
    DATASET_NAME = args.dataset_name
    SUBSET = args.subset
    NUM_CALIBRATION_SAMPLES = args.num_samples
    MAX_SEQUENCE_LENGTH = args.max_seq_len
    DO_SHUFFLE = not args.no_shuffle

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Load dataset (handle both split string and default config)
    if "ultrachat" in DATASET_NAME.lower():
        ds = load_dataset(DATASET_NAME, split=SUBSET)
    else:
        # For datasets like gsm8k that use 'main' or other splits
        ds = load_dataset(DATASET_NAME, SUBSET)

    # Normalize to a single Dataset
    if isinstance(ds, DatasetDict):
        if "train" in ds:
            ds = ds["train"]
        else:
            # pick first available split
            first_split = list(ds.values())[0]
            ds = first_split

    # Sample selection
    ds = safe_select(ds, NUM_CALIBRATION_SAMPLES, seed=args.seed, do_shuffle=DO_SHUFFLE)

    # Preprocess to model's chat format
    if "gsm8k" in DATASET_NAME.lower():
        ds = ds.map(lambda ex: preprocess_example_gsm8k(ex, tokenizer, args.bos_eos_policy))
    elif "ultrachat" in DATASET_NAME.lower():
        ds = ds.map(lambda ex: preprocess_example_ultrachat(ex, tokenizer, args.bos_eos_policy))
    else:
        print(f"[warn] Using generic preprocessing for {DATASET_NAME}", file=sys.stderr)
        ds = ds.map(generic_preprocess)

    # Tokenize (chat_template usually already includes BOS/EOS; do not double-add)
    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(tokenize, remove_columns=[c for c in ds.column_names if c != "input_ids"])

    # Build output path if not provided
    last_ds_name = DATASET_NAME.split("/")[-1]
    default_quant_path = f"{MODEL_ID.split('/')[-1]}-gptq-v2-{args.bits}bit-{last_ds_name}-{NUM_CALIBRATION_SAMPLES}"
    quant_path = args.quant_path or default_quant_path
    os.makedirs(quant_path, exist_ok=True)

    # Quantization config & run
    quant_config = QuantizeConfig(bits=args.bits, group_size=args.group_size)
    model = GPTQModel.load(MODEL_ID, quant_config)

    print(f"[info] Starting GPTQ quantization:")
    print(f"       model_id       = {MODEL_ID}")
    print(f"       dataset        = {DATASET_NAME} [{SUBSET}]")
    print(f"       samples        = {len(ds)}")
    print(f"       max_seq_len    = {MAX_SEQUENCE_LENGTH}")
    print(f"       bits/group     = {args.bits}/{args.group_size}")
    print(f"       batch_size     = {args.batch_size}")
    print(f"       output         = {quant_path}")

    # increase batch_size to match your GPU/VRAM specs to speed up
    model.quantize(ds, batch_size=args.batch_size)
    model.save(quant_path)

    print(f"[done] Saved quantized model to: {quant_path}")


if __name__ == "__main__":
    main()
