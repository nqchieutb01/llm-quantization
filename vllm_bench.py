#!/usr/bin/env python3
"""
vllm_bench.py — TTFT, throughput, and peak GPU RAM for vLLM

Usage (examples):
  python vllm_bench.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --prompts "Write a haiku about the ocean." "Explain transformers in 2 lines." \
    --max-new-tokens 128 --temperature 0.0 --tp 1 --dtype auto

  # Or read prompts from a file (one per line):
  python vllm_bench.py --model mistralai/Mistral-7B-Instruct-v0.3 \
    --prompts-file prompts.txt --max-new-tokens 128
"""
import argparse, asyncio, time, uuid, math, os
from statistics import mean
from typing import List, Optional

# vLLM
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.async_engine_args import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams

# Torch (for GPU memory stats)
import torch

# Optional NVML for extra GPU memory telemetry
try:
    import pynvml
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False

def now() -> float:
    return time.perf_counter()

def bytes_to_gib(n: int) -> float:
    return n / (1024**3)

async def run_benchmark(
    model: str,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    stop: List[str],
    tensor_parallel_size: int,
    dtype: str,
    gpu_memory_utilization: float,
    enforce_eager: bool,
    kv_cache_dtype: Optional[str],
    seed: Optional[int],
) -> None:

    # Torch peak memory tracking
    torch.cuda.reset_peak_memory_stats()
    start_nvml_used = None
    peak_nvml_used = None
    nvml_handle = None

    if _HAS_NVML and torch.cuda.is_available():
        pynvml.nvmlInit()
        nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
        start_nvml_used = mem.used
        peak_nvml_used = mem.used

    # Engine args
    engine_args = AsyncEngineArgs(
        model=model,
        # tensor_parallel_size=tensor_parallel_size,
        # dtype=None if dtype == "auto" else dtype,    # vLLM will choose if None
        gpu_memory_utilization=gpu_memory_utilization,
        # enforce_eager=enforce_eager,
        seed=seed,
        # kv_cache_dtype=kv_cache_dtype,
    )

    print(f"Loading model: {model}")
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    sampling = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        # top_p=top_p,
        # top_k=top_k if top_k >= 0 else None,
        # stop=stop if stop else None,
        n=1,
    )

    # Warmup (single short request to trigger graph compile / page setup)
    warmup_prompt = "Warm up."
    warmup_id = str(uuid.uuid4())
    await engine.add_request(
        warmup_id, warmup_prompt, sampling
    )
    async for _ in engine.get_request_stream(warmup_id):
        pass
    # Give a tiny breather
    await asyncio.sleep(0.05)

    # Actual run (batched: we submit all requests then read their streams)
    request_ids = []
    submit_t = now()
    for p in prompts:
        rid = str(uuid.uuid4())
        request_ids.append(rid)
        await engine.add_request(rid, p, sampling)

    # Per-request metrics we’ll fill during streaming
    ttfts = {rid: None for rid in request_ids}       # seconds
    first_token_times = {rid: None for rid in request_ids}
    start_times = {rid: submit_t for rid in request_ids}
    end_times = {rid: None for rid in request_ids}
    out_token_counts = {rid: 0 for rid in request_ids}

    # We’ll listen to *all* streams concurrently.
    async def consume(rid: str):
        nonlocal peak_nvml_used
        async for ro in engine.get_request_stream(rid):
            # ro contains incremental outputs; take the first output
            if not ro.outputs:
                continue
            o = ro.outputs[0]
            # Count tokens generated so far
            out_token_counts[rid] = len(o.token_ids)  # includes all emitted so far
            # TTFT: when we first observe any token
            if out_token_counts[rid] > 0 and ttfts[rid] is None:
                first_token_times[rid] = now()
                ttfts[rid] = first_token_times[rid] - start_times[rid]

            # Track peak via NVML during the stream
            if _HAS_NVML and nvml_handle is not None:
                mem = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
                if mem.used > (peak_nvml_used or 0):
                    peak_nvml_used = mem.used

        # stream is finished
        end_times[rid] = now()

    consumers = [consume(rid) for rid in request_ids]
    await asyncio.gather(*consumers)
    total_wall = max(end_times.values()) - submit_t

    # Tokens & throughput
    total_gen_tokens = sum(out_token_counts.values())
    tput = total_gen_tokens / total_wall if total_wall > 0 else float("nan")

    # TTFT stats (ignore any None, though they should all be filled)
    valid_ttfts = [t for t in ttfts.values() if t is not None]
    ttft_avg = mean(valid_ttfts) if valid_ttfts else float("nan")
    ttft_p50 = sorted(valid_ttfts)[len(valid_ttfts)//2] if valid_ttfts else float("nan")
    ttft_p95 = (lambda xs: xs[math.floor(0.95*(len(xs)-1))])(sorted(valid_ttfts)) if valid_ttfts else float("nan")

    # Peak GPU memory (PyTorch + NVML best-effort)
    torch_peak_reserved = torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
    torch_peak_alloc = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

    if _HAS_NVML and nvml_handle is not None:
        mem = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
        end_nvml_used = mem.used
        if peak_nvml_used is None:
            peak_nvml_used = end_nvml_used
        pynvml.nvmlShutdown()
    else:
        start_nvml_used = None
        peak_nvml_used = None

    # Report
    print("\n=== vLLM Benchmark ===")
    print(f"Model: {model}")
    print(f"Requests: {len(prompts)} | Max new tokens: {max_new_tokens}")
    print(f"TP size: {tensor_parallel_size} | dtype: {dtype} | gpu_mem_util: {gpu_memory_utilization}")
    print(f"Total generated tokens: {total_gen_tokens}")
    print(f"Wall time: {total_wall:.3f}s")
    print(f"Throughput (tokens/sec): {tput:.2f}")
    print(f"TTFT avg: {ttft_avg:.3f}s | p50: {ttft_p50:.3f}s | p95: {ttft_p95:.3f}s")

    print("\n--- GPU Memory (GiB) ---")
    if torch.cuda.is_available():
        print(f"PyTorch max_reserved: {bytes_to_gib(torch_peak_reserved):.2f} GiB")
        print(f"PyTorch max_allocated: {bytes_to_gib(torch_peak_alloc):.2f} GiB")
    else:
        print("CUDA not available.")

    if peak_nvml_used is not None:
        print(f"NVML peak used: {bytes_to_gib(peak_nvml_used):.2f} GiB")
        if start_nvml_used is not None:
            print(f"NVML delta (peak - start): {bytes_to_gib(peak_nvml_used - start_nvml_used):.2f} GiB")
    else:
        print("NVML not available (install `pynvml` for extra telemetry).")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF repo or local path")
    ap.add_argument("--prompts", nargs="*", default=[], help="Prompts (space-separated; quote each)")
    ap.add_argument("--prompts-file", type=str, default=None, help="Text file with one prompt per line")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=-1, help="-1 to disable")
    ap.add_argument("--stop", nargs="*", default=[])
    ap.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--gpu-mem-util", type=float, default=0.90, help="vLLM gpu_memory_utilization")
    ap.add_argument("--enforce-eager", action="store_true", help="Disable CUDA graph capture")
    ap.add_argument("--kv-cache-dtype", default=None, choices=[None, "auto", "fp8", "fp8_e5m2", "fp8_e4m3", "float16", "bfloat16"])
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    prompts = list(args.prompts)
    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            prompts += [line.strip() for line in f if line.strip()]
    if not prompts:
        prompts = ["Tell me a short, fun fact about dolphins."] * 8  # sensible default batch

    return {
        "model": args.model,
        "prompts": prompts,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "stop": args.stop,
        "tensor_parallel_size": args.tp,
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_mem_util,
        "enforce_eager": args.enforce_eager,
        "kv_cache_dtype": args.kv_cache_dtype,
        "seed": args.seed,
    }

if __name__ == "__main__":
    # Some vLLM setups prefer fork to reduce overhead (optional):
    os.environ.setdefault("VLLM_WORKER_MULTIPROCESSING_METHOD", "fork")
    cfg = parse_args()
    asyncio.run(run_benchmark(**cfg))
