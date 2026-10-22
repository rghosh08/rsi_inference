"""Roofline + continuous-batching simulator of the vLLM V1 scheduler.

It is not a substitute for a real benchmark. It exists so the agent loop can be developed, tested and
compiled (GEPA) without a GPU, and it reproduces the main first-order tradeoffs: batch size vs KV
capacity/preemption, token budget vs TTFT/TPOT, chunked prefill, fp8 KV, tensor parallelism.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from .config_space import VLLMConfig
from .metrics import RequestRecord
from .trace import Workload


@dataclass
class GPUProfile:
    name: str
    mem_gb: float
    tflops: float  # achievable dense bf16 TFLOP/s
    hbm_tbps: float
    num_gpus: int = 1


@dataclass
class ModelProfile:
    name: str
    params_b: float
    layers: int
    kv_heads: int
    head_dim: int
    hidden: int
    vocab: int

    def kv_bytes_per_token(self, dtype: str) -> float:
        return 2 * self.layers * self.kv_heads * self.head_dim * (1 if dtype == "fp8" else 2)


GPUS = {
    "H100-80GB": GPUProfile("H100-80GB", 80, 600, 3.35),
    "A100-80GB": GPUProfile("A100-80GB", 80, 250, 2.0),
    "L40S-48GB": GPUProfile("L40S-48GB", 48, 280, 0.86),
    "A10G-24GB": GPUProfile("A10G-24GB", 24, 90, 0.6),
}
MODELS = {
    "llama-3.1-8b": ModelProfile("meta-llama/Llama-3.1-8B-Instruct", 8.0, 32, 8, 128, 4096, 128256),
    "qwen2.5-14b": ModelProfile("Qwen/Qwen2.5-14B-Instruct", 14.7, 48, 8, 128, 5120, 152064),
    "llama-3.1-70b": ModelProfile("meta-llama/Llama-3.1-70B-Instruct", 70.6, 80, 8, 128, 8192, 128256),
}


class _Req:
    __slots__ = ("rec", "prompt", "out", "prefill_left", "computed", "blocks", "generated")

    def __init__(self, rec: RequestRecord):
        self.rec = rec
        self.prompt = rec.prompt_tokens
        self.out = rec.output_tokens
        self.prefill_left = rec.prompt_tokens
        self.computed = 0  # tokens resident in KV cache
        self.blocks = 0
        self.generated = 0


class StartupError(RuntimeError):
    pass


def simulate(cfg: VLLMConfig, w: Workload, gpu: GPUProfile, model: ModelProfile, horizon_s: float = 900.0) -> tuple[list[RequestRecord], dict]:
    tp = cfg.tensor_parallel_size
    weights = model.params_b * 1e9 * 2
    kv_tok = model.kv_bytes_per_token(cfg.kv_cache_dtype)

    # Memory accounting, roughly as vLLM's profiling run does it.
    usable = cfg.gpu_memory_utilization * gpu.mem_gb * 1e9 * tp
    if gpu.mem_gb * (1 - cfg.gpu_memory_utilization) < 1.2:
        raise StartupError("CUDA out of memory during profiling: gpu_memory_utilization leaves <1.2GB for CUDA context/NCCL")
    activations = cfg.max_num_batched_tokens * (model.hidden * 2 * 12) + cfg.max_num_seqs * model.vocab * 4
    cuda_graphs = 0.6e9 * tp
    kv_bytes = usable - weights - activations - cuda_graphs
    if kv_bytes <= 0:
        raise StartupError(f"No available memory for the cache blocks (weights {weights / 1e9:.1f}GB do not fit with tp={tp})")
    num_blocks = int(kv_bytes // (kv_tok * cfg.block_size))
    kv_capacity = num_blocks * cfg.block_size
    if kv_capacity < cfg.max_model_len:
        raise StartupError(f"max_model_len {cfg.max_model_len} exceeds KV cache capacity {kv_capacity} tokens")

    # Step-time model: max(compute, memory) + fixed overheads.
    flops_per_tok = 2 * model.params_b * 1e9
    compute_rate = gpu.tflops * 1e12 * tp * (0.92 if tp > 1 else 1.0)
    bw = gpu.hbm_tbps * 1e12 * tp
    attn_eff = {8: 0.93, 16: 1.0, 32: 1.02}[cfg.block_size]
    base_overhead = 0.0025 + (0.00002 * model.layers * 2 if tp > 1 else 0.0)
    overhead_mult = 1.01 if cfg.enable_prefix_caching else 1.0

    def step_time(batch_tokens: int, kv_read: int) -> float:
        compute = batch_tokens * flops_per_tok / compute_rate
        memory = (weights + kv_read * kv_tok / attn_eff) / bw
        eager = 0.0015 if batch_tokens > 512 else 0.0  # beyond the largest captured CUDA graph
        return (max(compute, memory) + base_overhead + eager) * overhead_mult

    recs = [RequestRecord(r.arrival, r.prompt_tokens, r.output_tokens) for r in w.requests]
    pending = deque(_Req(r) for r in recs)
    for q in list(pending):
        if q.prompt + q.out > cfg.max_model_len:
            q.rec.error = "prompt+output exceeds max_model_len (HTTP 400)"
    pending = deque(q for q in pending if q.rec.error is None)

    waiting: deque[_Req] = deque()
    running: list[_Req] = []
    free = num_blocks
    t = 0.0
    preemptions = 0
    steps = 0
    lpt = cfg.long_prefill_token_threshold or math.inf
    stop_at = w.duration + horizon_s

    def blocks_for(n: int) -> int:
        return -(-n // cfg.block_size)

    while (pending or waiting or running) and t < stop_at:
        while pending and pending[0].rec.arrival <= t:
            waiting.append(pending.popleft())
        if not running and not waiting:
            t = pending[0].rec.arrival
            continue

        budget = cfg.max_num_batched_tokens
        sched: list[tuple[_Req, int]] = []
        kv_read = 0
        i = 0
        while i < len(running):
            r = running[i]
            if r.prefill_left > 0:
                chunk = min(r.prefill_left, budget, lpt) if cfg.enable_chunked_prefill else (r.prefill_left if r.prefill_left <= budget else 0)
            else:
                chunk = 1 if budget >= 1 else 0
            if chunk == 0:
                i += 1
                continue
            need = blocks_for(r.computed + chunk) - r.blocks
            while free < need and running:
                victim = running.pop()  # lowest priority = most recently admitted
                free += victim.blocks
                victim.blocks = victim.computed = 0
                victim.prefill_left = victim.prompt + victim.generated  # recompute
                waiting.appendleft(victim)
                preemptions += 1
                if victim is r:
                    break
            if r not in running:
                break
            free -= need
            r.blocks += need
            sched.append((r, chunk))
            budget -= chunk
            kv_read += r.computed + chunk
            i += 1

        while waiting and len(running) < cfg.max_num_seqs and budget > 0:
            r = waiting[0]
            chunk = min(r.prefill_left, budget, lpt) if cfg.enable_chunked_prefill else (r.prefill_left if r.prefill_left <= budget else 0)
            if chunk == 0:
                break
            need = blocks_for(chunk)
            if free < need:
                break
            waiting.popleft()
            free -= need
            r.blocks = need
            running.append(r)
            sched.append((r, chunk))
            budget -= chunk
            kv_read += chunk

        if not sched:  # nothing fits (e.g. KV exhausted with no victims) — advance to next arrival
            if pending:
                t = max(t, pending[0].rec.arrival)
                continue
            break

        t += step_time(cfg.max_num_batched_tokens - budget, kv_read)
        steps += 1
        finished = []
        for r, chunk in sched:
            r.computed += chunk
            if r.prefill_left > 0:
                r.prefill_left -= chunk
                if r.prefill_left > 0:
                    continue
            r.generated += 1
            if r.rec.first_token is None:
                r.rec.first_token = t
            if r.generated >= r.out:
                r.rec.finish, r.rec.generated = t, r.generated
                finished.append(r)
        for r in finished:
            free += r.blocks
            running.remove(r)

    for q in list(pending) + list(waiting) + running:
        q.rec.error = q.rec.error or "timeout"
        q.rec.generated = q.generated
    info = {"kv_capacity_tokens": kv_capacity, "preemptions": preemptions, "engine_steps": steps}
    return recs, info
