"""The vLLM engine knobs the agent may tune, their legal values, and conversion to `vllm serve` flags."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields

# name -> (allowed values or (lo, hi) range, what it does). Rendered verbatim into the agent prompt.
SEARCH_SPACE: dict[str, tuple[object, str]] = {
    "max_num_seqs": ((8, 1024), "Max concurrent sequences in a batch. Higher = more decode throughput, more KV pressure and preemption."),
    "max_num_batched_tokens": ((512, 65536), "Per-step token budget (prefill chunks + decodes). Higher = faster prefill/TTFT, but long prefills stall decodes (TPOT)."),
    "enable_chunked_prefill": ([True, False], "Split long prompts across steps so decodes keep flowing. If False, max_num_batched_tokens must be >= max_model_len."),
    "long_prefill_token_threshold": ((0, 65536), "Cap on prefill tokens one request may take per step (0 = no cap). Lets short prompts overtake long ones."),
    "enable_prefix_caching": ([True, False], "Reuse KV of shared prefixes. Synthetic BurstGPT prompts share no prefixes, so this mainly costs hashing overhead."),
    "gpu_memory_utilization": ((0.5, 0.98), "Fraction of GPU memory vLLM may use; the remainder after weights/activations becomes KV cache."),
    "block_size": ([8, 16, 32], "KV cache block size in tokens."),
    "kv_cache_dtype": (["auto", "fp8"], "fp8 halves KV memory (≈2x KV capacity) at a small accuracy risk."),
    "tensor_parallel_size": ([1, 2, 4, 8], "Shard the model over N GPUs. Must not exceed available GPUs."),
    "max_model_len": ((1024, 131072), "Max context length served. Must cover the workload's longest prompt+output."),
}


@dataclass
class VLLMConfig:
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    enable_chunked_prefill: bool = True
    long_prefill_token_threshold: int = 0
    enable_prefix_caching: bool = True
    gpu_memory_utilization: float = 0.9
    block_size: int = 16
    kv_cache_dtype: str = "auto"
    tensor_parallel_size: int = 1
    max_model_len: int = 8192

    @classmethod
    def from_dict(cls, d: dict, base: "VLLMConfig | None" = None) -> "VLLMConfig":
        """Overlay `d` on `base` (defaults if None). Unknown keys raise so the agent learns the schema."""
        names = {f.name: f.type for f in fields(cls)}
        unknown = set(d) - set(names)
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}; allowed: {sorted(names)}")
        merged = asdict(base or cls())
        for k, v in d.items():
            default = merged[k]
            if isinstance(default, bool):
                v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes")
            elif isinstance(default, int):
                v = int(v)
            elif isinstance(default, float):
                v = float(v)
            merged[k] = v
        return cls(**merged)

    def validate(self, num_gpus: int, allow_fp8_kv: bool = True) -> list[str]:
        errs = []
        for name, (allowed, _) in SEARCH_SPACE.items():
            v = getattr(self, name)
            if isinstance(allowed, list) and v not in allowed:
                errs.append(f"{name}={v!r} not in {allowed}")
            elif isinstance(allowed, tuple) and not (allowed[0] <= v <= allowed[1]):
                errs.append(f"{name}={v} outside [{allowed[0]}, {allowed[1]}]")
        if self.tensor_parallel_size > num_gpus:
            errs.append(f"tensor_parallel_size={self.tensor_parallel_size} > available GPUs ({num_gpus})")
        if self.kv_cache_dtype == "fp8" and not allow_fp8_kv:
            errs.append("kv_cache_dtype=fp8 is disabled for this run (accuracy constraint)")
        if not self.enable_chunked_prefill and self.max_num_batched_tokens < self.max_model_len:
            errs.append("with chunked prefill off, max_num_batched_tokens must be >= max_model_len")
        if self.max_num_batched_tokens < self.max_num_seqs:
            errs.append("max_num_batched_tokens must be >= max_num_seqs")
        return errs

    def to_cli_args(self) -> list[str]:
        args = [
            "--max-num-seqs", str(self.max_num_seqs),
            "--max-num-batched-tokens", str(self.max_num_batched_tokens),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--block-size", str(self.block_size),
            "--kv-cache-dtype", self.kv_cache_dtype,
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--max-model-len", str(self.max_model_len),
            "--enable-chunked-prefill" if self.enable_chunked_prefill else "--no-enable-chunked-prefill",
            "--enable-prefix-caching" if self.enable_prefix_caching else "--no-enable-prefix-caching",
        ]
        if self.long_prefill_token_threshold:
            args += ["--long-prefill-token-threshold", str(self.long_prefill_token_threshold)]
        return args

    def key(self) -> str:
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:10]

    def diff(self, other: "VLLMConfig") -> dict:
        a, b = asdict(self), asdict(other)
        return {k: a[k] for k in a if a[k] != b[k]}


def describe_search_space() -> str:
    lines = []
    for name, (allowed, doc) in SEARCH_SPACE.items():
        rng = f"one of {allowed}" if isinstance(allowed, list) else f"range [{allowed[0]}, {allowed[1]}]"
        lines.append(f"- {name} (default {getattr(VLLMConfig(), name)!r}; {rng}): {doc}")
    return "\n".join(lines)
