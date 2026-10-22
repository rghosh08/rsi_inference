# burstgpt-tuner

A DSPy ReAct agent that tunes `vllm serve` engine settings for a [BurstGPT](https://github.com/HPMLL/BurstGPT) workload.
The objective is **goodput**: the number of requests per second that meet both the TTFT and TPOT SLOs.

```
BurstGPT CSV ──► window + time-compress ──► workload stats ─┐
                                                            ▼
                      ┌──────────── dspy.ReAct (TuneVLLMConfig) ────────────┐
                      │ tools: run_trial(config, hypothesis) · check_config │
                      │        trial_history                                │
                      └───────────────────────┬─────────────────────────────┘
                                              ▼
                 backend: SimBackend (no GPU)  |  VLLMBackend (launch vllm serve + replay trace)
                                              ▼
                         goodput, SLO attainment, TTFT/TPOT p50/p90/p99, preemptions
```

- The baseline (vLLM defaults) runs first. Every later trial counts against `--budget`.
- Invalid configs are rejected for free with an explanation. A duplicate config returns its cached result.
- The reported best config is the best **measured** trial, not whatever the LM says is best.
- The `compile` command uses **DSPy GEPA** to optimize the agent's own instructions. It trains on many simulated
  BurstGPT windows and hardware setups, scoring each against a 200-sample random-search reference.

## Setup

```bash
uv venv && uv pip install -e .          # add vllm on the GPU host: uv pip install -e '.[gpu]'
export ANTHROPIC_API_KEY=...            # or any LiteLLM provider; choose with --lm / TUNER_LM
```

`data/BurstGPT_1.csv` (the first 2 months, ~1.4M requests) downloads automatically if it is missing. Larger traces
are on the [v2.0 release](https://github.com/HPMLL/BurstGPT/releases/tag/v2.0). Pass `--trace` to use one.

## Usage

```bash
# Workload stats for a trace window (default: 09:00–12:00 on day 1, time-compressed to 10 rps)
python -m burstgpt_tuner stats --start 32400 --duration 10800 --target-rps 10

# Benchmark one config
python -m burstgpt_tuner trial --backend sim --sim-gpu L40S-48GB --target-rps 12 --config '{"kv_cache_dtype":"fp8"}'

# Run the agent on the simulator (no GPU needed)
python -m burstgpt_tuner tune --backend sim --sim-gpu L40S-48GB --target-rps 12 --budget 12

# Run the agent against real vLLM (on the GPU host)
python -m burstgpt_tuner tune --backend vllm --model meta-llama/Llama-3.1-8B-Instruct \
    --num-gpus 1 --target-rps 12 --ttft-slo 2 --tpot-slo 0.1 --budget 12 --out runs/llama8b

# Optimize the agent's prompts with GEPA on simulated tasks, then use the result
python -m burstgpt_tuner compile --out runs/compiled
python -m burstgpt_tuner tune --program runs/compiled/tuner.json ...
```

Outputs go to `--out`:
- `trials.jsonl`: every trial, with its config, hypothesis and metrics.
- `best_config.json`: the best config, the matching `vllm serve` command, and the agent's rationale.
- Server logs for each trial (vllm backend only), under `runs/server_logs/`.

## Tuned knobs

`max_num_seqs`, `max_num_batched_tokens`, `enable_chunked_prefill`, `long_prefill_token_threshold`,
`enable_prefix_caching`, `gpu_memory_utilization`, `block_size`, `kv_cache_dtype`, `tensor_parallel_size`,
`max_model_len`. To add a knob, add it to `SEARCH_SPACE`, `VLLMConfig` and `to_cli_args()` in
`burstgpt_tuner/config_space.py`.
`--no-fp8-kv` forbids fp8 KV cache for runs where accuracy must not change.

## Notes and limitations

- **Load scaling.** The raw trace averages well under 1 rps. `--target-rps` rescales inter-arrival times
  uniformly, so the trace's burst shape is kept rather than replaced with a Poisson process.
- **Prompts are synthetic.** BurstGPT has token counts but no text. The real backend sends random token-id
  prompts with `ignore_eos` and `min_tokens`, so output lengths match the trace exactly. As a result, prefix
  caching cannot help, which the agent is told.
- **Goodput ceiling.** At a fixed offered load, goodput tops out at the offered rate. Once every config meets the
  SLO, ties go to the one with the most p99 latency headroom. Raise `--target-rps` to find the configs that hold
  up under heavier load.
- **The simulator is approximate.** It models roofline step times and the V1 scheduler (token budget, chunked
  prefill, KV blocks, recompute preemption). Use it to develop and compile the agent, then confirm on real
  hardware.
- **Trial cost.** Each real trial restarts the server, typically 1–3 minutes for model load plus the replay
  time. Size `--budget` and `--max-requests` to fit.
- **LM choice.** The default LM is `anthropic/claude-opus-5-5`. The code sets no temperature, because current
  Claude models reject sampling parameters.
