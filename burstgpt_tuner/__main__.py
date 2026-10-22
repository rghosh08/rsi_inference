"""CLI: `python -m burstgpt_tuner {stats,trial,tune,compile} ...`"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .backends import SimBackend
from .config_space import VLLMConfig
from .metrics import SLO
from .simulator import GPUS, MODELS
from .trace import DEFAULT_TRACE_URL, download_trace, load_trace, make_workload, summarize


def add_workload_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--trace", default="data/BurstGPT_1.csv", help=f"BurstGPT CSV (downloaded from {DEFAULT_TRACE_URL} if missing)")
    p.add_argument("--trace-model", choices=["ChatGPT", "GPT-4"], help="keep only requests to this upstream model")
    p.add_argument("--log-type", choices=["Conversation log", "API log"])
    p.add_argument("--start", type=float, default=9 * 3600, help="window start, seconds into the trace (default: 09:00 day 1)")
    p.add_argument("--duration", type=float, default=3 * 3600, help="window length, seconds of trace time")
    p.add_argument("--target-rps", type=float, default=10.0,
                   help="time-compress the window to this mean request rate (the raw trace averages <1 rps); 0 = no compression")
    p.add_argument("--max-requests", type=int, default=1500)
    p.add_argument("--max-model-len", type=int, default=8192, help="clip trace token counts to fit this context")


def add_backend_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", choices=["sim", "vllm"], default="sim")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="HF model id (vllm backend)")
    p.add_argument("--sim-model", choices=list(MODELS), default="llama-3.1-8b")
    p.add_argument("--sim-gpu", choices=list(GPUS), default="H100-80GB")
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--port", type=int, default=8011)
    p.add_argument("--ttft-slo", type=float, default=2.0, help="seconds")
    p.add_argument("--tpot-slo", type=float, default=0.10, help="seconds per output token")


def build_workload(a):
    path = Path(a.trace)
    if not path.exists():
        print(f"downloading {DEFAULT_TRACE_URL} -> {path}", file=sys.stderr)
        download_trace(path)
    df = load_trace(path, model=a.trace_model, log_type=a.log_type)
    return make_workload(df, a.start, a.duration, a.target_rps or None, a.max_requests, a.max_model_len)


def build_backend(a):
    if a.backend == "sim":
        return SimBackend.from_names(a.sim_gpu, a.sim_model, a.num_gpus)
    from .backends import VLLMBackend

    return VLLMBackend(a.model, port=a.port, num_gpus=a.num_gpus)


def configure_lm(model: str):
    import dspy

    # No temperature: current Claude models reject sampling parameters.
    lm = dspy.LM(model, max_tokens=16000)
    dspy.configure(lm=lm)
    return lm


def cmd_stats(a):
    print(json.dumps(summarize(build_workload(a)), indent=2))


def cmd_trial(a):
    cfg = VLLMConfig.from_dict(json.loads(a.config))
    backend, w = build_backend(a), build_workload(a)
    errs = cfg.validate(backend.num_gpus)
    if errs:
        sys.exit("invalid config: " + "; ".join(errs))
    print(json.dumps(backend.run(cfg, w, SLO(a.ttft_slo, a.tpot_slo)), indent=2))


def cmd_tune(a):
    from .agent import TuningEnv, VLLMTuner

    configure_lm(a.lm)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    w = build_workload(a)
    base = VLLMConfig(max_model_len=a.max_model_len)
    env = TuningEnv(w, build_backend(a), SLO(a.ttft_slo, a.tpot_slo), a.budget, base=base,
                    allow_fp8_kv=not a.no_fp8_kv, log_path=out / "trials.jsonl")
    tuner = VLLMTuner()
    if a.program:
        tuner.load(a.program)
    pred = tuner(env=env)

    if not pred.best_config:
        sys.exit("no successful trial; see trials.jsonl and server logs")
    best = VLLMConfig(**pred.best_config)
    serve_cmd = " ".join(["vllm", "serve", a.model, *best.to_cli_args()])
    report = {
        "workload": summarize(w),
        "baseline_goodput_rps": pred.baseline_goodput,
        "best_goodput_rps": pred.best_goodput,
        "best_config": pred.best_config,
        "serve_command": serve_cmd,
        "rationale": pred.rationale,
        "trials": pred.num_trials,
    }
    (out / "best_config.json").write_text(json.dumps(report, indent=2))
    print(pred.history)
    print(f"\nbaseline goodput {pred.baseline_goodput:.3f} rps -> best {pred.best_goodput:.3f} rps")
    print(f"best config: {json.dumps(best.diff(VLLMConfig()))}")
    print(f"\n{serve_cmd}\n\nrationale: {pred.rationale}\nwritten to {out}/")


def cmd_compile(a):
    import dspy

    from .compile_agent import compile_tuner, make_task
    from .trace import load_trace

    configure_lm(a.lm)
    path = Path(a.trace)
    if not path.exists():
        download_trace(path)
    df = load_trace(path)
    # Busy hours only: quiet night windows are too sparse to stress the server even after compression.
    hourly = (df["Timestamp"] // 3600).value_counts()
    busy = sorted(hourly[hourly >= 200].index)
    # Vary window position, load level and hardware so the learned instructions generalise.
    tasks = []
    grid = [("H100-80GB", "llama-3.1-8b", 1), ("A100-80GB", "llama-3.1-8b", 1), ("L40S-48GB", "llama-3.1-8b", 1),
            ("H100-80GB", "qwen2.5-14b", 1), ("A100-80GB", "llama-3.1-70b", 4), ("H100-80GB", "llama-3.1-70b", 4)]
    for i in range(a.num_tasks):
        gpu, model, ngpu = grid[i % len(grid)]
        start = float(busy[(i * 37) % len(busy)] * 3600)
        rps = [4, 8, 15, 25][i % 4] * (0.3 if "70b" in model else 1.0)
        w = make_workload(df, start, 3 * 3600, rps, a.max_requests, 8192, seed=i)
        tasks.append(make_task(f"t{i}", w, SimBackend.from_names(gpu, model, ngpu), SLO(a.ttft_slo, a.tpot_slo), a.budget))
    split = max(1, len(tasks) * 2 // 3)
    reflection_lm = dspy.LM(a.reflection_lm or a.lm, max_tokens=32000)
    compiled = compile_tuner(tasks[:split], tasks[split:], reflection_lm, a.max_metric_calls, log_dir=a.out)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    compiled.save(os.path.join(a.out, "tuner.json"))
    print(f"saved optimized program to {a.out}/tuner.json (use with: tune --program {a.out}/tuner.json)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="burstgpt_tuner", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("stats", help="print workload statistics for a trace window")
    add_workload_args(p)
    p.set_defaults(fn=cmd_stats)

    p = sub.add_parser("trial", help="benchmark one config")
    add_workload_args(p)
    add_backend_args(p)
    p.add_argument("--config", default="{}", help="JSON overrides, e.g. '{\"max_num_seqs\": 128}'")
    p.set_defaults(fn=cmd_trial)

    p = sub.add_parser("tune", help="run the DSPy agent")
    add_workload_args(p)
    add_backend_args(p)
    p.add_argument("--lm", default=os.environ.get("TUNER_LM", "anthropic/claude-opus-5-5"), help="any LiteLLM model id")
    p.add_argument("--budget", type=int, default=12, help="benchmark trials, baseline included")
    p.add_argument("--no-fp8-kv", action="store_true", help="forbid kv_cache_dtype=fp8")
    p.add_argument("--program", help="optimized program saved by `compile`")
    p.add_argument("--out", default="runs/latest")
    p.set_defaults(fn=cmd_tune)

    p = sub.add_parser("compile", help="optimize the agent's prompts with GEPA on simulated tasks")
    p.add_argument("--trace", default="data/BurstGPT_1.csv")
    p.add_argument("--lm", default=os.environ.get("TUNER_LM", "anthropic/claude-opus-5-5"))
    p.add_argument("--reflection-lm")
    p.add_argument("--num-tasks", type=int, default=9)
    p.add_argument("--budget", type=int, default=10)
    p.add_argument("--max-requests", type=int, default=800)
    p.add_argument("--max-metric-calls", type=int, default=30)
    p.add_argument("--ttft-slo", type=float, default=2.0)
    p.add_argument("--tpot-slo", type=float, default=0.10)
    p.add_argument("--out", default="runs/compiled")
    p.set_defaults(fn=cmd_compile)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
