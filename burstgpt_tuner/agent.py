"""DSPy ReAct agent that searches vLLM serving configs for a BurstGPT workload.

The LM proposes configs and reasons over measured results; every number it sees comes from a benchmark
run (simulated or real). The returned best config is the best *measured* trial, never the LM's claim.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import dspy

from .backends import Backend
from .config_space import VLLMConfig, describe_search_space
from .metrics import SLO
from .trace import Workload, summarize


@dataclass
class Trial:
    index: int
    config: dict
    result: dict
    hypothesis: str
    wall_s: float

    @property
    def score(self) -> float:
        return self.result.get("goodput_rps", 0.0) if self.result.get("ok") else 0.0


@dataclass
class TuningEnv:
    """Everything one tuning session needs: workload, backend, SLO, trial budget and the trial log."""

    workload: Workload
    backend: Backend
    slo: SLO
    budget: int
    base: VLLMConfig = field(default_factory=VLLMConfig)
    allow_fp8_kv: bool = True
    log_path: Path | None = None
    trials: list[Trial] = field(default_factory=list)
    _cache: dict = field(default_factory=dict)

    @property
    def best(self) -> Trial | None:
        ok = [t for t in self.trials if t.result.get("ok")]
        # Ties on goodput (common once every request meets the SLO) go to the config with more latency headroom.
        def headroom(t: Trial) -> float:
            r = t.result
            return -(r["ttft_s"]["p99"] / self.slo.ttft_s + r["tpot_ms"]["p99"] / 1000 / self.slo.tpot_s)
        return max(ok, key=lambda t: (round(t.score, 2), headroom(t))) if ok else None

    def evaluate(self, overrides: dict, hypothesis: str = "") -> dict:
        try:
            cfg = VLLMConfig.from_dict(overrides, self.base)
        except (ValueError, TypeError) as e:
            return {"ok": False, "error": str(e), "counted_against_budget": False}
        errs = cfg.validate(self.backend.num_gpus, self.allow_fp8_kv)
        if errs:
            return {"ok": False, "error": "invalid config: " + "; ".join(errs), "counted_against_budget": False}
        if cfg.key() in self._cache:
            return {**self._cache[cfg.key()], "note": "duplicate of an earlier trial; cached result, budget not used"}
        if len(self.trials) >= self.budget:
            return {"ok": False, "error": "TRIAL BUDGET EXHAUSTED. Stop calling run_trial and finish."}

        t0 = time.time()
        result = self.backend.run(cfg, self.workload, self.slo)
        trial = Trial(len(self.trials) + 1, asdict(cfg), result, hypothesis, round(time.time() - t0, 1))
        self.trials.append(trial)
        self._cache[cfg.key()] = result
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(asdict(trial)) + "\n")
        best = self.best
        return {
            "trial": trial.index,
            "trials_left": self.budget - len(self.trials),
            "changed_vs_default": cfg.diff(VLLMConfig()),
            **result,
            "best_goodput_so_far": best.score if best else 0.0,
        }

    def history_table(self) -> str:
        if not self.trials:
            return "no trials yet"
        rows = []
        for t in self.trials:
            r = t.result
            diff = VLLMConfig(**t.config).diff(VLLMConfig())
            if r.get("ok"):
                rows.append(f"#{t.index} goodput={r['goodput_rps']} attain={r['slo_attainment']} "
                            f"ttft_p99={r['ttft_s']['p99']}s tpot_p99={r['tpot_ms']['p99']}ms "
                            f"preempt={r.get('preemptions', '?')} | {json.dumps(diff)}")
            else:
                rows.append(f"#{t.index} FAILED {r.get('error', '')[:160]} | {json.dumps(diff)}")
        return "\n".join(rows)


# ReAct tools are bound at construction time, so they read the active env from a thread-local.
_active = threading.local()


def _env() -> TuningEnv:
    env = getattr(_active, "env", None)
    if env is None:
        raise RuntimeError("no active TuningEnv; call VLLMTuner(env=...)")
    return env


def run_trial(config: dict, hypothesis: str) -> str:
    """Launch vLLM with `config` (only the keys you want to change from the defaults; unspecified keys use
    defaults) and replay the workload. `hypothesis` states what you expect and why. Returns goodput (requests/s
    meeting the SLO), SLO attainment, TTFT/TPOT percentiles, preemptions, or a startup error. Expensive."""
    return json.dumps(_env().evaluate(config, hypothesis))


def trial_history() -> str:
    """Return a compact table of all trials run so far, with each config's diff from the defaults."""
    return _env().history_table()


def check_config(config: dict) -> str:
    """Validate a config without running it (free). Returns 'valid' or the list of problems."""
    env = _env()
    try:
        errs = VLLMConfig.from_dict(config, env.base).validate(env.backend.num_gpus, env.allow_fp8_kv)
    except (ValueError, TypeError) as e:
        errs = [str(e)]
    return "valid" if not errs else "; ".join(errs)


class TuneVLLMConfig(dspy.Signature):
    """You are a vLLM performance engineer. Find the vLLM engine configuration that maximizes goodput
    (requests/s that meet BOTH the TTFT and TPOT SLO) when replaying the given BurstGPT workload.

    Each run_trial call is a full server launch plus benchmark, so spend the budget deliberately: reason
    from the workload statistics before the first trial; diagnose every result (TTFT-bound means queueing or
    prefill starvation; TPOT-bound means batches too large or long prefills stalling decodes; preemptions or
    startup errors mean KV memory pressure); change one or two knobs per trial once a good region is found;
    and revisit the best config's neighbours rather than wandering. Goodput cannot exceed the offered request
    rate, so once SLO attainment is ~1.0 only latency headroom (lower p99 TTFT/TPOT) is left to gain. Stop when
    the budget is spent or the last few trials no longer improve on the best."""

    workload_stats: str = dspy.InputField(desc="BurstGPT window statistics: load, burstiness, token lengths")
    deployment: str = dspy.InputField(desc="hardware, model and SLO")
    search_space: str = dspy.InputField(desc="tunable vLLM knobs with legal values")
    baseline: str = dspy.InputField(desc="result of the vLLM-defaults trial, already run")
    trial_budget: int = dspy.InputField(desc="total benchmark trials allowed, baseline included")
    best_config: dict = dspy.OutputField(desc="best config found (changed keys only)")
    rationale: str = dspy.OutputField(desc="why this config wins for this workload, citing measured trials")


def _last_finish_reason() -> str:
    try:
        return f"finish_reason={dspy.settings.lm.history[-1]['response'].finish_reason}"
    except (AttributeError, IndexError, KeyError, TypeError):
        return "finish_reason unknown"


class VLLMTuner(dspy.Module):
    def __init__(self, max_iters: int = 40):
        super().__init__()
        self.react = dspy.ReAct(TuneVLLMConfig, tools=[run_trial, trial_history, check_config], max_iters=max_iters)
        # ReAct's final extract step is a ChainOfThought, whose "reasoning" output field Claude Opus 5.5 refuses
        # (finish_reason=content_filter). The model already thinks natively, so a plain Predict loses nothing.
        self.react.extract = dspy.Predict(self.react.extract.predict.signature.delete("reasoning"))

    def forward(self, env: TuningEnv | Callable[[], TuningEnv]) -> dspy.Prediction:
        # A factory gives each call a fresh trial log (needed when an optimizer re-runs the same example).
        env = env() if callable(env) else env
        _active.env = env
        try:
            baseline = env.evaluate({}, "vLLM defaults (baseline)")
            try:
                out = self.react(
                    workload_stats=json.dumps(summarize(env.workload), indent=1),
                    deployment=f"{env.backend.describe()}. SLO: {env.slo}. "
                               f"fp8 KV cache {'allowed' if env.allow_fp8_kv else 'NOT allowed'}.",
                    search_space=describe_search_space(),
                    baseline=json.dumps(baseline),
                    trial_budget=env.budget,
                )
            except dspy.utils.exceptions.AdapterParseError:
                # The measured trials are what count; don't lose them because the final summary came back empty.
                out = dspy.Prediction(best_config=None, rationale=f"(LM gave no final answer: {_last_finish_reason()})")
        finally:
            _active.env = None
        best = env.best
        return dspy.Prediction(
            best_config=best.config if best else None,
            best_goodput=best.score if best else 0.0,
            baseline_goodput=env.trials[0].score if env.trials else 0.0,
            num_trials=len(env.trials),
            agent_claimed_config=out.best_config,
            rationale=out.rationale,
            history=env.history_table(),
            trajectory=getattr(out, "trajectory", None),
        )
