"""Optimize the tuner's own prompts with DSPy GEPA, using the simulator as a cheap training environment.

Training tasks are different BurstGPT windows / load levels / hardware. Each task's score is the agent's best
goodput divided by a reference found by random search with a much larger budget, so 1.0 means "matched a
brute-force search using far fewer trials". The textual feedback tells GEPA's reflection LM what went wrong.
"""

from __future__ import annotations

import functools
import random

import dspy

from .agent import TuningEnv, VLLMTuner
from .backends import SimBackend
from .config_space import SEARCH_SPACE, VLLMConfig
from .metrics import SLO
from .trace import Workload


def random_config(rng: random.Random) -> dict:
    d = {}
    for name, (allowed, _) in SEARCH_SPACE.items():
        if name == "max_model_len":
            continue
        if isinstance(allowed, list):
            d[name] = rng.choice(allowed)
        elif isinstance(allowed[0], float):
            d[name] = round(rng.uniform(*allowed), 2)
        else:  # log-uniform over powers of two for sizes
            lo, hi = max(allowed[0], 1).bit_length() - 1, allowed[1].bit_length() - 1
            d[name] = 0 if name == "long_prefill_token_threshold" and rng.random() < 0.5 else 2 ** rng.randint(lo, hi)
    return d


@functools.lru_cache(maxsize=None)
def _reference(task_id: str) -> tuple[float, dict]:
    env = _TASKS[task_id]()
    env.budget = 10_000
    rng = random.Random(0)
    env.evaluate({})
    for _ in range(200):
        env.evaluate(random_config(rng))
    return env.best.score, VLLMConfig(**env.best.config).diff(VLLMConfig())


class TaskFactory:
    """Builds a fresh TuningEnv per call; its repr is what GEPA's reflection LM sees as the task input."""

    def __init__(self, workload: Workload, backend: SimBackend, slo: SLO, budget: int):
        self.workload, self.backend, self.slo, self.budget = workload, backend, slo, budget

    def __call__(self) -> TuningEnv:
        return TuningEnv(self.workload, self.backend, self.slo, self.budget)

    def __repr__(self) -> str:
        return f"{self.workload.name} on {self.backend.describe()}, SLO {self.slo}, budget {self.budget}"


_TASKS: dict[str, TaskFactory] = {}


def make_task(task_id: str, workload: Workload, backend: SimBackend, slo: SLO, budget: int) -> dspy.Example:
    _TASKS[task_id] = TaskFactory(workload, backend, slo, budget)
    return dspy.Example(task_id=task_id, env=_TASKS[task_id]).with_inputs("env")


def metric(gold: dspy.Example, pred: dspy.Prediction, trace=None, pred_name=None, pred_trace=None):
    ref_score, ref_cfg = _reference(gold.task_id)
    score = min(1.0, pred.best_goodput / ref_score) if ref_score > 0 else 0.0
    feedback = (
        f"Agent best goodput {pred.best_goodput:.3f} rps vs baseline {pred.baseline_goodput:.3f} and reference "
        f"{ref_score:.3f} (reference config diff from defaults: {ref_cfg}). Trials used: {pred.num_trials}.\n"
        f"Trial history:\n{pred.history}"
    )
    return dspy.Prediction(score=score, feedback=feedback)


def compile_tuner(trainset: list[dspy.Example], valset: list[dspy.Example], reflection_lm: dspy.LM,
                  max_metric_calls: int = 40, log_dir: str | None = None) -> VLLMTuner:
    gepa = dspy.GEPA(metric=metric, reflection_lm=reflection_lm, max_metric_calls=max_metric_calls,
                     num_threads=1, track_stats=True, log_dir=log_dir)
    return gepa.compile(VLLMTuner(), trainset=trainset, valset=valset)
