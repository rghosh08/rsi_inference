"""Load BurstGPT traces (https://github.com/HPMLL/BurstGPT) and turn a window into a replayable workload."""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# BurstGPT_1.csv lives in the repo; the larger files are on the v2.0 release page.
DEFAULT_TRACE_URL = "https://raw.githubusercontent.com/HPMLL/BurstGPT/main/data/BurstGPT_1.csv"


@dataclass
class Request:
    arrival: float  # seconds since workload start
    prompt_tokens: int
    output_tokens: int


@dataclass
class Workload:
    name: str
    requests: list[Request]
    duration: float  # span of the arrival window, seconds
    meta: dict = field(default_factory=dict)

    @property
    def offered_rps(self) -> float:
        return len(self.requests) / max(self.duration, 1e-9)


def download_trace(dest: str | Path, url: str = DEFAULT_TRACE_URL) -> Path:
    dest = Path(dest)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
    return dest


def load_trace(
    path: str | Path,
    model: str | None = None,
    log_type: str | None = None,
    drop_failed: bool = True,
) -> pd.DataFrame:
    """Read a BurstGPT CSV. `model` is 'ChatGPT' or 'GPT-4'; `log_type` is 'Conversation log' or 'API log'."""
    df = pd.read_csv(path)
    if drop_failed:
        df = df[df["Response tokens"] > 0]
    if model:
        df = df[df["Model"] == model]
    if log_type:
        df = df[df["Log Type"] == log_type]
    df = df[df["Request tokens"] > 0]
    return df.sort_values("Timestamp").reset_index(drop=True)


def make_workload(
    df: pd.DataFrame,
    start: float = 0.0,
    duration: float = 3600.0,
    target_rps: float | None = None,
    max_requests: int | None = None,
    max_model_len: int = 8192,
    seed: int = 0,
) -> Workload:
    """Cut [start, start+duration) out of the trace and optionally time-compress it to `target_rps`.

    Compression keeps the relative burst shape of the trace (it rescales inter-arrival times uniformly)
    rather than replacing it with a Poisson process. Token counts are clipped to fit `max_model_len`.
    """
    rng = np.random.default_rng(seed)
    win = df[(df["Timestamp"] >= start) & (df["Timestamp"] < start + duration)]
    if win.empty:
        raise ValueError(f"no requests in window start={start} duration={duration}")

    # Timestamps have 1s resolution; jitter within the second so same-second requests don't land at once.
    arrivals = win["Timestamp"].to_numpy(dtype=float) - start + rng.uniform(0, 1, len(win))
    order = np.argsort(arrivals)
    arrivals = arrivals[order]
    prompts = win["Request tokens"].to_numpy()[order]
    outputs = win["Response tokens"].to_numpy()[order]

    natural_rps = len(win) / duration
    span = duration
    if target_rps:
        factor = natural_rps / target_rps
        arrivals = arrivals * factor
        span = duration * factor

    if max_requests and len(arrivals) > max_requests:
        arrivals, prompts, outputs = arrivals[:max_requests], prompts[:max_requests], outputs[:max_requests]
        span = float(arrivals[-1]) + 1e-3

    reqs = []
    for a, p, o in zip(arrivals, prompts, outputs):
        p = int(min(p, max_model_len - 16))
        o = int(max(1, min(o, max_model_len - p)))
        reqs.append(Request(float(a), p, o))

    name = f"burstgpt[{start:.0f}s+{duration:.0f}s" + (f"@{target_rps}rps" if target_rps else "") + "]"
    meta = {"start": start, "window_s": duration, "natural_rps": natural_rps, "target_rps": target_rps}
    return Workload(name=name, requests=reqs, duration=span, meta=meta)


def summarize(w: Workload) -> dict:
    """Workload statistics the agent reasons over: load level, burstiness, token-length shape."""
    a = np.array([r.arrival for r in w.requests])
    p = np.array([r.prompt_tokens for r in w.requests])
    o = np.array([r.output_tokens for r in w.requests])
    gaps = np.diff(a) if len(a) > 1 else np.array([0.0])
    bins = np.bincount((a // 10).astype(int)) / 10.0 if len(a) else np.array([0.0])

    def pct(x):
        return {k: int(np.percentile(x, q)) for k, q in (("p50", 50), ("p90", 90), ("p99", 99))} | {"max": int(x.max())}

    return {
        "name": w.name,
        "num_requests": len(w.requests),
        "duration_s": round(w.duration, 1),
        "mean_rps": round(w.offered_rps, 3),
        "peak_rps_10s": round(float(bins.max()), 3),
        "peak_to_mean": round(float(bins.max()) / max(w.offered_rps, 1e-9), 2),
        "interarrival_cv": round(float(gaps.std() / max(gaps.mean(), 1e-9)), 2),  # 1.0 = Poisson, >1 = bursty
        "prompt_tokens": pct(p),
        "output_tokens": pct(o),
        "mean_prompt_to_output": round(float(p.mean() / max(o.mean(), 1e-9)), 2),
        "frac_prompt_over_2k": round(float((p > 2048).mean()), 3),
        "offered_tokens_per_s": round(float((p.sum() + o.sum()) / max(w.duration, 1e-9)), 1),
    }
