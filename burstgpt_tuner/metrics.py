"""Per-request records -> latency percentiles, SLO attainment and goodput (the score the agent maximizes)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SLO:
    ttft_s: float = 2.0  # time to first token
    tpot_s: float = 0.10  # mean time per output token after the first

    def __str__(self) -> str:
        return f"TTFT <= {self.ttft_s}s and TPOT <= {self.tpot_s * 1000:.0f}ms"


@dataclass
class RequestRecord:
    arrival: float
    prompt_tokens: int
    output_tokens: int
    first_token: float | None = None
    finish: float | None = None
    generated: int = 0
    error: str | None = None

    @property
    def ttft(self) -> float | None:
        return None if self.first_token is None else self.first_token - self.arrival

    @property
    def tpot(self) -> float | None:
        if self.finish is None or self.first_token is None:
            return None
        return (self.finish - self.first_token) / max(self.generated - 1, 1)

    def meets(self, slo: SLO) -> bool:
        return self.error is None and self.finish is not None and self.ttft <= slo.ttft_s and self.tpot <= slo.tpot_s


def summarize_records(records: list[RequestRecord], slo: SLO, offered_duration: float) -> dict:
    done = [r for r in records if r.error is None and r.finish is not None]
    errors = len(records) - len(done)
    met = sum(r.meets(slo) for r in records)
    makespan = max([r.finish for r in done], default=offered_duration)
    span = max(offered_duration, makespan, 1e-9)

    def pct(vals, scale=1.0, nd=3):
        if not vals:
            return {"p50": None, "p90": None, "p99": None}
        v = np.array(vals) * scale
        return {k: round(float(np.percentile(v, q)), nd) for k, q in (("p50", 50), ("p90", 90), ("p99", 99))}

    return {
        "num_requests": len(records),
        "completed": len(done),
        "errors": errors,
        "slo_attainment": round(met / max(len(records), 1), 4),
        "goodput_rps": round(met / span, 4),
        "throughput_rps": round(len(done) / span, 4),
        "output_tok_s": round(sum(r.generated for r in done) / span, 1),
        "ttft_s": pct([r.ttft for r in done]),
        "tpot_ms": pct([r.tpot for r in done], 1000, 1),
        "e2e_s": pct([r.finish - r.arrival for r in done]),
        "makespan_s": round(makespan, 1),
    }
