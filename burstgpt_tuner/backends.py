"""Benchmark backends: run one VLLMConfig against a Workload and return a metrics dict.

- SimBackend: the analytical simulator (no GPU, seconds per trial).
- VLLMBackend: launches `vllm serve` with the config and replays the trace over the OpenAI API.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .config_space import VLLMConfig
from .metrics import SLO, RequestRecord, summarize_records
from .simulator import GPUS, MODELS, GPUProfile, ModelProfile, StartupError, simulate
from .trace import Workload


class Backend:
    num_gpus: int = 1

    def describe(self) -> str:
        raise NotImplementedError

    def run(self, cfg: VLLMConfig, w: Workload, slo: SLO) -> dict:
        raise NotImplementedError


@dataclass
class SimBackend(Backend):
    gpu: GPUProfile
    model: ModelProfile

    @classmethod
    def from_names(cls, gpu: str = "H100-80GB", model: str = "llama-3.1-8b", num_gpus: int = 1) -> "SimBackend":
        g = GPUS[gpu]
        return cls(GPUProfile(g.name, g.mem_gb, g.tflops, g.hbm_tbps, num_gpus), MODELS[model])

    @property
    def num_gpus(self) -> int:
        return self.gpu.num_gpus

    def describe(self) -> str:
        return (f"[SIMULATED] {self.gpu.num_gpus}x {self.gpu.name}, model {self.model.name} "
                f"({self.model.params_b}B params, bf16 weights)")

    def run(self, cfg: VLLMConfig, w: Workload, slo: SLO) -> dict:
        try:
            recs, info = simulate(cfg, w, self.gpu, self.model)
        except StartupError as e:
            return {"ok": False, "error": f"server failed to start: {e}"}
        return {"ok": True, **summarize_records(recs, slo, w.duration), **info}


class VLLMBackend(Backend):
    """Real benchmark. One server launch per trial, so each trial costs model-load time plus the replay."""

    def __init__(self, model: str, port: int = 8011, num_gpus: int | None = None, log_dir: str = "runs/server_logs",
                 startup_timeout_s: float = 900, request_timeout_s: float = 600, extra_serve_args: list[str] | None = None):
        if not shutil.which("vllm"):
            raise RuntimeError("`vllm` CLI not found on PATH; install vllm on the GPU host or use --backend sim")
        self.model = model
        self.port = port
        self.num_gpus = num_gpus or _count_gpus()
        self.log_dir = Path(log_dir)
        self.startup_timeout_s = startup_timeout_s
        self.request_timeout_s = request_timeout_s
        self.extra_serve_args = extra_serve_args or []

    def describe(self) -> str:
        return f"{self.num_gpus}x {_gpu_name()}, model {self.model}"

    def run(self, cfg: VLLMConfig, w: Workload, slo: SLO) -> dict:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"{cfg.key()}.log"
        cmd = ["vllm", "serve", self.model, "--port", str(self.port),
               *cfg.to_cli_args(), *self.extra_serve_args]
        with open(log_path, "w") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            err = self._wait_ready(proc)
            if err:
                return {"ok": False, "error": f"server failed to start: {err}\n{_tail(log_path)}"}
            asyncio.run(self._replay([RequestRecord(1, 8, 4)], warmup=True))
            recs = asyncio.run(self._replay([RequestRecord(r.arrival, r.prompt_tokens, r.output_tokens) for r in w.requests]))
            return {"ok": True, **summarize_records(recs, slo, w.duration)}
        finally:
            _kill(proc)

    def _wait_ready(self, proc: subprocess.Popen) -> str | None:
        deadline = time.time() + self.startup_timeout_s
        while time.time() < deadline:
            if proc.poll() is not None:
                return f"process exited with code {proc.returncode}"
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                    if r.status == 200:
                        return None
            except OSError:
                pass
            time.sleep(2)
        return f"not healthy after {self.startup_timeout_s}s"

    async def _replay(self, recs: list[RequestRecord], warmup: bool = False) -> list[RequestRecord]:
        import aiohttp

        url = f"http://127.0.0.1:{self.port}/v1/completions"
        rng = random.Random(0)
        timeout = aiohttp.ClientTimeout(total=self.request_timeout_s)
        async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as session:
            t0 = time.perf_counter()

            async def one(rec: RequestRecord):
                delay = 0 if warmup else rec.arrival - (time.perf_counter() - t0)
                if delay > 0:
                    await asyncio.sleep(delay)
                # Random token ids: exact prompt length, no accidental prefix sharing between requests.
                body = {
                    "model": self.model,
                    "prompt": [rng.randint(1000, 30000) for _ in range(rec.prompt_tokens)],
                    "max_tokens": rec.output_tokens,
                    "ignore_eos": True,
                    "temperature": 0.0,
                    "stream": True,
                }
                # vllm-metal rejects min_tokens; ignore_eos alone already runs to max_tokens there.
                if sys.platform != "darwin":
                    body["min_tokens"] = rec.output_tokens
                if warmup:
                    rec.arrival = time.perf_counter() - t0
                try:
                    async with session.post(url, json=body) as resp:
                        if resp.status != 200:
                            rec.error = f"HTTP {resp.status}: {(await resp.text())[:200]}"
                            return
                        async for raw in resp.content:
                            line = raw.decode().strip()
                            if not line.startswith("data:") or line == "data: [DONE]":
                                continue
                            chunk = json.loads(line[5:])
                            if chunk.get("choices") and chunk["choices"][0].get("text") is not None:
                                if rec.first_token is None:
                                    rec.first_token = time.perf_counter() - t0
                        rec.finish = time.perf_counter() - t0
                        rec.generated = rec.output_tokens  # exact, because of ignore_eos (+ min_tokens off macOS)
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    rec.error = f"{type(e).__name__}: {e}"

            await asyncio.gather(*(one(r) for r in recs))
        return recs


def _kill(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    time.sleep(5)  # let the GPU memory be released before the next launch


def _tail(path: Path, n: int = 25) -> str:
    return "".join(path.read_text(errors="replace").splitlines(keepends=True)[-n:])


def _count_gpus() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
        return max(1, len(out.strip().splitlines()))
    except (OSError, subprocess.CalledProcessError):
        return 1


def _gpu_name() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True, check=True).stdout.splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        return "unknown GPU"
