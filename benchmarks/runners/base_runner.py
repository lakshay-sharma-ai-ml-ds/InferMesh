"""
Benchmark Base Runner
======================
Abstract base class for all InferMesh benchmark scenarios.
Provides common HTTP client setup, load generation, and metrics collection.
"""

from __future__ import annotations

import abc
import asyncio
import random
import statistics
import time
from typing import Optional

import httpx
import structlog

logger = structlog.get_logger("benchmarks.base")

# ---------------------------------------------------------------------------
# Synthetic prompt generator
# ---------------------------------------------------------------------------

PROMPT_TEMPLATES = [
    "Explain the concept of {topic} in detail, covering history, theory, and practical applications.",
    "Write a comprehensive analysis of {topic}.",
    "What are the key differences between {topic} and related alternatives?",
    "Summarize the main points about {topic} for a technical audience.",
    "Describe the architecture of {topic} system.",
]

TOPICS = [
    "distributed systems", "neural network optimization", "GPU memory management",
    "transformer attention mechanisms", "KV-cache efficiency", "batch scheduling",
    "NUMA topology", "PCIe bandwidth", "tensor parallelism", "pipeline parallelism",
    "speculative decoding", "flash attention", "continuous batching", "vLLM",
    "inference optimization", "model quantization", "RLHF training", "LoRA adapters",
]


def generate_prompt(length: str = "medium") -> tuple[str, int]:
    """Generate a synthetic prompt. Returns (prompt, estimated_tokens)."""
    template = random.choice(PROMPT_TEMPLATES)
    topic = random.choice(TOPICS)
    base = template.format(topic=topic)

    if length == "short":
        text = base[:100]
    elif length == "long":
        text = base + " " + " ".join(random.choices(TOPICS, k=20))
    else:
        text = base

    estimated_tokens = max(1, len(text) // 4)
    return text, estimated_tokens


# ---------------------------------------------------------------------------
# Base runner
# ---------------------------------------------------------------------------

class BaseBenchmarkRunner(abc.ABC):
    """Abstract base for benchmark scenarios."""

    scenario_name: str = "base"

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        duration_s: int = 30,
        concurrency: int = 10,
        num_workers: int = 4,
    ):
        self.base_url = base_url
        self.duration_s = duration_s
        self.concurrency = concurrency
        self.num_workers = num_workers

        # Metrics storage
        self.ttft_samples: list[float] = []
        self.e2e_samples: list[float] = []
        self.errors: list[str] = []
        self.sla_violations: int = 0
        self.total_requests: int = 0
        self.total_tokens: int = 0
        self._start_time: float = 0.0

    @abc.abstractmethod
    async def run(self) -> dict:
        """Execute the benchmark scenario and return results dict."""
        ...

    # -----------------------------------------------------------------------
    # HTTP helpers
    # -----------------------------------------------------------------------

    async def _send_inference(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        max_new_tokens: int = 256,
        priority: str = "NORMAL",
        tenant_id: str = "bench",
    ) -> dict:
        """Send a single inference request and return timing metrics."""
        t_start = time.monotonic()
        try:
            resp = await client.post(
                f"{self.base_url}/v1/infer",
                json={
                    "prompt": prompt,
                    "max_new_tokens": max_new_tokens,
                    "priority": priority,
                    "tenant_id": tenant_id,
                },
                timeout=60.0,
            )
            elapsed_ms = (time.monotonic() - t_start) * 1000

            if resp.status_code == 200:
                data = resp.json()
                # Use `or` not just .get() — API may return null (None), not a missing key
                ttft_ms = data.get("ttft_ms") or (elapsed_ms * 0.3)
                tokens = data.get("generated_tokens") or max_new_tokens
                return {
                    "success": True,
                    "e2e_ms": elapsed_ms,
                    "ttft_ms": float(ttft_ms),
                    "tokens": int(tokens),
                    "sla_violated": data.get("sla_violated") or False,
                    "worker_id": data.get("worker_id") or "unknown",
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {resp.status_code}",
                    "e2e_ms": elapsed_ms,
                }
        except Exception as exc:
            elapsed_ms = (time.monotonic() - t_start) * 1000
            return {
                "success": False,
                "error": str(exc),
                "e2e_ms": elapsed_ms,
            }

    async def _warmup(self, client: httpx.AsyncClient, n: int = 5) -> None:
        """Send warmup requests to populate KV-cache and stabilize metrics."""
        logger.info("Warming up", requests=n)
        tasks = [
            self._send_inference(client, generate_prompt("short")[0], max_new_tokens=32)
            for _ in range(n)
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.sleep(1.0)

    # -----------------------------------------------------------------------
    # Metrics computation
    # -----------------------------------------------------------------------

    def _compute_percentile(self, data: list[float], pct: float) -> float:
        # Filter out None values before sorting — API may return null fields
        clean = [x for x in data if x is not None and isinstance(x, (int, float))]
        if not clean:
            return 0.0
        sorted_data = sorted(clean)
        idx = int(len(sorted_data) * pct / 100)
        return sorted_data[min(idx, len(sorted_data) - 1)]

    def _build_base_results(self) -> dict:
        """Build the standard results dict from collected metrics."""
        elapsed = time.monotonic() - self._start_time
        error_count = len(self.errors)
        success_count = self.total_requests - error_count

        ttft = self.ttft_samples
        e2e = self.e2e_samples

        return {
            "scenario": self.scenario_name,
            "duration_s": round(elapsed, 2),
            "total_requests": self.total_requests,
            "successful_requests": success_count,
            "failed_requests": error_count,
            "error_rate": round(error_count / max(1, self.total_requests), 4),
            "sla_violations": self.sla_violations,
            "sla_violation_rate": round(self.sla_violations / max(1, self.total_requests), 4),
            # TTFT
            "p50_ttft_ms": round(self._compute_percentile(ttft, 50), 2),
            "p75_ttft_ms": round(self._compute_percentile(ttft, 75), 2),
            "p95_ttft_ms": round(self._compute_percentile(ttft, 95), 2),
            "p99_ttft_ms": round(self._compute_percentile(ttft, 99), 2),
            "mean_ttft_ms": round(statistics.mean(ttft) if ttft else 0, 2),
            "std_ttft_ms": round(statistics.stdev(ttft) if len(ttft) > 1 else 0, 2),
            # E2E latency
            "p50_e2e_ms": round(self._compute_percentile(e2e, 50), 2),
            "p95_e2e_ms": round(self._compute_percentile(e2e, 95), 2),
            "p99_e2e_ms": round(self._compute_percentile(e2e, 99), 2),
            "mean_e2e_ms": round(statistics.mean(e2e) if e2e else 0, 2),
            # Throughput
            "throughput_rps": round(success_count / max(1, elapsed), 3),
            "throughput_tokens_per_sec": round(self.total_tokens / max(1, elapsed), 2),
        }
