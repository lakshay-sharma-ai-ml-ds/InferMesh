"""
Latency Sweep Benchmark
========================
Measures TTFT and E2E latency at varying concurrency levels.
Produces a latency-vs-concurrency curve to find the knee point.
"""

from __future__ import annotations

import asyncio
import time

import httpx
import structlog

from benchmarks.runners.base_runner import BaseBenchmarkRunner, generate_prompt

logger = structlog.get_logger("benchmarks.latency")


class LatencyBenchmarkRunner(BaseBenchmarkRunner):
    """
    Latency sweep benchmark.

    Tests TTFT and E2E latency across multiple concurrency levels:
    [1, 2, 4, 8, 16, 32, concurrency]

    Results show:
    - Latency-concurrency curve
    - Knee point (where latency starts degrading rapidly)
    - P50/P95/P99 at each concurrency level
    """

    scenario_name = "latency_sweep"

    async def run(self) -> dict:
        self._start_time = time.monotonic()

        async with httpx.AsyncClient() as client:
            await self._warmup(client)

            # Sweep concurrency levels
            sweep_results = []
            levels = [1, 2, 4, 8, min(16, self.concurrency), self.concurrency]
            levels = sorted(set(levels))

            for level in levels:
                logger.info(f"Latency sweep: concurrency={level}")
                level_results = await self._run_at_concurrency(client, level, duration_s=max(10, self.duration_s // len(levels)))
                sweep_results.append({
                    "concurrency": level,
                    **level_results,
                })

        # Aggregate from last (full concurrency) run
        base = self._build_base_results()
        base["sweep_results"] = sweep_results
        base["knee_point_concurrency"] = self._find_knee_point(sweep_results)
        return base

    async def _run_at_concurrency(
        self, client: httpx.AsyncClient, concurrency: int, duration_s: int
    ) -> dict:
        """Run requests at fixed concurrency for duration_s seconds."""
        ttft_samples = []
        e2e_samples = []
        errors = 0
        requests = 0

        deadline = time.monotonic() + duration_s
        semaphore = asyncio.Semaphore(concurrency)

        async def one_request():
            nonlocal errors, requests
            async with semaphore:
                prompt, _ = generate_prompt("medium")
                result = await self._send_inference(
                    client, prompt, max_new_tokens=128
                )
                requests += 1
                if result["success"]:
                    ttft_samples.append(result["ttft_ms"])
                    e2e_samples.append(result["e2e_ms"])
                    self.ttft_samples.append(result["ttft_ms"])
                    self.e2e_samples.append(result["e2e_ms"])
                    self.total_tokens += 128
                else:
                    errors += 1
                self.total_requests += 1

        tasks = []
        while time.monotonic() < deadline:
            task = asyncio.create_task(one_request())
            tasks.append(task)
            if len(tasks) >= concurrency * 3:
                done, tasks_set = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )
                tasks = list(tasks_set)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        return {
            "p50_ttft_ms": round(self._compute_percentile(ttft_samples, 50), 2),
            "p95_ttft_ms": round(self._compute_percentile(ttft_samples, 95), 2),
            "p99_ttft_ms": round(self._compute_percentile(ttft_samples, 99), 2),
            "p50_e2e_ms": round(self._compute_percentile(e2e_samples, 50), 2),
            "p99_e2e_ms": round(self._compute_percentile(e2e_samples, 99), 2),
            "rps": round(requests / max(1, duration_s), 2),
            "error_rate": round(errors / max(1, requests), 4),
        }

    def _find_knee_point(self, sweep: list[dict]) -> int:
        """
        Find the knee point in the latency-concurrency curve.
        Uses the maximum ratio method (largest relative latency jump).
        """
        if len(sweep) < 2:
            return sweep[-1]["concurrency"] if sweep else 1

        # Guard: filter out None or zero latency points
        valid = [
            s for s in sweep
            if s.get("p95_ttft_ms") and s["p95_ttft_ms"] > 0
        ]
        if len(valid) < 2:
            return sweep[-1]["concurrency"]

        latencies = [s["p95_ttft_ms"] for s in valid]
        concurrencies = [s["concurrency"] for s in valid]

        # Find the point of maximum relative latency increase
        best_ratio = 0.0
        knee_idx = 0
        for i in range(1, len(latencies)):
            prev = latencies[i - 1]
            if prev > 0:
                ratio = latencies[i] / prev
                if ratio > best_ratio:
                    best_ratio = ratio
                    knee_idx = i

        return concurrencies[max(0, knee_idx - 1)]
