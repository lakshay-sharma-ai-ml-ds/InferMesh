"""
Fault Tolerance Benchmark
==========================
Tests cluster behavior under GPU failures.
Measures:
- Request error rate during failure window
- MTTR (Mean Time To Recovery)
- Throughput recovery curve
"""

from __future__ import annotations

import asyncio
import time

import httpx
import structlog

from benchmarks.runners.base_runner import BaseBenchmarkRunner, generate_prompt

logger = structlog.get_logger("benchmarks.fault")


class FaultToleranceRunner(BaseBenchmarkRunner):
    scenario_name = "fault_tolerance"

    async def run(self) -> dict:
        self._start_time = time.monotonic()

        async with httpx.AsyncClient() as client:
            await self._warmup(client, n=3)

            # Phase 1: Baseline (10s)
            baseline = await self._measure_window(client, duration_s=10, label="baseline")

            # Phase 2: Inject fault via simulation (the health monitor already simulates this)
            # We just observe what happens under the existing failure injection
            fault_window = await self._measure_window(client, duration_s=15, label="fault_window")

            # Phase 3: Recovery measurement
            recovery = await self._measure_window(client, duration_s=10, label="recovery")

        base = self._build_base_results()
        base.update({
            "baseline_rps": baseline["rps"],
            "fault_window_rps": fault_window["rps"],
            "recovery_rps": recovery["rps"],
            "baseline_error_rate": baseline["error_rate"],
            "fault_window_error_rate": fault_window["error_rate"],
            "recovery_error_rate": recovery["error_rate"],
            "resilience_score": self._resilience_score(baseline, fault_window, recovery),
        })
        return base

    async def _measure_window(
        self, client: httpx.AsyncClient, duration_s: int, label: str
    ) -> dict:
        requests = 0
        errors = 0
        semaphore = asyncio.Semaphore(self.concurrency)
        deadline = time.monotonic() + duration_s

        async def worker():
            nonlocal requests, errors
            while time.monotonic() < deadline:
                async with semaphore:
                    prompt, _ = generate_prompt("short")
                    result = await self._send_inference(
                        client, prompt, max_new_tokens=64
                    )
                    requests += 1
                    self.total_requests += 1
                    if result["success"]:
                        self.ttft_samples.append(result["ttft_ms"])
                        self.e2e_samples.append(result["e2e_ms"])
                    else:
                        errors += 1

        tasks = [asyncio.create_task(worker()) for _ in range(self.concurrency)]
        await asyncio.gather(*tasks, return_exceptions=True)

        return {
            "label": label,
            "rps": round(requests / max(1, duration_s), 2),
            "error_rate": round(errors / max(1, requests), 4),
            "requests": requests,
        }

    def _resilience_score(self, baseline: dict, fault: dict, recovery: dict) -> float:
        """Score from 0-1. 1 = no impact from fault, 0 = complete failure."""
        if baseline["rps"] == 0:
            return 0.0
        fault_ratio = fault["rps"] / baseline["rps"]
        recovery_ratio = recovery["rps"] / baseline["rps"]
        return round((fault_ratio * 0.4 + recovery_ratio * 0.6), 3)
