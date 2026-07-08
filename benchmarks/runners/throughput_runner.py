"""
Throughput Benchmark
=====================
Finds maximum sustainable throughput (RPS and tokens/sec).
Uses an open-loop load generator to saturate the system and
measures the point at which error rate exceeds 1%.
"""

from __future__ import annotations

import asyncio
import time

import httpx
import structlog

from benchmarks.runners.base_runner import BaseBenchmarkRunner, generate_prompt

logger = structlog.get_logger("benchmarks.throughput")


class ThroughputBenchmarkRunner(BaseBenchmarkRunner):
    """
    Maximum throughput benchmark.

    Increases request rate until error rate > 1%, then backs off.
    Reports sustainable RPS, tokens/sec, and GPU utilization.
    """

    scenario_name = "throughput_max"

    async def run(self) -> dict:
        self._start_time = time.monotonic()

        async with httpx.AsyncClient() as client:
            await self._warmup(client)

            # Binary search for maximum sustainable throughput
            results = await self._throughput_sweep(client)

        base = self._build_base_results()
        base.update(results)
        return base

    async def _throughput_sweep(self, client: httpx.AsyncClient) -> dict:
        """
        Ramp up concurrency until error rate spikes.
        Records max sustainable throughput before degradation.
        """
        concurrency_levels = [4, 8, 16, 32, 48, 64, 96, 128]
        max_rps = 0
        max_tps = 0
        max_concurrency = 0
        sweep_points = []

        for concurrency in concurrency_levels:
            rps, tps, error_rate = await self._measure_at_concurrency(
                client, concurrency, window_s=max(8, self.duration_s // len(concurrency_levels))
            )
            sweep_points.append({
                "concurrency": concurrency,
                "rps": rps,
                "tokens_per_sec": tps,
                "error_rate": error_rate,
            })
            logger.info(
                "Throughput point",
                concurrency=concurrency,
                rps=f"{rps:.1f}",
                tps=f"{tps:.1f}",
                error_rate=f"{error_rate:.3f}",
            )

            if error_rate < 0.01:
                max_rps = rps
                max_tps = tps
                max_concurrency = concurrency
            else:
                logger.info("Error rate exceeded 1%, stopping ramp", concurrency=concurrency)
                break

        return {
            "throughput_rps": max_rps,
            "throughput_tokens_per_sec": max_tps,
            "max_sustainable_concurrency": max_concurrency,
            "sweep_points": sweep_points,
        }

    async def _measure_at_concurrency(
        self,
        client: httpx.AsyncClient,
        concurrency: int,
        window_s: int,
    ) -> tuple[float, float, float]:
        """Measure throughput at fixed concurrency."""
        requests = 0
        errors = 0
        tokens = 0
        semaphore = asyncio.Semaphore(concurrency)
        deadline = time.monotonic() + window_s

        async def worker():
            nonlocal requests, errors, tokens
            while time.monotonic() < deadline:
                async with semaphore:
                    prompt, _ = generate_prompt("medium")
                    result = await self._send_inference(
                        client, prompt, max_new_tokens=256
                    )
                    requests += 1
                    self.total_requests += 1
                    if result["success"]:
                        tokens += result["tokens"]
                        self.total_tokens += result["tokens"]
                        self.ttft_samples.append(result["ttft_ms"])
                        self.e2e_samples.append(result["e2e_ms"])
                    else:
                        errors += 1
                        self.errors.append(result.get("error", ""))

        worker_tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
        await asyncio.gather(*worker_tasks, return_exceptions=True)

        elapsed = max(1, window_s)
        return (
            requests / elapsed,
            tokens / elapsed,
            errors / max(1, requests),
        )
