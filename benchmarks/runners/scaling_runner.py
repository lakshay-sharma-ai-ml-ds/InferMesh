"""Scaling Behavior Benchmark"""
from __future__ import annotations
import asyncio, time
import httpx, structlog
from benchmarks.runners.base_runner import BaseBenchmarkRunner, generate_prompt

logger = structlog.get_logger("benchmarks.scaling")

class ScalingBenchmarkRunner(BaseBenchmarkRunner):
    scenario_name = "scaling_behavior"

    async def run(self) -> dict:
        self._start_time = time.monotonic()
        async with httpx.AsyncClient() as client:
            await self._warmup(client, n=3)
            # Measure at low, medium, high load
            low = await self._measure_load(client, concurrency=2, duration_s=max(8, self.duration_s // 4))
            medium = await self._measure_load(client, concurrency=self.concurrency // 2, duration_s=max(8, self.duration_s // 4))
            high = await self._measure_load(client, concurrency=self.concurrency, duration_s=max(8, self.duration_s // 4))

        base = self._build_base_results()
        base.update({
            "low_load_rps": low["rps"],
            "medium_load_rps": medium["rps"],
            "high_load_rps": high["rps"],
            "scale_up_efficiency": round(high["rps"] / max(1, low["rps"]), 3),
            "load_levels": [low, medium, high],
        })
        return base

    async def _measure_load(self, client, concurrency, duration_s):
        reqs = 0; errors = 0
        sem = asyncio.Semaphore(concurrency)
        deadline = time.monotonic() + duration_s
        async def worker():
            nonlocal reqs, errors
            while time.monotonic() < deadline:
                async with sem:
                    prompt, _ = generate_prompt()
                    r = await self._send_inference(client, prompt, max_new_tokens=128)
                    reqs += 1; self.total_requests += 1
                    if r["success"]:
                        self.ttft_samples.append(r["ttft_ms"]); self.e2e_samples.append(r["e2e_ms"])
                    else:
                        errors += 1
        tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
        await asyncio.gather(*tasks, return_exceptions=True)
        return {"concurrency": concurrency, "rps": round(reqs / max(1, duration_s), 2),
                "error_rate": round(errors / max(1, reqs), 4)}
