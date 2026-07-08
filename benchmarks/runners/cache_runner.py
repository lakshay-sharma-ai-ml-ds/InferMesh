"""Cache Efficiency Benchmark — measures KV-cache hit rates with varying prefix diversity."""
from __future__ import annotations
import asyncio, random, time
import httpx, structlog
from benchmarks.runners.base_runner import BaseBenchmarkRunner, TOPICS

logger = structlog.get_logger("benchmarks.cache")

SHARED_PREFIX = "You are an expert AI assistant specialized in distributed systems and infrastructure. Your task is to help users understand complex technical topics with clear, accurate explanations. "

class CacheBenchmarkRunner(BaseBenchmarkRunner):
    scenario_name = "cache_efficiency"

    async def run(self) -> dict:
        self._start_time = time.monotonic()
        cache_hit_counts = {"hot": 0, "cold": 0}
        total_by_type = {"hot": 0, "cold": 0}

        async with httpx.AsyncClient() as client:
            await self._warmup(client, n=3)

            # Hot cache: reuse shared prefix (should get high hit rate)
            hot_result = await self._run_cache_test(client, use_shared_prefix=True,
                                                     duration_s=self.duration_s // 2, label="hot")
            # Cold cache: unique prompts (should get low hit rate)
            cold_result = await self._run_cache_test(client, use_shared_prefix=False,
                                                      duration_s=self.duration_s // 2, label="cold")

            # Get actual hit rates from API
            try:
                resp = await client.get(f"{self.base_url}/v1/cluster/kvcache", timeout=5.0)
                kvcache_stats = resp.json() if resp.status_code == 200 else {}
            except Exception:
                kvcache_stats = {}

        avg_hit_rate = 0.0
        if kvcache_stats:
            rates = [v.get("hit_rate", 0) for v in kvcache_stats.values()]
            avg_hit_rate = sum(rates) / len(rates) if rates else 0.0

        base = self._build_base_results()
        base.update({
            "cache_hit_rate": round(avg_hit_rate, 4),
            "hot_cache_rps": hot_result["rps"],
            "cold_cache_rps": cold_result["rps"],
            "throughput_gain_from_cache": round(
                hot_result["rps"] / max(0.001, cold_result["rps"]), 3
            ),
            "kvcache_worker_stats": kvcache_stats,
        })
        return base

    async def _run_cache_test(self, client, use_shared_prefix, duration_s, label):
        reqs = 0; errors = 0
        sem = asyncio.Semaphore(self.concurrency)
        deadline = time.monotonic() + duration_s

        async def worker():
            nonlocal reqs, errors
            while time.monotonic() < deadline:
                async with sem:
                    if use_shared_prefix:
                        topic = random.choice(TOPICS[:5])  # fewer topics = more hits
                        prompt = SHARED_PREFIX + f"Explain {topic}."
                    else:
                        topic = f"unique_{random.randint(1, 10000)}"
                        prompt = f"Describe {topic} in great detail covering all aspects."

                    r = await self._send_inference(client, prompt, max_new_tokens=64)
                    reqs += 1; self.total_requests += 1
                    if r["success"]:
                        self.ttft_samples.append(r["ttft_ms"]); self.e2e_samples.append(r["e2e_ms"])
                    else:
                        errors += 1

        tasks = [asyncio.create_task(worker()) for _ in range(self.concurrency)]
        await asyncio.gather(*tasks, return_exceptions=True)
        return {"label": label, "rps": round(reqs / max(1, duration_s), 2),
                "error_rate": round(errors / max(1, reqs), 4)}
