"""SLA Compliance Benchmark — measures SLA adherence under varying loads."""
from __future__ import annotations
import asyncio, time
import httpx, structlog
from benchmarks.runners.base_runner import BaseBenchmarkRunner, generate_prompt

logger = structlog.get_logger("benchmarks.sla")

class SLAComplianceRunner(BaseBenchmarkRunner):
    scenario_name = "sla_compliance"

    async def run(self) -> dict:
        self._start_time = time.monotonic()
        sla_results = []

        async with httpx.AsyncClient() as client:
            await self._warmup(client, n=3)

            # Test different SLA targets
            sla_configs = [
                {"max_ttft_ms": 500, "max_e2e_ms": 5000, "label": "relaxed"},
                {"max_ttft_ms": 200, "max_e2e_ms": 2000, "label": "standard"},
                {"max_ttft_ms": 100, "max_e2e_ms": 1000, "label": "strict"},
            ]

            for cfg in sla_configs:
                result = await self._run_sla_test(client, cfg, duration_s=self.duration_s // 3)
                sla_results.append(result)

        base = self._build_base_results()
        base.update({
            "sla_test_results": sla_results,
            "overall_sla_compliance": round(
                sum(r["compliance_rate"] for r in sla_results) / max(1, len(sla_results)), 4
            ),
        })
        return base

    async def _run_sla_test(self, client, sla_cfg, duration_s):
        reqs = 0; violations = 0
        sem = asyncio.Semaphore(self.concurrency)
        deadline = time.monotonic() + duration_s

        async def worker():
            nonlocal reqs, violations
            while time.monotonic() < deadline:
                async with sem:
                    prompt, _ = generate_prompt("medium")
                    result = await self._send_inference(
                        client, prompt, max_new_tokens=128, priority="HIGH"
                    )
                    reqs += 1; self.total_requests += 1
                    if result["success"]:
                        self.ttft_samples.append(result["ttft_ms"])
                        self.e2e_samples.append(result["e2e_ms"])
                        if (result["ttft_ms"] > sla_cfg["max_ttft_ms"] or
                                result["e2e_ms"] > sla_cfg["max_e2e_ms"]):
                            violations += 1
                            self.sla_violations += 1
                    else:
                        violations += 1

        tasks = [asyncio.create_task(worker()) for _ in range(self.concurrency)]
        await asyncio.gather(*tasks, return_exceptions=True)

        compliance = 1.0 - (violations / max(1, reqs))
        return {
            "sla_label": sla_cfg["label"],
            "max_ttft_ms": sla_cfg["max_ttft_ms"],
            "max_e2e_ms": sla_cfg["max_e2e_ms"],
            "total_requests": reqs,
            "violations": violations,
            "compliance_rate": round(compliance, 4),
            "violation_rate": round(violations / max(1, reqs), 4),
        }
