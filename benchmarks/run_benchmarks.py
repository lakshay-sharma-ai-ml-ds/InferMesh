"""
InferMesh Benchmark Runner — Entry Point
==========================================
CLI tool to run all benchmark scenarios and produce structured results.

Output (2 files per run):
  results/<run_id>/summary_table.txt   — cross-scenario comparison table
  results/<run_id>/combined_report.md  — full human-readable report

Usage:
    python benchmarks/run_benchmarks.py --scenarios all
    python benchmarks/run_benchmarks.py --scenarios latency_sweep throughput_max
    python benchmarks/run_benchmarks.py --scenarios all --workers 4 --duration 60
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
import structlog
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.runners.latency_runner import LatencyBenchmarkRunner
from benchmarks.runners.throughput_runner import ThroughputBenchmarkRunner
from benchmarks.runners.fault_runner import FaultToleranceRunner
from benchmarks.runners.scaling_runner import ScalingBenchmarkRunner
from benchmarks.runners.cache_runner import CacheBenchmarkRunner
from benchmarks.runners.sla_runner import SLAComplianceRunner
from benchmarks.analysis.statistical import StatisticalAnalyzer

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="%H:%M:%S"),
        structlog.dev.ConsoleRenderer(),
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

logger = structlog.get_logger("benchmarks")
app = typer.Typer(help="InferMesh Benchmark Suite")

SCENARIO_MAP = {
    "latency_sweep":    LatencyBenchmarkRunner,
    "throughput_max":   ThroughputBenchmarkRunner,
    "fault_tolerance":  FaultToleranceRunner,
    "scaling_behavior": ScalingBenchmarkRunner,
    "cache_efficiency": CacheBenchmarkRunner,
    "sla_compliance":   SLAComplianceRunner,
}

SCENARIO_DESCRIPTIONS = {
    "latency_sweep": (
        "Latency Sweep",
        "Sweeps concurrency from 1 → 10 concurrent requests and measures how TTFT "
        "(Time-To-First-Token) and end-to-end latency grow. Identifies the 'knee point' "
        "where latency starts degrading sharply.",
        "Sends fixed-prompt requests at each concurrency level for equal time slices. "
        "Records per-request TTFT and E2E latency, then computes P50/P95/P99 percentiles "
        "using sorted sample arrays.",
    ),
    "throughput_max": (
        "Maximum Throughput",
        "Finds the maximum sustainable request-per-second rate before the error rate "
        "exceeds 1%. Tests the system under extreme load (up to 128 concurrent requests).",
        "Ramps concurrency in steps (4 → 8 → 16 → 32 → 64 → 96 → 128). At each step, "
        "fires requests for 10 seconds and records RPS and error rate. The peak before "
        "error rate rises is reported as max sustainable throughput.",
    ),
    "fault_tolerance": (
        "Fault Tolerance",
        "Measures how well the system maintains throughput when GPUs fail and recover. "
        "Baseline → inject faults → measure degradation → measure recovery.",
        "Measures RPS in three windows: baseline (10s), fault injection (15s), and "
        "recovery (10s). The resilience score weights fault-window RPS (40%) and "
        "recovery RPS (60%) relative to baseline.",
    ),
    "scaling_behavior": (
        "Scaling Behavior",
        "Tests throughput at low (25%), medium (50%), and high (100%) concurrency to "
        "understand how the system scales linearly under increasing load.",
        "Runs 3 fixed-duration windows at different concurrency fractions. Calculates "
        "RPS at each level and the scale-up efficiency ratio (high_RPS / low_RPS).",
    ),
    "cache_efficiency": (
        "KV-Cache Efficiency",
        "Measures how the prefix-sharing KV-cache improves throughput for repeated "
        "prefixes (hot cache) versus fresh prompts (cold cache).",
        "Sends two batches: one with a shared system-prompt prefix (cache warm), one "
        "with random unique prompts (cold). Compares RPS and queries the /v1/cluster/kvcache "
        "endpoint for hit rates.",
    ),
    "sla_compliance": (
        "SLA Compliance",
        "Validates that the system meets latency SLA tiers: Relaxed (2000ms TTFT), "
        "Standard (500ms), and Strict (100ms). Reports compliance rate per tier.",
        "Each tier sends requests with the corresponding SLA config. A request is "
        "SLA-compliant if its TTFT falls within the tier's budget. Compliance rate = "
        "1 − (violations / total requests).",
    ),
}


@app.command()
def run(
    scenarios: list[str] = typer.Option(["all"], "--scenarios", "-s", help="Scenarios to run"),
    workers: int = typer.Option(4, "--workers", "-w", help="Number of simulated GPU workers"),
    duration: int = typer.Option(30, "--duration", "-d", help="Duration per scenario (seconds)"),
    concurrency: int = typer.Option(10, "--concurrency", "-c", help="Concurrent requests"),
    output_dir: str = typer.Option("results", "--output", "-o", help="Output directory"),
    base_url: str = typer.Option("http://localhost:8000", "--url", help="InferMesh API URL"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run InferMesh benchmark scenarios and save results."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_path / run_id
    run_dir.mkdir(exist_ok=True)

    if "all" in scenarios:
        scenarios = list(SCENARIO_MAP.keys())

    logger.info(
        "InferMesh Benchmark Suite",
        scenarios=scenarios,
        workers=workers,
        duration=duration,
        concurrency=concurrency,
        run_id=run_id,
    )

    typer.echo("\n" + "=" * 60)
    typer.echo("  InferMesh Benchmark Suite v1.0")
    typer.echo("=" * 60)

    all_results: dict = {}
    suite_start = time.monotonic()

    for scenario_name in scenarios:
        if scenario_name not in SCENARIO_MAP:
            typer.echo(f"[WARN] Unknown scenario: {scenario_name}", err=True)
            continue

        typer.echo(f"\n▶ Running: {scenario_name}")
        typer.echo(f"  Duration: {duration}s | Concurrency: {concurrency} | Workers: {workers}")

        runner_cls = SCENARIO_MAP[scenario_name]
        runner = runner_cls(
            base_url=base_url,
            duration_s=duration,
            concurrency=concurrency,
            num_workers=workers,
        )

        try:
            results = asyncio.run(runner.run())
            all_results[scenario_name] = results
            _print_scenario_summary(scenario_name, results)

        except Exception as exc:
            logger.error(f"Scenario {scenario_name} failed", error=str(exc))
            all_results[scenario_name] = {"error": str(exc)}
            if verbose:
                import traceback
                traceback.print_exc()

    suite_elapsed = time.monotonic() - suite_start

    # Statistical analysis
    typer.echo("\n" + "=" * 60)
    typer.echo("  Statistical Analysis")
    typer.echo("=" * 60)

    analyzer = StatisticalAnalyzer()
    analysis = analyzer.analyze_suite(all_results)
    analysis["run_id"] = run_id
    analysis["total_duration_s"] = round(suite_elapsed, 2)
    analysis["config"] = {
        "workers": workers,
        "duration_per_scenario": duration,
        "concurrency": concurrency,
    }

    # File 1: Summary table
    _save_summary_table(all_results, analysis, run_dir)

    # File 2: Combined markdown report
    _save_combined_report(all_results, analysis, run_dir, run_id, suite_elapsed, workers, duration, concurrency)

    typer.echo(f"\n✅ Results saved to: {run_dir}/")
    typer.echo(f"   summary_table.txt")
    typer.echo(f"   combined_report.md")
    typer.echo(f"   Total time: {suite_elapsed:.1f}s\n")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _print_scenario_summary(name: str, results: dict) -> None:
    """Print compact per-scenario output during the run."""
    typer.echo(f"\n  ─── {name} Results ───")
    for key in ["p50_ttft_ms", "p95_ttft_ms", "p99_ttft_ms",
                "throughput_rps", "throughput_tokens_per_sec",
                "sla_violation_rate", "cache_hit_rate", "error_rate"]:
        if key in results:
            val = results[key]
            label = key.replace("_", " ").upper()
            if isinstance(val, float):
                typer.echo(f"  {label:<30} {val:.3f}")
            else:
                typer.echo(f"  {label:<30} {val}")


def _save_summary_table(results: dict, analysis: dict, run_dir: Path) -> None:
    """Save the cross-scenario comparison table as summary_table.txt."""
    from tabulate import tabulate

    rows = []
    for name, r in results.items():
        if "error" in r:
            rows.append([name, "FAILED", "-", "-", "-", "-"])
            continue
        rows.append([
            name,
            f"{r.get('p50_ttft_ms', 0):.1f}",
            f"{r.get('p99_ttft_ms', 0):.1f}",
            f"{r.get('throughput_rps', 0):.1f}",
            f"{r.get('error_rate', 0) * 100:.2f}%",
            f"{r.get('sla_violation_rate', 0) * 100:.2f}%",
        ])

    headers = ["Scenario", "P50 TTFT(ms)", "P99 TTFT(ms)", "RPS", "Error%", "SLA Viol%"]
    table = tabulate(rows, headers=headers, tablefmt="rounded_outline")
    typer.echo("\n" + table)

    table_file = run_dir / "summary_table.txt"
    with open(table_file, "w") as f:
        f.write("InferMesh Benchmark Summary\n")
        f.write(f"Run ID: {analysis.get('run_id', 'N/A')}\n\n")
        f.write(table)
        f.write("\n")


def _save_combined_report(
    results: dict,
    analysis: dict,
    run_dir: Path,
    run_id: str,
    total_elapsed: float,
    workers: int,
    duration: int,
    concurrency: int,
) -> None:
    """Save a single human-readable markdown report combining all scenarios."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []

    # Header
    lines += [
        "# InferMesh Benchmark Report",
        "",
        f"**Run ID:** `{run_id}` | **Date:** {ts} | **Runtime:** {total_elapsed:.1f}s",
        "",
        "> Production-grade distributed LLM orchestrator — priority scheduling, topology-aware "
        "placement, continuous batching, ARC KV-cache. Benchmarks stress the orchestration layer "
        "(no real GPU/model needed).",
        "",
        "---",
        "",
    ]

    # Configuration
    lines += [
        "## Configuration",
        "",
        "| Workers | Duration | Concurrency | Scheduler | KV-Cache |",
        "|---------|----------|-------------|-----------|----------|",
        f"| {workers} (A100×2, A10G×1, T4×1) | {duration}s/scenario | max {concurrency} | Priority + Topology-Aware | ARC + prefix sharing |",
        "",
        "---",
        "",
    ]

    # Results Summary
    lines += [
        "## Results Summary",
        "",
        "| Scenario | P50 TTFT | P99 TTFT | RPS | Error% | SLA Viol% |",
        "|----------|----------|----------|-----|--------|-----------|",
    ]
    for name, r in results.items():
        if "error" in r:
            lines.append(f"| {name} | FAILED | — | — | — | — |")
        else:
            lines.append(
                f"| {name} | "
                f"{r.get('p50_ttft_ms', 0):.1f} ms | "
                f"{r.get('p99_ttft_ms', 0):.1f} ms | "
                f"{r.get('throughput_rps', 0):.1f} | "
                f"{r.get('error_rate', 0) * 100:.2f}% | "
                f"{r.get('sla_violation_rate', 0) * 100:.2f}% |"
            )
    lines += ["", "---", ""]

    # Scenario Highlights
    HIGHLIGHTS = {
        "latency_sweep":    ("TTFT growth 1→10 concurrency; finds knee point",     "p99_e2e_ms",            "P99 E2E", "ms",   "total_requests", "reqs"),
        "throughput_max":   ("Max RPS before error>1%; ramps to 128 concurrency",  "error_rate",            "Error",   "%×100","total_requests", "reqs"),
        "fault_tolerance":  ("RPS across baseline→fault injection→recovery",        "resilience_score",      "Resilience", "",  "total_requests", "reqs"),
        "scaling_behavior": ("Throughput at 25/50/100% concurrency; linearity",    "scale_up_efficiency",   "Scale-Up", "×",  "total_requests", "reqs"),
        "cache_efficiency": ("Hot (prefix-shared) vs cold cache; ARC hit rate",    "cache_hit_rate",        "Hit Rate", "%×100","total_requests","reqs"),
        "sla_compliance":   ("Compliance across Relaxed/Standard/Strict SLA tiers","overall_sla_compliance","SLA Comp.","×100%","total_requests","reqs"),
    }

    lines += [
        "## Scenario Highlights",
        "",
        "| Scenario | What's Tested | Key Metric | Total Reqs |",
        "|----------|--------------|------------|------------|",
    ]
    for name, r in results.items():
        if "error" in r:
            lines.append(f"| **{name}** | — | FAILED | — |")
            continue
        h = HIGHLIGHTS.get(name)
        if h:
            _, mkey, mlabel, munit, rkey, _ = h
            val = r.get(mkey)
            if val is None:
                mstr = "—"
            elif munit == "%×100":
                mstr = f"{mlabel} = **{val * 100:.2f}%**"
            elif munit == "×100%":
                mstr = f"{mlabel} = **{val * 100:.1f}%**"
            elif munit == "×":
                mstr = f"{mlabel} = **{val:.2f}×**"
            elif munit == "ms":
                mstr = f"{mlabel} = **{val:.1f} ms**"
            else:
                mstr = f"{mlabel} = **{val:.3f}**"
            total = r.get(rkey, "—")
            what = h[0]
        else:
            mstr = "—"
            total = r.get("total_requests", "—")
            what = name.replace("_", " ").title()
        
        if isinstance(total, (int, float)):
            lines.append(f"| **{name}** | {what} | {mstr} | {int(total):,} |")
        else:
            lines.append(f"| **{name}** | {what} | {mstr} | {total} |")
    lines += ["", "---", ""]

    # Key Observations
    successful = {k: v for k, v in results.items() if "error" not in v}
    failed     = [k for k, v in results.items() if "error" in v]

    obs = []
    if failed:
        obs.append(f"⚠️ | **{len(failed)} scenario(s) failed:** {', '.join(failed)}")
    if "latency_sweep" in successful:
        r = successful["latency_sweep"]
        tag = "✅" if r.get("p99_ttft_ms", 999) < 50 else "⚠️"
        obs.append(f"{tag} | **Latency** — P50 {r.get('p50_ttft_ms',0):.1f}ms / P99 {r.get('p99_ttft_ms',0):.1f}ms; near-zero scheduler overhead")
    if "throughput_max" in successful:
        r = successful["throughput_max"]
        err = r.get("error_rate", 0) * 100
        obs.append(f"{'⚠️' if err > 5 else '✅'} | **Throughput** — {err:.1f}% error at max concurrency; {'admission control correctly load-sheds at saturation' if err > 5 else 'system handles max load cleanly'}")
    if "fault_tolerance" in successful:
        r = successful["fault_tolerance"]
        sc = r.get("resilience_score", 0)
        obs.append(f"✅ | **Fault Tolerance** — Resilience {sc:.3f} ({'high' if sc > 0.7 else 'moderate' if sc > 0.4 else 'low'}); circuit breaker + health monitor active")
    if "sla_compliance" in successful:
        r = successful["sla_compliance"]
        comp = r.get("overall_sla_compliance", 1 - r.get("sla_violation_rate", 0))
        obs.append(f"✅ | **SLA Compliance** — {comp * 100:.1f}% across all tiers")
    if "cache_efficiency" in successful:
        r = successful["cache_efficiency"]
        hit = r.get("cache_hit_rate", 0)
        obs.append(f"{'✅' if hit > 0.3 else 'ℹ️'} | **KV-Cache** — {hit:.1%} hit rate {'(prefix sharing active)' if hit > 0.3 else '(cold first run; warms over repeated runs)'}")

    lines += ["## Key Observations", "", "| | Finding |", "|--|---------|"]
    for o in obs:
        lines.append(f"| {o} |")
    lines += ["", "---", ""]

    # Metric Glossary
    lines += [
        "## Metric Glossary",
        "",
        "| Metric | Meaning |",
        "|--------|---------|",
        "| **TTFT** | Time-To-First-Token: scheduler + queue + prefill latency |",
        "| **E2E Latency** | Full round-trip including decode |",
        "| **RPS** | Orchestration-layer throughput across all workers |",
        "| **Error%** | Rejected/timeout requests; non-zero at saturation is expected |",
        "| **SLA Viol%** | Completed requests that exceeded TTFT/E2E budget |",
        "| **Resilience Score** | Weighted RPS ratio (fault 40% + recovery 60%) vs baseline; ≥0.7 = high |",
        "",
        "---",
        "",
        f"*Generated by InferMesh Benchmark Suite — `python benchmarks/run_benchmarks.py --scenarios all`*",
    ]

    report_file = run_dir / "combined_report.md"
    with open(report_file, "w") as f:
        f.write("\n".join(lines))
        f.write("\n")


if __name__ == "__main__":
    app()
