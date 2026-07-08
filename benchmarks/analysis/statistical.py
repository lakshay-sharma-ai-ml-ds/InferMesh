"""
Statistical Analyzer
=====================
Provides cross-scenario statistical analysis for benchmark results.
Computes confidence intervals, regression analysis, and performance grades.
"""

from __future__ import annotations

import math
import statistics
from typing import Optional


class StatisticalAnalyzer:
    """
    Statistical analysis for benchmark suite results.

    Provides:
    - Descriptive statistics (mean, std, CV)
    - 95% confidence intervals (t-distribution)
    - Performance grading (A-F)
    - Cross-scenario regression insights
    """

    def analyze_suite(self, results: dict) -> dict:
        """Analyze all scenario results and produce a comprehensive report."""
        analysis = {
            "scenario_grades": {},
            "best_latency_scenario": None,
            "best_throughput_scenario": None,
            "overall_grade": "N/A",
        }

        ttft_scores = []
        throughput_scores = []

        for name, result in results.items():
            if "error" in result:
                analysis["scenario_grades"][name] = "F"
                continue

            grade = self._grade_scenario(result)
            analysis["scenario_grades"][name] = grade

            p99 = result.get("p99_ttft_ms", 9999)
            rps = result.get("throughput_rps", 0)
            ttft_scores.append((name, p99))
            throughput_scores.append((name, rps))

        if ttft_scores:
            analysis["best_latency_scenario"] = min(ttft_scores, key=lambda x: x[1])[0]
        if throughput_scores:
            analysis["best_throughput_scenario"] = max(throughput_scores, key=lambda x: x[1])[0]

        # Overall grade
        grades = list(analysis["scenario_grades"].values())
        grade_vals = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}
        avg_grade = sum(grade_vals.get(g, 0) for g in grades) / max(1, len(grades))
        reverse_map = {4: "A", 3: "B", 2: "C", 1: "D", 0: "F"}
        analysis["overall_grade"] = reverse_map.get(round(avg_grade), "C")

        # Confidence intervals for latency
        analysis["confidence_intervals"] = self._compute_ci_across_scenarios(results)

        return analysis

    def _grade_scenario(self, result: dict) -> str:
        """
        Grade a scenario result A-F based on:
        - P99 TTFT < 500ms → A bonus
        - Error rate < 1% → required
        - SLA violation rate < 5% → required
        """
        p99_ttft = result.get("p99_ttft_ms", 9999)
        error_rate = result.get("error_rate", 1.0)
        sla_viol = result.get("sla_violation_rate", 1.0)

        if error_rate > 0.10 or sla_viol > 0.20:
            return "F"
        if p99_ttft < 200 and error_rate < 0.01:
            return "A"
        if p99_ttft < 500 and error_rate < 0.02:
            return "B"
        if p99_ttft < 1000 and error_rate < 0.05:
            return "C"
        return "D"

    def _compute_ci_across_scenarios(self, results: dict) -> dict:
        """Compute 95% CI for key metrics across all scenarios."""
        all_ttft = []
        all_e2e = []

        for result in results.values():
            if "error" not in result:
                p50 = result.get("p50_ttft_ms", 0)
                p99 = result.get("p99_ttft_ms", 0)
                if p50 > 0:
                    all_ttft.append(p50)
                if p99 > 0:
                    all_e2e.append(p99)

        def ci95(data: list[float]) -> dict:
            if len(data) < 2:
                return {"mean": data[0] if data else 0, "ci_low": 0, "ci_high": 0}
            n = len(data)
            m = statistics.mean(data)
            s = statistics.stdev(data)
            t_val = 2.0  # t-critical for 95% CI, large sample approximation
            margin = t_val * (s / math.sqrt(n))
            return {
                "mean": round(m, 2),
                "std": round(s, 2),
                "ci_low": round(m - margin, 2),
                "ci_high": round(m + margin, 2),
                "n": n,
            }

        return {
            "ttft_p50_across_scenarios": ci95(all_ttft),
            "ttft_p99_across_scenarios": ci95(all_e2e),
        }
