# InferMesh

**Distributed LLM Inference Orchestrator** - a production-grade, Kubernetes-inspired scheduler that intelligently distributes LLM inference across heterogeneous GPU clusters while minimizing latency and maximizing utilization.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                               │
│              REST API (FastAPI)  ·  gRPC  ·  Rate Limiting              │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼────────────────────────────────────┐
│                          ADMISSION CONTROL                              │
│         SLA Feasibility Check  ·  User Quotas  ·  Load Shedding          │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼────────────────────────────────────┐
│                            PRIORITY QUEUE                               │
│      Critical  →  High  →  Normal  →  Low  →  Background Priorities     │
│             Fair Request Queue  ·  Starvation Prevention                │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼────────────────────────────────────┐
│                           SCHEDULER LAYER                               │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐   │
│  │   Priority &     │    │  Topology-Aware  │    │  Gang Scheduler  │   │
│  │   Preemption     │    │  (GPU Locality)  │    │   (Multi-GPU)    │   │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘   │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                ┌────────────────────┼────────────────────┐
                │                    │                    │
        ┌───────▼──────┐     ┌───────▼──────┐     ┌───────▼──────┐
        │ GPU Worker 0 │     │ GPU Worker 1 │     │ GPU Worker N │
        │              │     │              │     │              │
        │ ┌──────────┐ │     │ ┌──────────┐ │     │ ┌──────────┐ │
        │ │Continuous│ │     │ │Continuous│ │     │ │Continuous│ │
        │ │ Batching │ │     │ │ Batching │ │     │ │ Batching │ │
        │ └──────────┘ │     │ └──────────┘ │     │ └──────────┘ │
        │ ┌──────────┐ │     │ ┌──────────┐ │     │ ┌──────────┐ │
        │ │ KV-Cache │ │     │ │ KV-Cache │ │     │ │ KV-Cache │ │
        │ │ Manager  │ │     │ │ Manager  │ │     │ │ Manager  │ │
        │ └──────────┘ │     │ └──────────┘ │     │ └──────────┘ │
        └──────────────┘     └──────────────┘     └──────────────┘
                │                    │                    │
┌───────────────▼────────────────────▼────────────────────▼───────────────┐
│                        MONITORING & CONTROL PLANE                       │
│                                                                         │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐   │
│  │  Health Monitor  │    │    Autoscaler    │    │    Metrics &     │   │
│  │ (Failure Detect) │    │ (Scale Up/Down)  │    │    Dashboards    │   │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Features

| Component | Implementation |
|-----------|----------------|
| **Scheduler** | Priority + Preemption, Topology-Aware (NUMA/NVLink), Gang (TP) |
| **Queue** | Weighted Fair Queuing, SortedList O(log n), per-tenant quotas |
| **Dynamic Batching** | Continuous (iteration-level), Chunked Prefill, FFD Packing |
| **KV-Cache** | ARC eviction, Radix-tree prefix sharing, cross-worker migration |
| **Health Monitor** | Circuit Breaker FSM, heartbeat tracking, MTTR measurement |
| **Autoscaler** | HPA (reactive) + Predictive (Holt's EWMA), cooldown control |
| **Admission Control** | Token bucket rate limiting, SLA feasibility, load shedding |
| **Observability** | Prometheus (40+ metrics), Grafana dashboards, structured logs |

---

## Benchmark Suite

Run all scenarios with:
```bash
python benchmarks/run_benchmarks.py --scenarios all --duration 60
```

Results are saved to `results/<timestamp>/`:

| Scenario | What it measures |
|----------|-----------------|
| `latency_sweep` | TTFT/E2E latency across concurrency levels + knee point |
| `throughput_max` | Maximum sustainable RPS and tokens/sec |
| `fault_tolerance` | Resilience score during simulated GPU failures |
| `scaling_behavior` | Throughput at low/medium/high load |
| `cache_efficiency` | KV-cache hit rate, hot vs cold prefix performance |
| `sla_compliance` | Adherence to relaxed/standard/strict SLA tiers |

---

## Installation

### Docker Compose 
```bash
git clone https://github.com/yourusername/infermesh.git
cd infermesh/infermesh
docker compose up --build
```

Services:
- **InferMesh API**: `http://localhost:8000`
- **Prometheus**: `http://localhost:9090`
- **Grafana**: `http://localhost:3000` (admin / infermesh)

### Local Development
```bash
cd infermesh
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload --port 8000
```

### Run Tests
```bash
pytest tests/ -v --cov=app --cov-report=term-missing
```

---

## Project Structure

```
infermesh/
├── app/
│   ├── scheduler/          # Priority, Topology-Aware, Gang schedulers
│   ├── resource/           # GPU pool, memory tracking, topology graph
│   ├── queue/              # WFQ priority queue, admission control
│   ├── batching/           # Continuous batcher, chunked prefill
│   ├── kvcache/            # ARC cache, prefix sharing
│   ├── health/             # Circuit breaker, health monitor
│   ├── autoscaler/         # HPA + Predictive autoscaler
│   ├── metrics/            # Prometheus registry + collectors
│   ├── api/                # FastAPI routes + middleware
│   ├── config.py           # Pydantic v2 hierarchical config
│   ├── models.py           # Domain models + GPU specs registry
│   └── orchestrator.py     # Central coordination engine
├── benchmarks/
│   ├── runners/            # 6 benchmark scenarios
│   ├── analysis/           # Statistical analysis, grading
│   └── run_benchmarks.py   # CLI entry point
├── tests/
│   ├── unit/               # Scheduler, KV-cache, queue tests
│   └── integration/        # Full orchestrator pipeline tests
├── monitoring/
│   ├── prometheus.yml      # Prometheus scrape config
│   ├── rules/alerts.yml    # Alerting rules
│   └── grafana/            # Auto-provisioned dashboard
├── configs/default.yaml    # Full system configuration
├── results/                # Benchmark output (auto-generated)
├── Dockerfile
└── docker-compose.yml      # InferMesh + Redis + Prometheus + Grafana
```

---

## Configuration

All settings are controlled using `configs/default.yaml` or environment variables (`INFERMESH_<KEY>`):

```bash
INFERMESH_NUM_SIMULATED_GPUS=8        
INFERMESH_SCHEDULER_ALGORITHM=topology_aware
INFERMESH_AUTOSCALER_MAX_WORKERS=16
INFERMESH_KVCACHE_EVICTION_POLICY=arc
INFERMESH_LOG_LEVEL=DEBUG
```

---
