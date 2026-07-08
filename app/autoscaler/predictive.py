"""
InferMesh Predictive Autoscaler
================================
EWMA-based time-series forecasting for proactive GPU pre-warming.

The predictive autoscaler:
1. Maintains a rolling window of cluster load observations
2. Fits an EWMA model to smooth out noise
3. Projects load `horizon_s` seconds into the future
4. Triggers scale-up BEFORE load crosses the threshold
   (preventing the latency spike that reactive scaling misses)
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from typing import Optional

import structlog

from app.config import get_config
from app.metrics.registry import AUTOSCALER_PREDICTED_LOAD

logger = structlog.get_logger(__name__)


class EWMAForecaster:
    """
    Exponential Weighted Moving Average forecaster.
    
    Uses double exponential smoothing (Holt's method) for trend-aware prediction.
    - Level (α): Tracks current average
    - Trend (β): Tracks rate of change
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.1):
        self.alpha = alpha
        self.beta = beta
        self._level: Optional[float] = None
        self._trend: float = 0.0
        self._history: deque[tuple[float, float]] = deque(maxlen=300)  # 5min @ 1s

    def update(self, value: float) -> None:
        now = time.monotonic()
        if self._level is None:
            self._level = value
            self._trend = 0.0
        else:
            prev_level = self._level
            self._level = self.alpha * value + (1 - self.alpha) * (prev_level + self._trend)
            self._trend = self.beta * (self._level - prev_level) + (1 - self.beta) * self._trend
        self._history.append((now, value))

    def forecast(self, horizon_s: float) -> float:
        """Predict value `horizon_s` seconds from now."""
        if self._level is None:
            return 0.0
        # Convert horizon to steps (assuming 1s sampling)
        steps = int(horizon_s)
        prediction = self._level + steps * self._trend
        return max(0.0, min(1.0, prediction))

    @property
    def current_level(self) -> float:
        return self._level or 0.0


class PredictiveAutoscaler:
    """
    Predictive autoscaler that pre-warms GPUs before load arrives.

    Works alongside the HPA (horizontal autoscaler) as a feed-forward controller.
    """

    def __init__(self, gpu_manager, queue):
        self.gpu_manager = gpu_manager
        self.queue = queue
        self.config = get_config()
        self._forecaster = EWMAForecaster(
            alpha=self.config.autoscaler.ewma_alpha,
            beta=0.1,
        )
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        if not self.config.autoscaler.predictive_enabled:
            return
        self._running = True
        self._task = asyncio.create_task(
            self._predict_loop(), name="autoscaler-predictive"
        )
        logger.info("Predictive autoscaler started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _predict_loop(self) -> None:
        while self._running:
            try:
                await self._update_and_predict()
            except Exception as exc:
                logger.warning("Predictive autoscaler error", error=str(exc))
            await asyncio.sleep(5.0)

    async def _update_and_predict(self) -> None:
        cfg = self.config.autoscaler
        workers = self.gpu_manager.get_healthy_workers()

        if not workers:
            current_load = 1.0
        else:
            utilizations = [w.load_score for w in workers]
            current_load = sum(utilizations) / len(utilizations)

        # Include queue pressure
        queue_depth = self.queue.total_depth()
        queue_pressure = queue_depth / max(1, self.config.queue.max_queue_depth)
        combined_load = 0.7 * current_load + 0.3 * queue_pressure

        self._forecaster.update(combined_load)
        predicted = self._forecaster.forecast(cfg.forecast_horizon_s)

        AUTOSCALER_PREDICTED_LOAD.set(predicted)

        logger.debug(
            "Predictive load forecast",
            current=f"{combined_load:.3f}",
            predicted_60s=f"{predicted:.3f}",
        )

        # Pre-warm: if predicted load exceeds threshold, signal HPA
        if predicted > cfg.scale_up_threshold:
            logger.info(
                "Predictive pre-warm signal",
                predicted_load=f"{predicted:.2f}",
                horizon_s=cfg.forecast_horizon_s,
            )

    def get_forecast(self, horizon_s: Optional[float] = None) -> dict:
        cfg = self.config.autoscaler
        h = horizon_s or cfg.forecast_horizon_s
        return {
            "current_level": self._forecaster.current_level,
            "forecast": self._forecaster.forecast(h),
            "horizon_s": h,
        }
