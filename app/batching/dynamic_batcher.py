"""
InferMesh Dynamic Batching Engine
===================================
Implements continuous/iteration-level batching (vLLM-style) with:
- Chunked prefill to prevent decode starvation
- Sequence packing with first-fit-decreasing bin packing
- Per-step token budget management
- Adaptive batch size based on real-time GPU memory pressure

Continuous batching key insight:
  Each "step" = one transformer forward pass.
  Completed sequences exit mid-batch; new ones enter.
  This eliminates padding waste of traditional static batching.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from typing import Optional

import structlog

from app.config import get_config
from app.models import (
    BatchToken,
    BatchingStrategy,
    InferenceBatch,
    InferenceRequest,
    RequestStatus,
)
from app.metrics.registry import BATCH_SIZE, BATCH_TOKENS, TOKENS_GENERATED_TOTAL, TOKENS_PER_SECOND

logger = structlog.get_logger(__name__)


class BatchSlot:
    """Tracks an active request within the continuous batch."""

    def __init__(self, request: InferenceRequest):
        self.request = request
        self.tokens_generated = 0
        self.is_prefilling = True
        self.prefill_offset = 0    # How many prompt tokens have been processed
        self.kvcache_block_ids: list[int] = []
        self.first_token_time: Optional[float] = None

    @property
    def tokens_remaining(self) -> int:
        return self.request.max_new_tokens - self.tokens_generated

    @property
    def is_done(self) -> bool:
        return self.tokens_generated >= self.request.max_new_tokens

    @property
    def needs_chunked_prefill(self) -> bool:
        return (
            self.is_prefilling and
            self.request.prompt_tokens > get_config().batching.prefill_chunk_size
        )


class ContinuousBatcher:
    """
    Continuous (iteration-level) batching engine.

    Processes incoming requests and schedules them for transformer forward passes.
    Each "step" produces:
    1. A prefill batch (new requests entering the system)
    2. A decode batch (ongoing generation for active sequences)

    Key features:
    - Chunked prefill: Large prompts split into chunks to interleave with decodes
    - Sequence packing: Multiple short prompts packed into one forward pass
    - Token budget: Hard cap on total tokens processed per step
    """

    def __init__(self, worker_id: str, gpu_manager, kvcache_manager):
        self.worker_id = worker_id
        self.gpu_manager = gpu_manager
        self.kvcache_manager = kvcache_manager
        self.config = get_config()

        # Active slots: request_id -> BatchSlot
        self._active_slots: dict[str, BatchSlot] = {}

        # Pending prefill queue (FIFO within worker)
        self._prefill_queue: deque[BatchSlot] = deque()

        self._step_count = 0
        self._total_tokens_generated = 0
        self._last_tps_reset = time.monotonic()
        self._tps_token_count = 0

    # -----------------------------------------------------------------------
    # Slot management
    # -----------------------------------------------------------------------

    def add_request(self, request: InferenceRequest) -> bool:
        """Add request to this worker's batching pool."""
        cfg = self.config.batching
        if len(self._active_slots) + len(self._prefill_queue) >= cfg.max_batch_size:
            return False

        slot = BatchSlot(request)
        request.status = RequestStatus.PREFILLING
        self._prefill_queue.append(slot)
        return True

    def remove_request(self, request_id: str) -> Optional[BatchSlot]:
        """Remove a request (completed, cancelled, or preempted)."""
        return self._active_slots.pop(request_id, None)

    # -----------------------------------------------------------------------
    # Step scheduling
    # -----------------------------------------------------------------------

    def schedule_step(self) -> Optional[InferenceBatch]:
        """
        Schedule one iteration-level step.

        Returns an InferenceBatch describing:
        - Which requests to process (prefill vs. decode)
        - How many tokens to process for each
        """
        cfg = self.config.batching
        token_budget = cfg.max_batch_tokens
        decode_budget = cfg.decode_token_budget
        prefill_chunk = cfg.prefill_chunk_size

        batch_tokens: list[BatchToken] = []
        total_prefill = 0
        total_decode = 0
        is_chunked = False
        chunk_index = 0

        # --- Phase 1: Schedule decode tokens for active slots ---
        decode_remaining = min(decode_budget, token_budget)
        slots_to_remove = []

        for slot_id, slot in list(self._active_slots.items()):
            if slot.is_done:
                slots_to_remove.append(slot_id)
                continue

            tokens_this_step = min(slot.tokens_remaining, 1)  # decode = 1 token/step
            if decode_remaining < tokens_this_step:
                break

            bt = BatchToken(
                request_id=slot.request.request_id,
                prompt_tokens=slot.request.prompt_tokens,
                decode_tokens_remaining=slot.tokens_remaining,
                kvcache_block_ids=slot.kvcache_block_ids,
            )
            batch_tokens.append(bt)
            total_decode += tokens_this_step
            decode_remaining -= tokens_this_step

            # Simulate token generation
            slot.tokens_generated += tokens_this_step
            if slot.first_token_time is None:
                slot.first_token_time = time.monotonic()

        # Remove completed slots
        for sid in slots_to_remove:
            self._active_slots.pop(sid)

        # --- Phase 2: Schedule prefill chunks ---
        prefill_remaining = token_budget - total_decode

        while self._prefill_queue and prefill_remaining > 0:
            slot = self._prefill_queue[0]
            req = slot.request

            remaining_prompt = req.prompt_tokens - slot.prefill_offset
            chunk = min(remaining_prompt, prefill_chunk, prefill_remaining)

            if chunk <= 0:
                break

            bt = BatchToken(
                request_id=req.request_id,
                prompt_tokens=chunk,
                decode_tokens_remaining=req.max_new_tokens,
                kvcache_block_ids=slot.kvcache_block_ids,
            )
            batch_tokens.append(bt)
            total_prefill += chunk
            prefill_remaining -= chunk
            slot.prefill_offset += chunk

            if slot.prefill_offset >= req.prompt_tokens:
                # Prefill complete → move to active decode
                self._prefill_queue.popleft()
                slot.is_prefilling = False
                req.status = RequestStatus.DECODING
                self._active_slots[req.request_id] = slot
                if remaining_prompt > prefill_chunk:
                    is_chunked = True
                    chunk_index += 1
            else:
                # Still prefilling (chunked)
                is_chunked = True
                chunk_index += 1
                break  # Only process one partial prefill per step

        if not batch_tokens:
            return None

        self._step_count += 1
        self._total_tokens_generated += total_decode
        self._tps_token_count += total_decode

        # Update TPS metric
        elapsed = time.monotonic() - self._last_tps_reset
        if elapsed > 1.0:
            tps = self._tps_token_count / elapsed
            TOKENS_PER_SECOND.labels(worker_id=self.worker_id).set(tps)
            TOKENS_GENERATED_TOTAL.labels(worker_id=self.worker_id).inc(self._tps_token_count)
            self._tps_token_count = 0
            self._last_tps_reset = time.monotonic()

        batch = InferenceBatch(
            worker_id=self.worker_id,
            requests=batch_tokens,
            strategy=BatchingStrategy.CONTINUOUS,
            prefill_tokens=total_prefill,
            decode_tokens=total_decode,
            is_chunked_prefill=is_chunked,
            chunk_index=chunk_index,
        )

        BATCH_SIZE.labels(strategy="continuous", worker_id=self.worker_id).observe(
            len(batch_tokens)
        )
        BATCH_TOKENS.labels(worker_id=self.worker_id).observe(batch.total_tokens)

        return batch

    # -----------------------------------------------------------------------
    # Packing
    # -----------------------------------------------------------------------

    def pack_short_sequences(self, requests: list[InferenceRequest]) -> list[InferenceBatch]:
        """
        First-Fit Decreasing (FFD) bin packing for short sequences.
        Groups multiple small requests into a single forward pass.
        """
        cfg = self.config.batching
        token_budget = cfg.max_batch_tokens

        # Sort by descending size (FFD algorithm)
        sorted_reqs = sorted(requests, key=lambda r: r.prompt_tokens, reverse=True)

        bins: list[list[InferenceRequest]] = []
        bin_sizes: list[int] = []

        for req in sorted_reqs:
            placed = False
            for i, (bin_reqs, bin_size) in enumerate(zip(bins, bin_sizes)):
                if bin_size + req.prompt_tokens <= token_budget:
                    bin_reqs.append(req)
                    bin_sizes[i] += req.prompt_tokens
                    placed = True
                    break
            if not placed:
                bins.append([req])
                bin_sizes.append(req.prompt_tokens)

        batches = []
        for bin_reqs in bins:
            tokens = [
                BatchToken(
                    request_id=r.request_id,
                    prompt_tokens=r.prompt_tokens,
                    decode_tokens_remaining=r.max_new_tokens,
                )
                for r in bin_reqs
            ]
            batch = InferenceBatch(
                worker_id=self.worker_id,
                requests=tokens,
                strategy=BatchingStrategy.CONTINUOUS,
                prefill_tokens=sum(t.prompt_tokens for t in tokens),
            )
            batches.append(batch)

        return batches

    # -----------------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------------

    @property
    def active_request_count(self) -> int:
        return len(self._active_slots) + len(self._prefill_queue)

    def stats(self) -> dict:
        return {
            "worker_id": self.worker_id,
            "active_slots": len(self._active_slots),
            "prefill_queue": len(self._prefill_queue),
            "total_steps": self._step_count,
            "total_tokens_generated": self._total_tokens_generated,
        }
