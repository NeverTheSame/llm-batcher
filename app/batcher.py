"""Operational microbatching for the proxy.

Concurrent requests that arrive close together are grouped into a short
admission window and then dispatched as N concurrent realtime upstream calls,
gated by a shared concurrency semaphore. This is not the Anthropic Message
Batches API (which is offline, with a slow SLA); it is request-path admission
control. The goal is to protect the upstream from thundering-herd spikes and
raise throughput under concurrency while adding only a few milliseconds of
admission delay.

A window flushes when either:
  * it reaches ``max_batch_size`` (immediate), or
  * ``max_wait_ms`` elapses since the first item was queued (timer).

Every submitted Future is guaranteed to resolve: per-item errors are isolated,
a whole-batch timeout or crash fails every still-pending Future, and shutdown
fails any remaining Futures.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional


DispatchOne = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass
class _Item:
    params: dict[str, Any]
    future: "asyncio.Future[dict[str, Any]]"


@dataclass
class BatcherStats:
    submitted: int = 0
    windows: int = 0
    batch_sizes: list[int] = field(default_factory=list)


class MicroBatcher:
    """Groups concurrent requests into admission windows and fans them out.

    Parameters
    ----------
    dispatch_one:
        Async callable that performs a single realtime upstream call for one
        request's params and returns its (Anthropic) response dict.
    max_batch_size:
        Flush a window immediately once this many items are queued.
    max_wait_ms:
        Maximum time to wait for a window to fill before flushing.
    max_concurrency:
        Cap on simultaneous in-flight upstream calls across all windows.
    batch_timeout_s:
        Hard ceiling on how long a single window's dispatch may take. On
        timeout, every still-pending Future in that window fails.
    on_flush:
        Optional sync hook invoked with the window size when a window is
        dispatched. Used for observability and deterministic tests.
    """

    def __init__(
        self,
        dispatch_one: DispatchOne,
        *,
        max_batch_size: int = 16,
        max_wait_ms: int = 20,
        max_concurrency: int = 8,
        batch_timeout_s: float = 60.0,
        on_flush: Optional[Callable[[int], None]] = None,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self._dispatch_one = dispatch_one
        self._max_batch_size = max_batch_size
        self._max_wait_s = max_wait_ms / 1000.0
        self._batch_timeout_s = batch_timeout_s
        self._on_flush = on_flush

        self._pending: list[_Item] = []
        self._lock = asyncio.Lock()
        self._timer: Optional[asyncio.Task[None]] = None
        self._dispatch_tasks: set[asyncio.Task[None]] = set()
        self._sem = asyncio.Semaphore(max_concurrency)
        self._closed = False
        self.stats = BatcherStats()

    async def submit(self, params: dict[str, Any]) -> dict[str, Any]:
        """Queue one request and await its upstream response."""
        if self._closed:
            raise RuntimeError("MicroBatcher is closed")

        loop = asyncio.get_running_loop()
        future: "asyncio.Future[dict[str, Any]]" = loop.create_future()
        item = _Item(params=params, future=future)

        batch: Optional[list[_Item]] = None
        async with self._lock:
            self._pending.append(item)
            self.stats.submitted += 1
            if len(self._pending) >= self._max_batch_size:
                batch = self._take_pending_locked()
            elif self._timer is None:
                self._timer = asyncio.create_task(self._timer_flush())

        if batch is not None:
            self._spawn_dispatch(batch)

        try:
            return await future
        except asyncio.CancelledError:
            # Caller went away. Drop the item if it has not been dispatched yet
            # so we do not waste an upstream call on a result nobody awaits.
            async with self._lock:
                if item in self._pending:
                    self._pending.remove(item)
            raise

    def _take_pending_locked(self) -> list[_Item]:
        """Snapshot + clear the pending list and cancel the pending timer.

        Must be called while holding ``self._lock`` and never from inside the
        timer task itself (the timer clears its own reference instead).
        """
        batch = self._pending
        self._pending = []
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        return batch

    async def _timer_flush(self) -> None:
        try:
            await asyncio.sleep(self._max_wait_s)
        except asyncio.CancelledError:
            return

        batch: Optional[list[_Item]] = None
        async with self._lock:
            # We are the timer; clear our own reference without cancelling.
            self._timer = None
            if self._pending:
                batch = self._pending
                self._pending = []

        if batch:
            await self._dispatch(batch)

    def _spawn_dispatch(self, batch: list[_Item]) -> None:
        task = asyncio.create_task(self._dispatch(batch))
        # Hold a strong reference so the task is not GC'd mid-flight.
        self._dispatch_tasks.add(task)
        task.add_done_callback(self._dispatch_tasks.discard)

    async def _dispatch(self, batch: list[_Item]) -> None:
        if not batch:
            return

        self.stats.windows += 1
        self.stats.batch_sizes.append(len(batch))
        if self._on_flush is not None:
            try:
                self._on_flush(len(batch))
            except Exception:  # pragma: no cover - hook must never break dispatch
                pass

        async def run_one(item: _Item) -> Any:
            async with self._sem:
                return await self._dispatch_one(item.params)

        try:
            results = await asyncio.wait_for(
                asyncio.gather(
                    *(run_one(it) for it in batch),
                    return_exceptions=True,
                ),
                timeout=self._batch_timeout_s,
            )
        except asyncio.TimeoutError:
            self._fail_all(batch, TimeoutError("batch dispatch timed out"))
            return
        except Exception as exc:  # pragma: no cover - defensive catch-all
            self._fail_all(batch, exc)
            return

        for item, result in zip(batch, results):
            if item.future.done():
                continue
            if isinstance(result, BaseException):
                item.future.set_exception(result)
            else:
                item.future.set_result(result)

    @staticmethod
    def _fail_all(batch: list[_Item], exc: BaseException) -> None:
        for item in batch:
            if not item.future.done():
                item.future.set_exception(exc)

    async def aclose(self) -> None:
        """Cancel the timer and fail any pending Futures. Idempotent."""
        self._closed = True
        async with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            pending = self._pending
            self._pending = []
        self._fail_all(pending, RuntimeError("MicroBatcher is shutting down"))
