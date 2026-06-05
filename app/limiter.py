"""Global in-flight admission control for the request path.

This caps how many client requests the proxy will process at once, which is a
different concern from the batcher's cap on concurrent upstream calls. Under a
load spike an unbounded request queue is the failure mode that hurts: memory
grows and every request's latency climbs together. The limiter admits up to
`max_inflight` requests, lets a bounded number (`max_queue`) wait briefly for a
slot, and sheds the rest fast so callers can back off instead of piling on.

State is kept explicitly under an asyncio.Condition rather than leaning on a
Semaphore's internal counters, so admission, queue accounting, timeout, and
shutdown are each a single atomic transition that cannot disagree with each
other under cancellation.
"""

from __future__ import annotations

import asyncio
import contextlib


class Rejected(Exception):
    """Raised when a request cannot be admitted. `reason` explains why."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class InflightLimiter:
    """Bounded-concurrency admission gate with a short, bounded wait queue."""

    def __init__(
        self,
        max_inflight: int,
        max_queue: int = 0,
        acquire_timeout_s: float = 0.5,
    ) -> None:
        if max_inflight <= 0:
            raise ValueError("max_inflight must be > 0")
        if max_queue < 0:
            raise ValueError("max_queue must be >= 0")
        if acquire_timeout_s < 0:
            raise ValueError("acquire_timeout_s must be >= 0")

        self._max_inflight = max_inflight
        self._max_queue = max_queue
        self._timeout = acquire_timeout_s
        self._active = 0
        self._waiting = 0
        self._closed = False
        self._cond = asyncio.Condition()

    async def acquire(self) -> None:
        """Admit the caller or raise Rejected. Pair every success with release()."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._timeout

        async with self._cond:
            if self._closed:
                raise Rejected("shutdown")

            # Fast path: a slot is free and nobody is already queued ahead of us.
            if self._active < self._max_inflight and self._waiting == 0:
                self._active += 1
                return

            if self._waiting >= self._max_queue:
                raise Rejected("queue_full")

            self._waiting += 1
            admitted = False
            try:
                while True:
                    if self._closed:
                        raise Rejected("shutdown")
                    if self._active < self._max_inflight:
                        self._active += 1
                        admitted = True
                        return
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        raise Rejected("queue_timeout")
                    try:
                        await asyncio.wait_for(self._cond.wait(), remaining)
                    except asyncio.TimeoutError:
                        raise Rejected("queue_timeout")
            finally:
                self._waiting -= 1
                # If we were woken but left without taking the slot, hand the
                # wake-up to the next waiter so capacity is never stranded.
                if (
                    not admitted
                    and not self._closed
                    and self._active < self._max_inflight
                    and self._waiting > 0
                ):
                    self._cond.notify(1)

    async def release(self) -> None:
        async with self._cond:
            if self._active <= 0:
                raise RuntimeError("release() called without a matching acquire()")
            self._active -= 1
            self._cond.notify(1)

    async def aclose(self) -> None:
        """Reject any pending waiters and refuse new admissions."""
        async with self._cond:
            self._closed = True
            self._cond.notify_all()

    @contextlib.asynccontextmanager
    async def slot(self):
        await self.acquire()
        try:
            yield
        finally:
            await self.release()

    async def snapshot(self) -> dict:
        async with self._cond:
            return {
                "active": self._active,
                "waiting": self._waiting,
                "max_inflight": self._max_inflight,
                "max_queue": self._max_queue,
            }
