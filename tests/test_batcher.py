"""Unit tests for the microbatching accumulator.

These run without a network call or API key. The upstream dispatcher is a
plain async fake, so the suite is deterministic and CI-friendly. Each test
drives the batcher through ``asyncio.run`` to avoid extra async-test plugins.
"""

import asyncio

import pytest

from app.batcher import MicroBatcher


def test_flush_by_size_groups_into_one_window():
    """Reaching max_batch_size flushes immediately in a single window."""
    flushes: list[int] = []

    async def dispatch_one(params):
        return {"echo": params["n"]}

    async def scenario():
        batcher = MicroBatcher(
            dispatch_one,
            max_batch_size=3,
            max_wait_ms=10_000,  # large, so the timer can't be what flushes
            on_flush=flushes.append,
        )
        results = await asyncio.gather(
            batcher.submit({"n": 0}),
            batcher.submit({"n": 1}),
            batcher.submit({"n": 2}),
        )
        return results

    results = asyncio.run(scenario())
    assert results == [{"echo": 0}, {"echo": 1}, {"echo": 2}]
    assert flushes == [3]


def test_flush_by_time_when_window_not_full():
    """A partial window flushes after max_wait_ms via the timer."""
    flushes: list[int] = []

    async def dispatch_one(params):
        return {"echo": params["n"]}

    async def scenario():
        batcher = MicroBatcher(
            dispatch_one,
            max_batch_size=100,  # never reached
            max_wait_ms=20,
            on_flush=flushes.append,
        )
        results = await asyncio.gather(
            batcher.submit({"n": 0}),
            batcher.submit({"n": 1}),
        )
        return results

    results = asyncio.run(asyncio.wait_for(scenario(), timeout=5))
    assert results == [{"echo": 0}, {"echo": 1}]
    assert flushes == [2]


def test_results_route_to_the_correct_caller():
    """Each submitter gets its own result even when shuffled by the dispatcher."""

    async def dispatch_one(params):
        # Vary completion order to prove routing is by item, not by arrival.
        await asyncio.sleep((5 - params["n"]) / 1000.0)
        return {"value": params["n"] * 10}

    async def scenario():
        batcher = MicroBatcher(dispatch_one, max_batch_size=5, max_wait_ms=10_000)
        return await asyncio.gather(*(batcher.submit({"n": i}) for i in range(5)))

    results = asyncio.run(asyncio.wait_for(scenario(), timeout=5))
    assert results == [{"value": i * 10} for i in range(5)]


def test_per_item_error_is_isolated():
    """One failing item does not poison the rest of the window."""

    async def dispatch_one(params):
        if params["n"] == 1:
            raise ValueError("boom")
        return {"ok": params["n"]}

    async def scenario():
        batcher = MicroBatcher(dispatch_one, max_batch_size=3, max_wait_ms=10_000)
        return await asyncio.gather(
            batcher.submit({"n": 0}),
            batcher.submit({"n": 1}),
            batcher.submit({"n": 2}),
            return_exceptions=True,
        )

    results = asyncio.run(asyncio.wait_for(scenario(), timeout=5))
    assert results[0] == {"ok": 0}
    assert isinstance(results[1], ValueError)
    assert results[2] == {"ok": 2}


def test_batch_timeout_fails_all_pending():
    """A stuck dispatcher trips the batch timeout and fails every caller."""

    async def dispatch_one(params):
        await asyncio.sleep(10)  # longer than the batch timeout
        return {"never": True}

    async def scenario():
        batcher = MicroBatcher(
            dispatch_one,
            max_batch_size=2,
            max_wait_ms=10_000,
            batch_timeout_s=0.05,
        )
        return await asyncio.gather(
            batcher.submit({"n": 0}),
            batcher.submit({"n": 1}),
            return_exceptions=True,
        )

    results = asyncio.run(asyncio.wait_for(scenario(), timeout=5))
    assert all(isinstance(r, TimeoutError) for r in results)


def test_aclose_fails_pending_futures():
    """Shutdown resolves any still-queued Futures instead of hanging."""

    async def dispatch_one(params):  # pragma: no cover - never reached
        return {"ok": True}

    async def scenario():
        batcher = MicroBatcher(dispatch_one, max_batch_size=10, max_wait_ms=10_000)
        task = asyncio.ensure_future(batcher.submit({"n": 0}))
        await asyncio.sleep(0)  # let submit enqueue the item
        await batcher.aclose()
        with pytest.raises(RuntimeError):
            await task

    asyncio.run(asyncio.wait_for(scenario(), timeout=5))
