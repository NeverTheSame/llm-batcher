"""Tests for in-flight admission control and load shedding.

These run without a network call or API key. The limiter is exercised directly
with asyncio.run, and the request path is exercised through the app with a
stubbed limiter and a mocked upstream. Time-based waits are bounded so a logic
bug fails fast instead of hanging the suite.
"""

import asyncio
import importlib
import os
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.limiter import InflightLimiter, Rejected
from app.metrics import Metrics


def _run(coro):
    return asyncio.run(asyncio.wait_for(coro, timeout=5))


def test_constructor_validates_args():
    with pytest.raises(ValueError):
        InflightLimiter(max_inflight=0)
    with pytest.raises(ValueError):
        InflightLimiter(max_inflight=1, max_queue=-1)
    with pytest.raises(ValueError):
        InflightLimiter(max_inflight=1, acquire_timeout_s=-0.1)


def test_admits_up_to_max_inflight():
    async def scenario():
        lim = InflightLimiter(max_inflight=2, max_queue=0)
        await lim.acquire()
        await lim.acquire()
        snap = await lim.snapshot()
        assert snap["active"] == 2
        return lim

    lim = _run(scenario())
    assert lim._active == 2


def test_queue_full_is_shed_immediately():
    async def scenario():
        lim = InflightLimiter(max_inflight=1, max_queue=0)
        await lim.acquire()  # fills the only slot
        with pytest.raises(Rejected) as ei:
            await lim.acquire()  # no slot, no queue room
        assert ei.value.reason == "queue_full"

    _run(scenario())


def test_queue_timeout_when_slot_never_frees():
    async def scenario():
        lim = InflightLimiter(max_inflight=1, max_queue=1, acquire_timeout_s=0.05)
        await lim.acquire()  # holds the slot for the whole test
        with pytest.raises(Rejected) as ei:
            await lim.acquire()  # waits, then times out
        assert ei.value.reason == "queue_timeout"

    _run(scenario())


def test_release_frees_a_waiter():
    async def scenario():
        lim = InflightLimiter(max_inflight=1, max_queue=1, acquire_timeout_s=2.0)
        await lim.acquire()

        admitted = asyncio.Event()

        async def waiter():
            await lim.acquire()
            admitted.set()

        task = asyncio.create_task(waiter())
        await asyncio.sleep(0.01)  # let the waiter queue up
        assert (await lim.snapshot())["waiting"] == 1
        await lim.release()  # hand the slot to the waiter
        await asyncio.wait_for(admitted.wait(), timeout=1)
        await task
        snap = await lim.snapshot()
        assert snap["active"] == 1
        assert snap["waiting"] == 0

    _run(scenario())


def test_release_without_acquire_raises():
    async def scenario():
        lim = InflightLimiter(max_inflight=1)
        with pytest.raises(RuntimeError):
            await lim.release()

    _run(scenario())


def test_aclose_rejects_pending_waiters():
    async def scenario():
        lim = InflightLimiter(max_inflight=1, max_queue=1, acquire_timeout_s=2.0)
        await lim.acquire()

        outcome = {}

        async def waiter():
            try:
                await lim.acquire()
            except Rejected as exc:
                outcome["reason"] = exc.reason

        task = asyncio.create_task(waiter())
        await asyncio.sleep(0.01)
        await lim.aclose()
        await asyncio.wait_for(task, timeout=1)
        assert outcome["reason"] == "shutdown"

    _run(scenario())


def test_acquire_after_close_is_shutdown():
    async def scenario():
        lim = InflightLimiter(max_inflight=1)
        await lim.aclose()
        with pytest.raises(Rejected) as ei:
            await lim.acquire()
        assert ei.value.reason == "shutdown"

    _run(scenario())


def test_metrics_record_rejected_math():
    m = Metrics()
    m.record_rejected("queue_full")
    m.record_rejected("queue_full")
    m.record_rejected("queue_timeout")
    snap = m.snapshot()
    assert snap["rejected"] == {
        "total": 3,
        "reasons": {"queue_full": 2, "queue_timeout": 1},
    }
    # Shed requests stay out of the processed-request counters.
    assert snap["requests"] == {"total": 0, "success": 0, "error": 0}
    assert snap["latency"]["sample_count"] == 0


def test_limiter_disabled_by_default():
    from app.main import app  # default import has MAX_INFLIGHT off

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
    from app import main
    assert main._limiter is None


def test_endpoint_sheds_with_429_and_retry_after():
    class _FullLimiter:
        async def acquire(self):
            raise Rejected("queue_full")

        async def release(self):
            pass

        async def snapshot(self):
            return {"active": 1, "waiting": 0, "max_inflight": 1, "max_queue": 0}

        async def aclose(self):
            pass

    env = {
        "METRICS_ENABLED": "1",
        "MAX_INFLIGHT": "1",
        "RETRY_AFTER_S": "7",
        "ANTHROPIC_API_KEY": "sk-ant-test",
    }
    with patch.dict(os.environ, env):
        from app import main

        main = importlib.reload(main)
        with TestClient(main.app) as client:
            main._limiter = _FullLimiter()
            r = client.post(
                "/v1/chat/completions",
                json={
                    "model": "claude-3-5-haiku-latest",
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 8,
                },
            )
            assert r.status_code == 429
            assert r.headers["Retry-After"] == "7"
            snap = client.get("/metrics").json()
            assert snap["rejected"]["total"] == 1
            assert snap["rejected"]["reasons"] == {"queue_full": 1}
            assert snap["requests"]["total"] == 0

    from app import main as main_default

    importlib.reload(main_default)


def test_endpoint_admits_and_releases_when_limiter_on():
    fake_upstream = {
        "content": [{"type": "text", "text": "pong"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 1},
    }

    class _Resp:
        status_code = 200

        def json(self):
            return fake_upstream

    mock_client = AsyncMock()
    mock_client.post.return_value = _Resp()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = False

    env = {
        "METRICS_ENABLED": "1",
        "MAX_INFLIGHT": "4",
        "ANTHROPIC_API_KEY": "sk-ant-test",
    }
    with patch.dict(os.environ, env):
        from app import main

        main = importlib.reload(main)
        with patch.object(main.httpx, "AsyncClient", return_value=mock_client):
            with TestClient(main.app) as client:
                r = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "claude-3-5-haiku-latest",
                        "messages": [{"role": "user", "content": "ping"}],
                        "max_tokens": 8,
                    },
                )
                assert r.status_code == 200
                snap = client.get("/metrics").json()
                assert snap["requests"]["success"] == 1
                # Slot was released, so nothing is left in flight.
                assert snap["concurrency"]["active"] == 0
                assert snap["concurrency"]["max_inflight"] == 4

    from app import main as main_default

    importlib.reload(main_default)
