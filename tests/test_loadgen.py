"""No-network tests for the load harness in bench/loadgen.py.

These cover the pure summary math, the result reconciliation logic on a fake
result, and two tiny in-process smoke runs (one closed loop baseline, one
limiter overload) that exercise the real app over the ASGI transport with a
stubbed upstream. Time-based runs are bounded with asyncio.wait_for so a hang
fails loudly instead of stalling the suite. The module reloads app.main with a
clean env at the end so it cannot pollute other test modules.
"""

from __future__ import annotations

import asyncio
import importlib
import os

import pytest

from bench import loadgen
from bench.loadgen import ScenarioConfig


def test_percentile_app_matches_app_semantics():
    assert loadgen.percentile_app([], 50) is None
    assert loadgen.percentile_app([42.0], 99) == 42.0
    vals = [float(i) for i in range(1, 101)]  # 1..100
    # rank = round(p/100 * (n-1)); n=100 -> round(0.5*99)=50 -> vals[50]=51
    assert loadgen.percentile_app(vals, 50) == 51.0
    assert loadgen.percentile_app(vals, 95) == 95.0
    assert loadgen.percentile_app(vals, 99) == 99.0


def test_summarize_latencies_shape_and_values():
    out = loadgen.summarize_latencies([10.0, 20.0, 30.0, 40.0])
    assert out["count"] == 4
    assert out["max_ms"] == 40.0
    assert out["mean_ms"] == 25.0
    for key in ("p50_ms", "p95_ms", "p99_ms"):
        assert out[key] is not None

    empty = loadgen.summarize_latencies([])
    assert empty["count"] == 0
    assert empty["p50_ms"] is None


def test_build_result_reconciliation_flags():
    cfg = ScenarioConfig(name="fake", mode="closed", concurrency=1)
    records = [(200, 5.0), (200, 6.0), (429, 1.0)]
    snap = {
        "requests": {"total": 2, "success": 2, "error": 0},
        "rejected": {"total": 1, "reasons": {"queue_full": 1}},
        "latency": {"p50_ms": 5.0, "p95_ms": 6.0, "p99_ms": 6.0, "max_ms": 6.0},
        "tokens": {"total": 2000},
        "cost": {"estimated_cost_usd": 0.01},
        "concurrency": {"active": 0, "waiting": 0, "max_inflight": 1, "max_queue": 0},
    }
    res = loadgen._build_result(cfg, records, 1.0, snap, None, None, None, 0)
    assert res["client"]["ok_200"] == 2
    assert res["client"]["rejected_429"] == 1
    assert res["reconciliation"]["client_200_eq_server_success"] is True
    assert res["reconciliation"]["client_429_eq_server_rejected"] is True

    bad = loadgen._build_result(cfg, records, 1.0,
                                {**snap, "requests": {"success": 99}}, None, None, None, 0)
    assert bad["reconciliation"]["client_200_eq_server_success"] is False


def test_closed_loop_smoke_baseline():
    cfg = ScenarioConfig(
        name="smoke_baseline",
        mode="closed",
        concurrency=2,
        duration_s=0.2,
        warmup_s=0.05,
        service_time_ms=2.0,
    )
    res = asyncio.run(asyncio.wait_for(loadgen.run_scenario(cfg), timeout=10))
    assert res["client"]["ok_200"] > 0
    assert res["client"]["rejected_429"] == 0
    assert res["reconciliation"]["client_200_eq_server_success"] is True
    assert res["leaked_tasks"] == 0


def test_open_loop_limiter_sheds_load():
    cfg = ScenarioConfig(
        name="smoke_limiter",
        mode="open",
        offered_rps=300.0,
        duration_s=0.4,
        warmup_s=0.05,
        service_time_ms=25.0,
        max_inflight=1,
        max_queue=0,
        acquire_timeout_s=0.02,
    )
    res = asyncio.run(asyncio.wait_for(loadgen.run_scenario(cfg), timeout=10))
    assert res["client"]["rejected_429"] > 0
    assert res["reconciliation"]["client_429_eq_server_rejected"] is True
    assert res["leaked_tasks"] == 0


@pytest.fixture(autouse=True)
def _restore_main_after():
    yield
    for key in ("BATCH_ENABLED", "METRICS_ENABLED", "MAX_INFLIGHT",
                "MAX_QUEUE", "ACQUIRE_TIMEOUT_S", "ANTHROPIC_API_KEY"):
        os.environ.pop(key, None)
    from app import main as _main
    importlib.reload(_main)
