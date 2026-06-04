"""Tests for the cost and latency observatory.

These run without a network call or API key. They exercise the Metrics registry
math directly and the /metrics endpoint through the app, all deterministic.
"""

import importlib
import os
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.metrics import Metrics, _percentile, _price_for


def test_percentile_edges():
    assert _percentile([42.0], 50) == 42.0
    assert _percentile([42.0], 99) == 42.0
    vals = [float(i) for i in range(1, 101)]  # 1..100 ascending
    # Nearest-rank over (n-1): index = round(pct/100 * 99), clamped.
    assert _percentile(vals, 50) == 51.0
    assert _percentile(vals, 95) == 95.0
    assert _percentile(vals, 99) == 99.0
    assert _percentile(vals, 100) == 100.0


def test_price_lookup_matches_family_and_aliases():
    assert _price_for("claude-3-5-haiku-latest") == (0.80, 4.00)
    assert _price_for("claude-3-5-haiku-20241022") == (0.80, 4.00)
    assert _price_for("claude-3-opus-latest") == (15.00, 75.00)
    assert _price_for("some-unknown-model") is None


def test_cost_math_for_known_model():
    m = Metrics()
    # 1,000,000 input tokens at 0.80 + 1,000,000 output at 4.00 = 4.80 USD.
    m.record_success("claude-3-5-haiku-latest", 1_000_000, 1_000_000, latency_ms=12.0)
    snap = m.snapshot()
    assert snap["requests"] == {"total": 1, "success": 1, "error": 0}
    assert snap["tokens"] == {"input": 1_000_000, "output": 1_000_000, "total": 2_000_000}
    assert snap["cost"]["estimated_cost_usd"] == 4.80
    assert snap["cost"]["estimate_complete"] is True
    assert snap["latency"]["sample_count"] == 1
    assert snap["latency"]["p95_ms"] == 12.0


def test_unknown_model_is_unpriced_not_zero():
    m = Metrics()
    m.record_success("mystery-model", 100, 50, latency_ms=5.0)
    snap = m.snapshot()
    assert snap["cost"]["estimated_cost_usd"] == 0.0
    assert snap["cost"]["estimate_complete"] is False
    assert snap["cost"]["unpriced_request_count"] == 1
    assert snap["cost"]["unpriced_input_tokens"] == 100
    assert snap["cost"]["unpriced_output_tokens"] == 50
    assert snap["cost"]["unpriced_models"] == {"mystery-model": 1}
    # Unpriced usage is not added to the priced token totals.
    assert snap["tokens"]["total"] == 0


def test_error_recording_counts_and_latency():
    m = Metrics()
    m.record_error(latency_ms=30.0)
    m.record_success("claude-3-5-haiku-latest", 10, 5, latency_ms=10.0)
    snap = m.snapshot()
    assert snap["requests"] == {"total": 2, "success": 1, "error": 1}
    assert snap["latency"]["sample_count"] == 2
    assert snap["latency"]["max_ms"] == 30.0


def test_empty_snapshot_has_null_percentiles():
    snap = Metrics().snapshot()
    assert snap["latency"]["sample_count"] == 0
    assert snap["latency"]["p50_ms"] is None
    assert snap["latency"]["max_ms"] is None
    assert snap["requests"]["total"] == 0


def test_latency_window_is_clamped():
    m = Metrics(latency_window=0)
    assert m._window == 1
    m2 = Metrics(latency_window=10**9)
    assert m2._window == 100_000


def test_metrics_endpoint_404_by_default():
    # Default import has METRICS_ENABLED off, so the route must 404.
    from app.main import app

    with TestClient(app) as client:
        assert client.get("/metrics").status_code == 404


def test_metrics_endpoint_reports_after_request_when_enabled():
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

    with patch.dict(os.environ, {"METRICS_ENABLED": "1", "ANTHROPIC_API_KEY": "sk-ant-test"}):
        from app import main

        main = importlib.reload(main)
        with patch.object(main.httpx, "AsyncClient", return_value=mock_client):
            with TestClient(main.app) as client:
                assert client.get("/metrics").json()["requests"]["total"] == 0
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
                assert snap["requests"]["total"] == 1
                assert snap["requests"]["success"] == 1
                assert snap["tokens"]["total"] == 6
                assert snap["cost"]["estimated_cost_usd"] > 0

    # Restore the default (metrics-off) module for any later tests.
    from app import main as main_default

    importlib.reload(main_default)
