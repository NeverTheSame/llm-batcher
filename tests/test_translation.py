"""Unit tests for the OpenAI<->Anthropic translation layer.

These run without a network call or API key: the upstream Anthropic request
is mocked, so the suite is deterministic and CI-friendly.
"""

import os
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app import main
from app.main import (
    ChatCompletionRequest,
    ChatMessage,
    _to_anthropic_payload,
    _to_openai_response,
    app,
)

client = TestClient(app)


def test_payload_lifts_system_message():
    req = ChatCompletionRequest(
        model="claude-3-5-haiku-latest",
        messages=[
            ChatMessage(role="system", content="be terse"),
            ChatMessage(role="user", content="hi"),
        ],
        max_tokens=10,
    )
    payload = _to_anthropic_payload(req)
    assert payload["system"] == "be terse"
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert payload["max_tokens"] == 10
    assert payload["model"] == "claude-3-5-haiku-latest"


def test_payload_requires_a_non_system_message():
    req = ChatCompletionRequest(messages=[ChatMessage(role="system", content="x")])
    with pytest.raises(Exception):
        _to_anthropic_payload(req)


def test_response_translation_shape():
    anthropic_json = {
        "content": [{"type": "text", "text": "pong"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 2},
    }
    out = _to_openai_response(anthropic_json, "claude-3-5-haiku-latest")
    assert out["object"] == "chat.completion"
    assert out["choices"][0]["message"]["content"] == "pong"
    assert out["choices"][0]["finish_reason"] == "stop"
    assert out["usage"]["total_tokens"] == 7


def test_endpoint_round_trip_mocked():
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

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
        with patch.object(main.httpx, "AsyncClient", return_value=mock_client):
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 8,
                },
            )

    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "pong"


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
