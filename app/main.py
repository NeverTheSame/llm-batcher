"""llm-batcher — Day 1: OpenAI-compatible proxy in front of the Anthropic Messages API.

This is the skeleton. It accepts an OpenAI-style POST /v1/chat/completions
request, translates it to the Anthropic Messages API, calls Anthropic, then
translates the response back into the OpenAI chat-completion shape.

No batching yet — that lands on Day 2. The point of Day 1 is a clean,
tested round-trip and a repo a hiring manager can read.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).parent.parent / ".env")

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "claude-3-5-haiku-latest")
DEFAULT_MAX_TOKENS = int(os.environ.get("DEFAULT_MAX_TOKENS", "1024"))
REQUEST_TIMEOUT_S = float(os.environ.get("REQUEST_TIMEOUT_S", "60"))

app = FastAPI(
    title="llm-batcher",
    version="0.1.0",
    description="OpenAI-compatible proxy in front of the Anthropic Messages API.",
)


# ---------------------------------------------------------------------------
# OpenAI-compatible request schema (the subset we support on Day 1).
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool = False


def _to_anthropic_payload(req: ChatCompletionRequest) -> dict[str, Any]:
    """Translate an OpenAI chat request into an Anthropic Messages payload.

    OpenAI carries the system prompt as a message with role="system".
    Anthropic takes it as a top-level `system` field, so we lift it out.
    """
    system_parts: list[str] = []
    messages: list[dict[str, str]] = []
    for m in req.messages:
        if m.role == "system":
            system_parts.append(m.content)
        else:
            # Anthropic only accepts "user" and "assistant".
            role = "assistant" if m.role == "assistant" else "user"
            messages.append({"role": role, "content": m.content})

    if not messages:
        raise HTTPException(
            status_code=400,
            detail="At least one non-system message is required.",
        )

    payload: dict[str, Any] = {
        "model": req.model or DEFAULT_MODEL,
        "max_tokens": req.max_tokens or DEFAULT_MAX_TOKENS,
        "messages": messages,
    }
    if system_parts:
        payload["system"] = "\n\n".join(system_parts)
    if req.temperature is not None:
        payload["temperature"] = req.temperature
    return payload


def _to_openai_response(anthropic_json: dict[str, Any], model: str) -> dict[str, Any]:
    """Translate an Anthropic Messages response into OpenAI chat-completion shape."""
    text = "".join(
        block.get("text", "")
        for block in anthropic_json.get("content", [])
        if block.get("type") == "text"
    )
    stop_reason = anthropic_json.get("stop_reason")
    finish_reason = "stop" if stop_reason in ("end_turn", "stop_sequence") else "length"
    usage = anthropic_json.get("usage", {})
    prompt_tokens = usage.get("input_tokens", 0)
    completion_tokens = usage.get("output_tokens", 0)

    return {
        "id": "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "llm-batcher", "version": "0.1.0"}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest) -> dict[str, Any]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY is not set.")
    if req.stream:
        raise HTTPException(status_code=400, detail="Streaming is not supported yet (Day 1).")

    payload = _to_anthropic_payload(req)
    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S) as client:
            resp = await client.post(ANTHROPIC_API_URL, json=payload, headers=headers)
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Upstream request failed: {exc}") from exc

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Anthropic API error: {resp.text}",
        )

    return _to_openai_response(resp.json(), payload["model"])
