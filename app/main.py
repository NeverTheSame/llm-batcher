"""llm-batcher: OpenAI-compatible proxy in front of the Anthropic Messages API.

It accepts an OpenAI-style POST /v1/chat/completions request, translates it to
the Anthropic Messages API, calls Anthropic, then translates the response back
into the OpenAI chat-completion shape.

An optional microbatching mode (BATCH_ENABLED=1) groups concurrent requests
into a short admission window and fans them out as concurrency-limited realtime
upstream calls. It is off by default, so the direct realtime path is unchanged
unless you opt in.

An optional cost and latency observatory (METRICS_ENABLED=1) records one sample
per request and exposes a JSON snapshot at GET /metrics. It is off by default
and the endpoint returns 404 until enabled.
"""

from __future__ import annotations

import contextlib
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.batcher import MicroBatcher
from app.limiter import InflightLimiter, Rejected
from app.metrics import Metrics

load_dotenv(Path(__file__).parent.parent / ".env")

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "claude-3-5-haiku-latest")
DEFAULT_MAX_TOKENS = int(os.environ.get("DEFAULT_MAX_TOKENS", "1024"))
REQUEST_TIMEOUT_S = float(os.environ.get("REQUEST_TIMEOUT_S", "60"))

# --- Optional microbatching configuration (opt-in, default off). ---
BATCH_ENABLED = os.environ.get("BATCH_ENABLED", "0").lower() in ("1", "true", "yes")
BATCH_MAX_SIZE = int(os.environ.get("BATCH_MAX_SIZE", "16"))
BATCH_MAX_WAIT_MS = int(os.environ.get("BATCH_MAX_WAIT_MS", "20"))
BATCH_MAX_CONCURRENCY = int(os.environ.get("BATCH_MAX_CONCURRENCY", "8"))
BATCH_TIMEOUT_S = float(os.environ.get("BATCH_TIMEOUT_S", str(REQUEST_TIMEOUT_S)))

# --- Optional cost and latency observatory (opt-in, default off). ---
METRICS_ENABLED = os.environ.get("METRICS_ENABLED", "0").lower() in ("1", "true", "yes")
METRICS_LATENCY_WINDOW = int(os.environ.get("METRICS_LATENCY_WINDOW", "1024"))

# --- Optional in-flight admission control / backpressure (opt-in, default off). ---
# MAX_INFLIGHT <= 0 disables the limiter, so the request path is unchanged.
MAX_INFLIGHT = int(os.environ.get("MAX_INFLIGHT", "0"))
LIMITER_ENABLED = MAX_INFLIGHT > 0
MAX_QUEUE = max(0, int(os.environ.get("MAX_QUEUE", "0")))
ACQUIRE_TIMEOUT_S = max(0.0, float(os.environ.get("ACQUIRE_TIMEOUT_S", "0.5")))
RETRY_AFTER_S = max(0, int(os.environ.get("RETRY_AFTER_S", "1")))

_shared_client: Optional[httpx.AsyncClient] = None
_batcher: Optional[MicroBatcher] = None
_metrics: Optional[Metrics] = None
_limiter: Optional[InflightLimiter] = None


async def _anthropic_dispatch_one(payload: dict[str, Any]) -> dict[str, Any]:
    """Perform one realtime Anthropic Messages call on the shared client."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY is not set.")
    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    try:
        resp = await _shared_client.post(  # type: ignore[union-attr]
            ANTHROPIC_API_URL, json=payload, headers=headers
        )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502, detail=f"Upstream request failed: {exc}"
        ) from exc
    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Anthropic API error: {resp.text}",
        )
    return resp.json()


@contextlib.asynccontextmanager
async def _lifespan(_app: "FastAPI"):
    global _shared_client, _batcher, _metrics, _limiter
    if METRICS_ENABLED:
        _metrics = Metrics(latency_window=METRICS_LATENCY_WINDOW)
    if LIMITER_ENABLED:
        _limiter = InflightLimiter(
            max_inflight=MAX_INFLIGHT,
            max_queue=MAX_QUEUE,
            acquire_timeout_s=ACQUIRE_TIMEOUT_S,
        )
    if BATCH_ENABLED:
        _shared_client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S)
        _batcher = MicroBatcher(
            _anthropic_dispatch_one,
            max_batch_size=BATCH_MAX_SIZE,
            max_wait_ms=BATCH_MAX_WAIT_MS,
            max_concurrency=BATCH_MAX_CONCURRENCY,
            batch_timeout_s=BATCH_TIMEOUT_S,
        )
    try:
        yield
    finally:
        if _limiter is not None:
            await _limiter.aclose()
            _limiter = None
        if _batcher is not None:
            await _batcher.aclose()
            _batcher = None
        if _shared_client is not None:
            with contextlib.suppress(Exception):
                await _shared_client.aclose()
            _shared_client = None
        _metrics = None


app = FastAPI(
    title="llm-batcher",
    version="0.1.0",
    description="OpenAI-compatible proxy in front of the Anthropic Messages API.",
    lifespan=_lifespan,
)


# ---------------------------------------------------------------------------
# OpenAI-compatible request schema (the subset we support).
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


@app.get("/metrics")
async def metrics() -> dict[str, Any]:
    if not METRICS_ENABLED or _metrics is None:
        raise HTTPException(status_code=404, detail="Metrics are not enabled.")
    snap = _metrics.snapshot()
    if _limiter is not None:
        snap["concurrency"] = await _limiter.snapshot()
    return snap


async def _direct_dispatch(payload: dict[str, Any]) -> dict[str, Any]:
    """Perform one realtime Anthropic call on a per-request client (batching off)."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY is not set.")

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

    return resp.json()


async def _dispatch_and_record(
    payload: dict[str, Any], model: str, start: float
) -> dict[str, Any]:
    """Run the upstream dispatch (batched or direct) and record metrics.

    `start` is captured by the caller before any admission wait, so the recorded
    latency is end-to-end for admitted requests, including time spent queued.
    """
    try:
        if BATCH_ENABLED and _batcher is not None:
            anthropic_json = await _batcher.submit(payload)
        else:
            anthropic_json = await _direct_dispatch(payload)
    except Exception:
        if _metrics is not None:
            _metrics.record_error((time.perf_counter() - start) * 1000.0)
        raise

    latency_ms = (time.perf_counter() - start) * 1000.0
    if _metrics is not None:
        usage = anthropic_json.get("usage", {})
        _metrics.record_success(
            model,
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
            latency_ms,
        )
    return _to_openai_response(anthropic_json, model)


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest) -> dict[str, Any]:
    if req.stream:
        raise HTTPException(status_code=400, detail="Streaming is not supported yet.")

    payload = _to_anthropic_payload(req)
    model = payload["model"]
    start = time.perf_counter()

    if _limiter is None:
        return await _dispatch_and_record(payload, model, start)

    try:
        await _limiter.acquire()
    except Rejected as exc:
        if _metrics is not None:
            _metrics.record_rejected(exc.reason)
        raise HTTPException(
            status_code=429,
            detail=f"Request shed by admission control ({exc.reason}).",
            headers={"Retry-After": str(RETRY_AFTER_S)},
        ) from exc

    try:
        return await _dispatch_and_record(payload, model, start)
    finally:
        await _limiter.release()
