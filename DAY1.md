# Day 1 — Building an OpenAI-Compatible Proxy in Front of Claude

*A Senior SRE's first step from "infrastructure for databases" to "infrastructure for LLMs."*

---

## TL;DR

In one ~90-minute session I built **`llm-batcher`** — a small FastAPI service that
speaks the **OpenAI Chat Completions API** on the front and the **Anthropic
Messages API** on the back. Any tool that already talks to OpenAI can point at it
and get **Claude** instead, with zero client changes.

It's deliberately tiny today (a clean pass-through proxy with request/response
translation and a real test suite), but it's the seed of something that mirrors
what an **inference-serving team** actually runs in production: adaptive batching,
routing, backpressure, and a cost/latency observatory.

- **Repo:** https://github.com/NeverTheSame/llm-batcher
- **Stack:** Python 3.12, FastAPI, httpx (async), Pydantic v2, Docker
- **Tests:** 5 unit tests, mocked upstream, no API key required, all green

---

## Why I'm building this

I've spent 12 years as an SRE keeping highly-available, distributed systems alive —
Kubernetes, Terraform, observability, incident response, the on-call pager. I'm now
deliberately repositioning toward **AI inference systems**, the part of the ML stack
that is *least* about training models and *most* about the thing I already do well:
**serving traffic reliably, cheaply, and fast at scale.**

The insight that unlocked this project: **an LLM inference proxy is just a reverse
proxy with interesting backpressure.** The hard parts — connection pooling, request
batching, queueing under load, latency budgets, cost telemetry — are exactly the
problems I've solved for databases and microservices. I don't need to "learn ML from
scratch." I need to point skills I already have at a new kind of upstream.

So `llm-batcher` is the first brick. The rule I'm holding myself to:

> **No learning goal without a deliverable. No project without a benchmark.
> No benchmark without a README. No README without interview talking points.**

---

## What it does (today)

```
┌────────────────┐   OpenAI-shaped    ┌───────────────┐   Anthropic-shaped   ┌──────────────────┐
│ OpenAI client  │ ─ POST /v1/chat/ ─▶│  llm-batcher  │ ─ POST /v1/messages ▶│  Anthropic API   │
│ (SDK, curl,    │   completions      │  (translates  │                      │  (Claude models) │
│  LangChain…)   │◀─ chat.completion ─│   both ways)  │◀─ messages response ─│                  │
└────────────────┘                    └───────────────┘                      └──────────────────┘
```

Concretely, the proxy:

1. **Accepts an OpenAI `POST /v1/chat/completions` request** — the de-facto
   standard shape that nearly every LLM tool already emits.
2. **Translates the request to Anthropic's Messages format.** The two APIs differ
   in subtle, important ways (more below).
3. **Forwards it** to `https://api.anthropic.com/v1/messages` over async httpx with
   a timeout and proper error propagation.
4. **Translates Claude's response back** into the OpenAI `chat.completion` shape —
   including `usage` token counts and a mapped `finish_reason` — so the caller never
   knows it wasn't talking to OpenAI.
5. Exposes **`GET /health`** for liveness checks (the SRE in me refuses to ship a
   service without one).

---

## The interesting part: the two APIs are *almost* the same

This is the kind of detail that looks trivial until you actually wire it up. OpenAI
and Anthropic both do "chat," but the contracts diverge in ways that will bite you:

### 1. The system prompt lives in a different place

**OpenAI** carries the system prompt as a *message* with `role: "system"` inside the
`messages` array:

```json
{
  "messages": [
    {"role": "system", "content": "You are terse."},
    {"role": "user", "content": "Hello"}
  ]
}
```

**Anthropic** does *not* allow a `system` role inside `messages`. The system prompt
is a **top-level field**:

```json
{
  "system": "You are terse.",
  "messages": [
    {"role": "user", "content": "Hello"}
  ]
}
```

So the translator has to *lift* every `system` message out of the array and join
them into the top-level `system` field. Here's the actual logic:

```python
def _to_anthropic_payload(req: ChatCompletionRequest) -> dict[str, Any]:
    system_parts: list[str] = []
    messages: list[dict[str, str]] = []
    for m in req.messages:
        if m.role == "system":
            system_parts.append(m.content)
        else:
            role = "assistant" if m.role == "assistant" else "user"
            messages.append({"role": role, "content": m.content})

    if not messages:
        raise HTTPException(status_code=400,
                            detail="At least one non-system message is required.")

    payload = {
        "model": req.model or DEFAULT_MODEL,
        "max_tokens": req.max_tokens or DEFAULT_MAX_TOKENS,
        "messages": messages,
    }
    if system_parts:
        payload["system"] = "\n\n".join(system_parts)
    return payload
```

### 2. `max_tokens` is optional in OpenAI but **required** in Anthropic

If you forward an OpenAI request that omits `max_tokens`, Anthropic rejects it. The
proxy defaults it (`DEFAULT_MAX_TOKENS`, configurable via env) so callers don't have
to care.

### 3. The response envelope is shaped differently

**Anthropic** returns content as a list of typed blocks and reports stop reasons and
token usage with its own names:

```json
{
  "content": [{"type": "text", "text": "pong"}],
  "stop_reason": "end_turn",
  "usage": {"input_tokens": 5, "output_tokens": 2}
}
```

**OpenAI** clients expect `choices[].message.content`, a `finish_reason`, and
`usage` keyed as `prompt_tokens` / `completion_tokens` / `total_tokens`. The
translator flattens the content blocks, maps the stop reason, and renames the usage
fields:

```python
def _to_openai_response(anthropic_json, model):
    text = "".join(b.get("text", "")
                   for b in anthropic_json.get("content", [])
                   if b.get("type") == "text")
    stop = anthropic_json.get("stop_reason")
    finish_reason = "stop" if stop in ("end_turn", "stop_sequence") else "length"
    usage = anthropic_json.get("usage", {})
    pt, ct = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": text},
                     "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": pt, "completion_tokens": ct,
                  "total_tokens": pt + ct},
    }
```

### 4. Auth and versioning headers differ

OpenAI uses `Authorization: Bearer <key>`. Anthropic uses a custom header pair:

```python
headers = {
    "x-api-key": api_key,
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
}
```

Forgetting `anthropic-version` is a classic first-time 400. Pinning it in one place
means I'll never chase that bug across call sites.

---

## Engineering choices (and the SRE reasoning behind them)

**Async all the way down (`httpx.AsyncClient`).** A proxy's whole job is to wait on
an upstream. Blocking I/O would cap concurrency at the worker count. Async means one
process can hold thousands of in-flight requests cheaply — which matters enormously
once Day 2 adds batching, because batching is *literally* "hold requests and wait."

**Timeouts and explicit error mapping.** Every upstream call has a timeout
(`REQUEST_TIMEOUT_S`). Network failures become `502 Bad Gateway`; a missing key
becomes a clear `500` with a human message; Anthropic's own errors are propagated
with their status code. No silent hangs, no leaked stack traces. This is the
difference between "works on my laptop" and "survives a bad afternoon in prod."

**Config via environment, secrets via `.env` (git-ignored).** `ANTHROPIC_API_KEY`,
`DEFAULT_MODEL`, `DEFAULT_MAX_TOKENS`, and `REQUEST_TIMEOUT_S` are all env-driven.
The repo ships a `.env.example` template and a `.gitignore` that keeps real keys out
of version control — verified before I made the repo public.

**Pydantic v2 request models.** The incoming request is validated against a typed
schema, so malformed input fails fast with a 422 instead of exploding three layers
deep.

---

## Testing without spending a cent (or leaking a key)

I wanted a test suite that runs in CI **with no API key and no network**, because:

- Tests that hit the real API are slow, flaky, and cost money.
- A key in CI is a key that can leak.

So the upstream Anthropic call is **mocked**, and the deterministic translation
logic is tested directly. Five tests cover:

1. System-message lifting into the top-level `system` field
2. Rejecting a request with no non-system message
3. Response-shape translation (content flattening, `finish_reason`, token math)
4. A full **mocked round-trip** through the FastAPI endpoint
5. The `/health` endpoint

```
$ pytest -q
.....                                                    [100%]
5 passed in 1.53s
```

For the real thing, there's a `tests/smoke.sh` that does a live round-trip once you
drop in an actual key — but it's never required for the suite to pass.

---

## Try it yourself

```bash
git clone https://github.com/NeverTheSame/llm-batcher
cd llm-batcher

python3 -m venv venv
./venv/bin/pip install -r requirements.txt

cp .env.example .env          # then paste your ANTHROPIC_API_KEY
./venv/bin/uvicorn app.main:app --reload --port 8000
```

Now talk to Claude using an **OpenAI-shaped** request:

```bash
curl -sS -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-haiku-latest",
    "messages": [
      {"role": "system", "content": "You are terse."},
      {"role": "user", "content": "Reply with exactly: pong"}
    ],
    "max_tokens": 16
  }'
```

Or run it in a container:

```bash
docker compose up --build
```

---

## The roadmap (why it's called *llm-batcher*)

Today it's a pass-through. The name is a promise about where it's going:

| Day | What it adds | The skill it proves |
|-----|--------------|---------------------|
| **1 ✅** | Pass-through proxy + OpenAI⇄Anthropic translation + tests | API design, clean service boundaries |
| **2** | **Request batching** — accumulate concurrent requests into one upstream call window | Queueing, latency/throughput tradeoffs |
| **3** | **Cost & latency observatory** — per-request metrics, p50/p95/p99, $ estimate | Observability, telemetry |
| **4** | **Backpressure + concurrency caps** | The core SRE skill: graceful degradation under load |
| **5** | **Benchmark harness** — throughput vs. latency vs. cost under synthetic load | Performance engineering, evidence |

Each step is a commit, a benchmark number, and a paragraph I can defend in an
interview. The point isn't to build a production gateway (those exist). The point is
to **demonstrate, in public, that I think about LLM serving the way an inference
engineer does** — in terms of tail latency, fleet efficiency, and dollars per
million tokens, not model accuracy.

---

## What I'd tell an interviewer about Day 1

- **"Why a proxy?"** Because the OpenAI API shape is the lingua franca; meeting
  clients where they are removes all adoption friction, and a proxy is the natural
  place to add batching, caching, and observability without touching callers.
- **"What surprised you?"** How much of the real work is *contract translation* —
  the system-prompt placement, the required `max_tokens`, the response envelope.
  Small mismatches, but each one is a production incident if you miss it.
- **"What would break first under load?"** Right now: nothing batches, so I'd burn
  one upstream connection per request and hit Anthropic's rate limits fast. That's
  exactly what Day 2 fixes — and why the project is named for the thing it doesn't
  do yet.

---

*This is Day 1 of a build-in-public series. Follow the repo to watch a reverse-proxy
turn into an inference-serving primitive, one 90-minute session at a time.*
