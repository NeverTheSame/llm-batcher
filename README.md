# llm-batcher

An **OpenAI-compatible proxy** in front of the **Anthropic Messages API**: the
first primitive an inference-serving team ships. Point any OpenAI client at it,
get Claude back, with a clean path to add adaptive batching, routing, and a
cost/latency observatory.

> **Status:** Working pass-through proxy + request/response translation, plus
> opt-in microbatching. Fully unit-tested.

## Why this exists

I'm a Senior SRE repositioning toward AI inference systems. This repo applies the
exact skills I already have (distributed systems, proxies, observability) to LLM
serving instead of databases. The roadmap turns it into a realistic miniature of
what an inference proxy fleet does in production.

## What it does today

```
OpenAI client ──▶  POST /v1/chat/completions  ──▶  Anthropic Messages API
                   (this proxy translates                ▲
                    request + response shapes)            │
                                  ◀──────────────────────┘
```

- Accepts OpenAI-style `POST /v1/chat/completions`
- Lifts `system` messages into Anthropic's top-level `system` field
- Maps roles, forwards to `https://api.anthropic.com/v1/messages`
- Translates the Anthropic response back into the OpenAI chat-completion shape
  (incl. `usage` token counts and `finish_reason`)
- `GET /health` for liveness

## Roadmap

| Day | Adds |
|-----|------|
| 1 ✅ | Pass-through proxy + OpenAI/Anthropic translation + tests |
| 2 ✅ | Microbatching: group concurrent requests into a short admission window, fan out under a concurrency cap |
| 3 | Cost & latency observatory: per-request metrics, p50/p95, $ estimate |
| 4 | Backpressure + concurrency caps (the SRE part) |
| 5 | Benchmark harness: throughput vs. latency vs. cost under load |

## Microbatching (opt-in)

Real inference proxies rarely send one upstream call per client request under
load. They group requests that arrive close together, then dispatch them
together so the upstream is hit in controlled waves instead of an uncoordinated
stampede. This proxy implements that as request-path admission control.

When `BATCH_ENABLED=1`, concurrent requests are collected into an admission
window that flushes when either:

- it reaches `BATCH_MAX_SIZE` requests, or
- `BATCH_MAX_WAIT_MS` milliseconds pass since the first request in the window.

The window is then fanned out as realtime Anthropic calls on a shared HTTP
client, gated by an `asyncio` semaphore (`BATCH_MAX_CONCURRENCY`) so the
upstream never sees more than N in-flight calls at once. A per-window timeout
(`BATCH_TIMEOUT_S`) guarantees no caller can hang on a stuck batch, and every
queued request resolves even on shutdown.

This is deliberately *not* the Anthropic Message Batches API. That API is an
offline bulk primitive with a slow SLA (up to 24h), which would be the wrong
thing to put behind a synchronous, OpenAI-compatible chat endpoint. This is
low-latency admission control: it adds a few milliseconds of windowing to gain
throughput smoothing, upstream protection, and failure isolation.

```bash
# .env
BATCH_ENABLED=1
BATCH_MAX_SIZE=16
BATCH_MAX_WAIT_MS=20
BATCH_MAX_CONCURRENCY=8
BATCH_TIMEOUT_S=60
```

The default is off, so the plain realtime path is unchanged unless you opt in.

> Note: batching is per process. Running multiple workers gives each its own
> window, which is fine for protecting the upstream but is not a single global
> queue.

## Quickstart

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt

cp .env.example .env
# edit .env → set ANTHROPIC_API_KEY (https://console.anthropic.com)

./venv/bin/uvicorn app.main:app --reload --port 8000
```

Live round-trip test:

```bash
./tests/smoke.sh
```

Or point an existing OpenAI client at `http://localhost:8000/v1`.

## Run the tests (no API key needed)

```bash
./venv/bin/pip install pytest
./venv/bin/pytest -q
```

The translation logic is tested with a mocked upstream, so the suite is fast,
deterministic, and runs in CI without secrets.

## Docker

```bash
docker compose up --build
```

## Layout

```
llm-batcher/
├── app/
│   ├── main.py        # the proxy (FastAPI)
│   └── batcher.py     # opt-in microbatching accumulator
├── tests/
│   ├── test_translation.py   # unit tests, mocked upstream
│   ├── test_batcher.py       # microbatching unit tests, no network
│   └── smoke.sh              # live round-trip (needs a real key)
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```
