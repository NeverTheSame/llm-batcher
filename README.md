# llm-batcher

An **OpenAI-compatible proxy** in front of the **Anthropic Messages API** — the
first primitive an inference-serving team ships. Point any OpenAI client at it,
get Claude back, with a clean path to add adaptive batching, routing, and a
cost/latency observatory.

> **Status:** Day 1 — working pass-through proxy + request/response translation,
> fully unit-tested. Batching lands on Day 2.

## Why this exists

I'm a Senior SRE repositioning toward AI inference systems. This repo applies the
exact skills I already have — distributed systems, proxies, observability — to LLM
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
| 1 ✅ | Pass-through proxy + OpenAI⇄Anthropic translation + tests |
| 2 | Request batching: accumulate concurrent requests into one upstream call window |
| 3 | Cost & latency observatory: per-request metrics, p50/p95, $ estimate |
| 4 | Backpressure + concurrency caps (the SRE part) |
| 5 | Benchmark harness: throughput vs. latency vs. cost under load |

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
├── app/main.py        # the proxy (FastAPI)
├── tests/
│   ├── test_translation.py   # unit tests, mocked upstream
│   └── smoke.sh              # live round-trip (needs a real key)
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```
