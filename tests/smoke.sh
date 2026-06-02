#!/usr/bin/env bash
# Manual round-trip test: proves the proxy forwards to Anthropic and
# returns an OpenAI-shaped response. Requires the server running on :8000.
set -euo pipefail

HOST="${1:-http://localhost:8000}"

echo "== /health =="
curl -sS "${HOST}/health"
echo

echo "== /v1/chat/completions =="
curl -sS -X POST "${HOST}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-haiku-latest",
    "messages": [
      {"role": "system", "content": "You are terse."},
      {"role": "user", "content": "Reply with exactly: pong"}
    ],
    "max_tokens": 16
  }'
echo
