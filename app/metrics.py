"""In-process cost and latency observatory for the proxy.

This is an opt-in, single-process observability snapshot. When enabled, the
proxy records one sample per non-streaming request: the end-to-end request
latency, the input/output token counts reported by the upstream, and an
estimated dollar cost derived from a static pricing table. A snapshot exposes
request counts, latency percentiles over a bounded recent window, cumulative
tokens, and an estimated spend.

Design notes (kept honest on purpose):
- Percentiles use the nearest-rank method over a bounded ring buffer, so they
  describe the recent window, not all-time history.
- Cost is an estimate from a static table. Requests whose model is not in the
  table are counted separately as "unpriced" and never reported as zero cost,
  so the estimate is never silently wrong.
- All state lives in one process. Behind multiple workers each process keeps its
  own counters; aggregate externally if that ever matters.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Optional

# Static pricing table, US dollars per million tokens (input, output).
# Source: Anthropic pricing page. Update by hand; treated as an estimate.
PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-7-sonnet": (3.00, 15.00),
    "claude-3-haiku": (0.25, 1.25),
    "claude-3-opus": (15.00, 75.00),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-opus-4": (15.00, 75.00),
}


def _price_for(model: str) -> Optional[tuple[float, float]]:
    """Return (input, output) price per Mtok for a model, or None if unknown.

    Matches the longest pricing key that the model name starts with, so that
    versioned or dated aliases like "claude-3-5-haiku-latest" or
    "claude-3-5-haiku-20241022" still resolve to the base family price.
    """
    name = (model or "").lower()
    best: Optional[str] = None
    for key in PRICING_USD_PER_MTOK:
        if name.startswith(key) and (best is None or len(key) > len(best)):
            best = key
    return PRICING_USD_PER_MTOK[best] if best is not None else None


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Nearest-rank percentile over a non-empty, ascending list.

    pct is in [0, 100]. The rank index is clamped into range, so p100 maps to
    the max and a single sample maps to itself for every percentile.
    """
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    rank = int(round((pct / 100.0) * (n - 1)))
    rank = max(0, min(n - 1, rank))
    return sorted_values[rank]


class Metrics:
    """Thread-safe registry of request counts, latency, tokens, and cost."""

    def __init__(self, latency_window: int = 1024) -> None:
        # Clamp the window so a misconfigured env can never make snapshots heavy.
        self._window = max(1, min(int(latency_window), 100_000))
        self._lock = threading.Lock()
        self._latencies_ms: deque[float] = deque(maxlen=self._window)

        self.requests_total = 0
        self.requests_success = 0
        self.requests_error = 0

        self.input_tokens = 0
        self.output_tokens = 0
        self.estimated_cost_usd = 0.0

        self.unpriced_request_count = 0
        self.unpriced_input_tokens = 0
        self.unpriced_output_tokens = 0
        self.unpriced_models: dict[str, int] = {}

        self.rejected_total = 0
        self.rejected_reasons: dict[str, int] = {}

    def record_success(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
    ) -> None:
        price = _price_for(model)
        with self._lock:
            self.requests_total += 1
            self.requests_success += 1
            self._latencies_ms.append(float(latency_ms))
            if price is None:
                self.unpriced_request_count += 1
                self.unpriced_input_tokens += int(input_tokens)
                self.unpriced_output_tokens += int(output_tokens)
                self.unpriced_models[model] = self.unpriced_models.get(model, 0) + 1
                return
            in_price, out_price = price
            self.input_tokens += int(input_tokens)
            self.output_tokens += int(output_tokens)
            self.estimated_cost_usd += (
                int(input_tokens) / 1_000_000.0 * in_price
                + int(output_tokens) / 1_000_000.0 * out_price
            )

    def record_error(self, latency_ms: float) -> None:
        with self._lock:
            self.requests_total += 1
            self.requests_error += 1
            self._latencies_ms.append(float(latency_ms))

    def record_rejected(self, reason: str) -> None:
        """Count a request shed by admission control. Not a processed request,
        so it stays out of requests_total and the latency samples."""
        with self._lock:
            self.rejected_total += 1
            self.rejected_reasons[reason] = self.rejected_reasons.get(reason, 0) + 1

    def snapshot(self) -> dict:
        with self._lock:
            samples = sorted(self._latencies_ms)
            n = len(samples)
            latency = {
                "sample_count": n,
                "window_size": self._window,
                "p50_ms": round(_percentile(samples, 50), 3) if n else None,
                "p95_ms": round(_percentile(samples, 95), 3) if n else None,
                "p99_ms": round(_percentile(samples, 99), 3) if n else None,
                "max_ms": round(samples[-1], 3) if n else None,
            }
            cost_complete = self.unpriced_request_count == 0
            return {
                "requests": {
                    "total": self.requests_total,
                    "success": self.requests_success,
                    "error": self.requests_error,
                },
                "latency": latency,
                "tokens": {
                    "input": self.input_tokens,
                    "output": self.output_tokens,
                    "total": self.input_tokens + self.output_tokens,
                },
                "cost": {
                    "estimated_cost_usd": round(self.estimated_cost_usd, 6),
                    "estimate_complete": cost_complete,
                    "pricing_basis": "static table, USD per million tokens",
                    "unpriced_request_count": self.unpriced_request_count,
                    "unpriced_input_tokens": self.unpriced_input_tokens,
                    "unpriced_output_tokens": self.unpriced_output_tokens,
                    "unpriced_models": dict(self.unpriced_models),
                },
                "note": "in-process, single-worker, opt-in estimate",
                "rejected": {
                    "total": self.rejected_total,
                    "reasons": dict(self.rejected_reasons),
                },
            }
