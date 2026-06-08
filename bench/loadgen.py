"""In-process load harness for the llm-batcher proxy.

This drives synthetic concurrent load at the proxy with no network, no API key,
and no spend, so the same script produces repeatable numbers anywhere. It runs
the real FastAPI app in process over httpx's ASGI transport, replaces the two
upstream dispatch seams with a stub that sleeps a fixed service time, and then
measures client side throughput and latency while reconciling against the
server's own /metrics snapshot.

Two load modes are supported. Closed loop holds a fixed number of workers, each
firing requests back to back; it is the simple baseline but it self throttles
under overload (a blocked worker stops generating arrivals), which hides queue
buildup. Open loop schedules arrivals at a fixed rate regardless of how the
server is coping, so it is the honest way to show an overloaded limiter shedding
load. The overload scenario uses open loop for exactly that reason.

Run it directly to execute the default sweep and regenerate results.html:

    ./venv/bin/python -m bench.loadgen

The deep dive in BENCHMARK.md explains the method, the measurement choices, and
the limitations (single process, single worker, dispatch stub rather than the
real transport path).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx


# --- Pure summary math (mirrors the app so client and server numbers line up). ---

def percentile_app(sorted_vals: list[float], pct: float) -> Optional[float]:
    """Nearest rank over (n-1), matching app/metrics.py so the two reconcile."""
    n = len(sorted_vals)
    if n == 0:
        return None
    if n == 1:
        return sorted_vals[0]
    rank = round((pct / 100.0) * (n - 1))
    rank = max(0, min(n - 1, rank))
    return sorted_vals[rank]


def summarize_latencies(latencies_ms: list[float]) -> dict[str, Any]:
    s = sorted(latencies_ms)
    n = len(s)
    if n == 0:
        return {"count": 0, "mean_ms": None, "p50_ms": None,
                "p95_ms": None, "p99_ms": None, "max_ms": None}
    return {
        "count": n,
        "mean_ms": round(sum(s) / n, 3),
        "p50_ms": round(percentile_app(s, 50), 3),
        "p95_ms": round(percentile_app(s, 95), 3),
        "p99_ms": round(percentile_app(s, 99), 3),
        "max_ms": round(s[-1], 3),
    }


# --- Scenario definition. ---

@dataclass
class ScenarioConfig:
    name: str
    mode: str = "closed"          # "closed" or "open"
    concurrency: int = 32         # closed loop worker count
    offered_rps: float = 0.0      # open loop arrival rate
    duration_s: float = 1.0
    warmup_s: float = 0.3
    service_time_ms: float = 8.0
    input_tokens: int = 800
    output_tokens: int = 200
    model: str = "claude-3-5-haiku-latest"
    batch_enabled: bool = False
    batch_max_size: int = 16
    batch_max_wait_ms: int = 10
    batch_max_concurrency: int = 8
    max_inflight: int = 0
    max_queue: int = 0
    acquire_timeout_s: float = 0.5
    max_outstanding: int = 1024   # open loop safety cap on concurrent tasks
    note: str = ""


def _env_for(cfg: ScenarioConfig) -> dict[str, str]:
    return {
        "ANTHROPIC_API_KEY": "sk-ant-bench",
        "METRICS_ENABLED": "1",
        "BATCH_ENABLED": "1" if cfg.batch_enabled else "0",
        "BATCH_MAX_SIZE": str(cfg.batch_max_size),
        "BATCH_MAX_WAIT_MS": str(cfg.batch_max_wait_ms),
        "BATCH_MAX_CONCURRENCY": str(cfg.batch_max_concurrency),
        "MAX_INFLIGHT": str(cfg.max_inflight),
        "MAX_QUEUE": str(cfg.max_queue),
        "ACQUIRE_TIMEOUT_S": str(cfg.acquire_timeout_s),
    }


def _reload_main_with_env(cfg: ScenarioConfig):
    for key, val in _env_for(cfg).items():
        os.environ[key] = val
    from app import main as _main
    return importlib.reload(_main)


def _install_dispatch_stub(main, cfg: ScenarioConfig) -> None:
    """Replace both upstream seams with a sleep of fixed service time.

    Patched before the lifespan starts so the batcher, which captures
    _anthropic_dispatch_one by reference at construction, gets the stub too.
    """
    message = {
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": cfg.input_tokens,
            "output_tokens": cfg.output_tokens,
        },
    }
    service_s = cfg.service_time_ms / 1000.0

    async def _stub(_payload: dict[str, Any]) -> dict[str, Any]:
        await asyncio.sleep(service_s)
        return message

    main._direct_dispatch = _stub
    main._anthropic_dispatch_one = _stub


def _assert_invariants(main, cfg: ScenarioConfig) -> None:
    assert main._metrics is not None, "metrics should be enabled in bench runs"
    if cfg.batch_enabled:
        assert main._batcher is not None, "batcher expected but not built"
    else:
        assert main._batcher is None, "batcher built when it should be off"
    if cfg.max_inflight > 0:
        assert main._limiter is not None, "limiter expected but not built"
    else:
        assert main._limiter is None, "limiter built when it should be off"


def _payload(cfg: ScenarioConfig) -> dict[str, Any]:
    return {
        "model": cfg.model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": cfg.output_tokens,
    }


async def _fire(client: httpx.AsyncClient, payload: dict[str, Any]) -> tuple[int, float]:
    t0 = time.perf_counter()
    try:
        resp = await client.post("/v1/chat/completions", json=payload)
        status = resp.status_code
    except Exception:
        status = 0
    return status, (time.perf_counter() - t0) * 1000.0


async def _run_closed(client, cfg, payload, duration_s, records) -> None:
    deadline = time.perf_counter() + duration_s

    async def worker() -> None:
        while time.perf_counter() < deadline:
            status, dt = await _fire(client, payload)
            records.append((status, dt))

    await asyncio.gather(*[worker() for _ in range(cfg.concurrency)])


async def _run_open(client, cfg, payload, duration_s, records) -> tuple[int, int]:
    interval = 1.0 / cfg.offered_rps if cfg.offered_rps > 0 else 0.0
    tasks: set[asyncio.Task] = set()
    started = 0
    dropped = 0
    end = time.perf_counter() + duration_s
    next_send = time.perf_counter()

    async def one() -> None:
        status, dt = await _fire(client, payload)
        records.append((status, dt))

    while time.perf_counter() < end:
        now = time.perf_counter()
        if now >= next_send:
            if len(tasks) < cfg.max_outstanding:
                task = asyncio.create_task(one())
                tasks.add(task)
                task.add_done_callback(tasks.discard)
                started += 1
            else:
                dropped += 1
            next_send += interval
        else:
            await asyncio.sleep(min(next_send - now, 0.001))

    if tasks:
        await asyncio.gather(*list(tasks), return_exceptions=True)
    return started, dropped


def _count_leaked_tasks() -> int:
    current = asyncio.current_task()
    return sum(
        1 for t in asyncio.all_tasks() if t is not current and not t.done()
    )


async def run_scenario(cfg: ScenarioConfig) -> dict[str, Any]:
    """Run one scenario end to end and return a structured result.

    Each scenario reloads the app under its own env, patches the dispatch seam,
    enters the app lifespan, drives load, snapshots /metrics, then tears down.
    os.environ is restored on the way out so scenarios cannot bleed into one
    another or into the surrounding test process.
    """
    saved_env = dict(os.environ)
    started = dropped = None
    try:
        main = _reload_main_with_env(cfg)
        _install_dispatch_stub(main, cfg)
        async with main._lifespan(main.app):
            _assert_invariants(main, cfg)
            transport = httpx.ASGITransport(app=main.app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://bench", timeout=30.0
            ) as client:
                payload = _payload(cfg)

                # Warmup traffic, discarded, to shed first call setup skew.
                if cfg.warmup_s > 0:
                    warm: list[tuple[int, float]] = []
                    if cfg.mode == "open":
                        await _run_open(client, cfg, payload, cfg.warmup_s, warm)
                    else:
                        await _run_closed(client, cfg, payload, cfg.warmup_s, warm)

                # Baseline counts after warmup so the measured window reconciles.
                baseline = (await client.get("/metrics")).json()

                records: list[tuple[int, float]] = []
                t0 = time.perf_counter()
                if cfg.mode == "open":
                    started, dropped = await _run_open(
                        client, cfg, payload, cfg.duration_s, records
                    )
                else:
                    await _run_closed(client, cfg, payload, cfg.duration_s, records)
                measure_s = time.perf_counter() - t0

                snap = (await client.get("/metrics")).json()
        leaked = _count_leaked_tasks()
        return _build_result(cfg, records, measure_s, snap, baseline,
                             started, dropped, leaked)
    finally:
        os.environ.clear()
        os.environ.update(saved_env)


def _build_result(cfg, records, measure_s, snap, baseline, started, dropped, leaked) -> dict[str, Any]:
    n_200 = sum(1 for s, _ in records if s == 200)
    n_429 = sum(1 for s, _ in records if s == 429)
    n_other = sum(1 for s, _ in records if s not in (200, 429))
    attempted = len(records)
    lat_all = [dt for _, dt in records]
    lat_ok = [dt for s, dt in records if s == 200]
    per_s = (lambda c: round(c / measure_s, 1)) if measure_s > 0 else (lambda c: 0.0)

    server_lat = snap.get("latency", {}) or {}
    server_req = snap.get("requests", {}) or {}
    server_rej = snap.get("rejected", {}) or {}
    base = baseline or {}
    base_req = base.get("requests", {}) or {}
    base_rej = base.get("rejected", {}) or {}

    def _delta(cur: dict, prev: dict, key: str):
        cv, pv = cur.get(key), prev.get(key)
        if cv is None or pv is None:
            return cv
        return cv - pv

    server_success_delta = _delta(server_req, base_req, "success")
    server_rejected_delta = _delta(server_rej, base_rej, "total")

    return {
        "name": cfg.name,
        "note": cfg.note,
        "config": {
            "mode": cfg.mode,
            "concurrency": cfg.concurrency,
            "offered_rps": cfg.offered_rps,
            "duration_s": cfg.duration_s,
            "warmup_s": cfg.warmup_s,
            "service_time_ms": cfg.service_time_ms,
            "input_tokens": cfg.input_tokens,
            "output_tokens": cfg.output_tokens,
            "batch_enabled": cfg.batch_enabled,
            "batch_max_size": cfg.batch_max_size,
            "batch_max_concurrency": cfg.batch_max_concurrency,
            "max_inflight": cfg.max_inflight,
            "max_queue": cfg.max_queue,
            "stub_mode": "function_stub",
        },
        "client": {
            "attempted": attempted,
            "ok_200": n_200,
            "rejected_429": n_429,
            "other": n_other,
            "measure_s": round(measure_s, 3),
            "offered_rps": per_s(attempted),
            "goodput_rps": per_s(n_200),
            "reject_rps": per_s(n_429),
            "reject_pct": round(100.0 * n_429 / attempted, 1) if attempted else 0.0,
            "open_started": started,
            "open_dropped": dropped,
            "latency_all_ms": summarize_latencies(lat_all),
            "latency_ok_ms": summarize_latencies(lat_ok),
        },
        "server": {
            "requests": server_req,
            "rejected": server_rej,
            "success_in_window": server_success_delta,
            "rejected_in_window": server_rejected_delta,
            "latency_ok_ms": {
                k: server_lat.get(k) for k in ("p50_ms", "p95_ms", "p99_ms", "max_ms")
            },
            "concurrency": snap.get("concurrency"),
            "tokens_total": (snap.get("tokens", {}) or {}).get("total"),
            "estimated_cost_usd": (snap.get("cost", {}) or {}).get("estimated_cost_usd"),
        },
        "reconciliation": {
            "client_200_eq_server_success": n_200 == server_success_delta,
            "client_429_eq_server_rejected": n_429 == server_rejected_delta,
        },
        "leaked_tasks": leaked,
    }


def default_scenarios() -> list[ScenarioConfig]:
    common = dict(service_time_ms=8.0, input_tokens=800, output_tokens=200)
    return [
        ScenarioConfig(
            name="baseline",
            mode="closed",
            concurrency=32,
            duration_s=1.0,
            warmup_s=0.3,
            note="Direct path, no features. Reference throughput and latency.",
            **common,
        ),
        ScenarioConfig(
            name="microbatch",
            mode="closed",
            concurrency=32,
            duration_s=1.0,
            warmup_s=0.3,
            batch_enabled=True,
            batch_max_size=16,
            batch_max_wait_ms=10,
            batch_max_concurrency=8,
            note="Groups arrivals and caps upstream concurrency to 8. Trades "
                 "latency for a bounded, predictable upstream call rate.",
            **common,
        ),
        ScenarioConfig(
            name="limiter_overload",
            mode="open",
            offered_rps=800.0,
            duration_s=1.5,
            warmup_s=0.3,
            service_time_ms=20.0,
            input_tokens=800,
            output_tokens=200,
            max_inflight=8,
            max_queue=8,
            acquire_timeout_s=0.05,
            note="Open loop at 800 rps against ~400 rps of capacity. Goodput "
                 "should plateau near capacity while the excess is shed with 429.",
        ),
    ]


async def run_sweep(scenarios: list[ScenarioConfig]) -> list[dict[str, Any]]:
    results = []
    for cfg in scenarios:
        results.append(await run_scenario(cfg))
    return results


def print_table(results: list[dict[str, Any]]) -> None:
    cols = ("scenario", "offered", "goodput", "429/s", "429%", "p50 ok", "p95 ok", "p99 ok", "recon")
    widths = (18, 8, 8, 7, 6, 8, 8, 8, 6)
    line = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    print(line)
    print("-" * len(line))
    for r in results:
        c = r["client"]
        lok = c["latency_ok_ms"]
        recon = "ok" if all(r["reconciliation"].values()) else "FAIL"
        row = (
            r["name"][:18],
            str(c["offered_rps"]),
            str(c["goodput_rps"]),
            str(c["reject_rps"]),
            str(c["reject_pct"]),
            str(lok["p50_ms"]),
            str(lok["p95_ms"]),
            str(lok["p99_ms"]),
            recon,
        )
        print("  ".join(str(v).ljust(w) for v, w in zip(row, widths)))


def write_results_html(results: list[dict[str, Any]], path: Path) -> None:
    data_json = json.dumps(
        {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "results": results},
        indent=2,
    )
    html = _RESULTS_HTML_TEMPLATE.replace("/*__DATA__*/", data_json)
    path.write_text(html, encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="llm-batcher load harness")
    parser.add_argument("--json", action="store_true", help="print raw JSON results")
    parser.add_argument(
        "--html",
        default=str(Path(__file__).resolve().parent.parent / "results.html"),
        help="path to write the results visualization",
    )
    parser.add_argument(
        "--no-html", action="store_true", help="skip writing results.html"
    )
    args = parser.parse_args(argv)

    results = asyncio.run(run_sweep(default_scenarios()))

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_table(results)

    if not args.no_html:
        out = Path(args.html)
        write_results_html(results, out)
        print(f"\nWrote {out}")


_RESULTS_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Load benchmark results : llm-batcher</title>
<style>
  :root {
    --bg: #0b0e14; --panel: #131822; --panel-2: #1a2130; --ink: #e6edf3;
    --muted: #9aa7b8; --line: #263041; --accent: #ff8a4c; --accent-2: #5ec8ff;
    --good: #4cd884; --bad: #ff5d6c; --warn: #ffd166;
    --radius: 14px; --shadow: 0 8px 30px rgba(0,0,0,.35);
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: radial-gradient(1200px 800px at 80% -10%, #16202e 0%, var(--bg) 55%);
    color: var(--ink); font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
  a { color: var(--accent-2); text-decoration: none; }
  a:hover { text-decoration: underline; }
  .wrap { max-width: 1040px; margin: 0 auto; padding: 48px 22px 96px; }
  header.hero { text-align: center; padding: 28px 0 8px; }
  .badge { display: inline-block; font-size: 12px; letter-spacing: .12em; text-transform: uppercase;
    color: var(--accent); border: 1px solid var(--line); border-radius: 999px; padding: 6px 14px; margin-bottom: 18px; background: var(--panel); }
  h1 { font-size: clamp(28px, 5vw, 46px); line-height: 1.1; margin: 6px 0 14px; }
  h1 .grad { background: linear-gradient(90deg, var(--accent), var(--accent-2)); -webkit-background-clip: text; background-clip: text; color: transparent; }
  .lede { color: var(--muted); max-width: 760px; margin: 0 auto; font-size: 18px; }
  section { margin-top: 56px; }
  h2 { font-size: 26px; margin: 0 0 6px; }
  .sub { color: var(--muted); margin: 0 0 22px; }
  .card { background: linear-gradient(180deg, var(--panel), var(--panel-2)); border: 1px solid var(--line);
    border-radius: var(--radius); padding: 20px; box-shadow: var(--shadow); }
  .kicker { color: var(--accent); font-weight: 700; letter-spacing: .04em; text-transform: uppercase; font-size: 12px; }
  table { width: 100%; border-collapse: collapse; font-size: 14px; }
  th, td { text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--line); vertical-align: top; }
  th { color: var(--muted); font-weight: 600; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  code { background: #0c1118; border: 1px solid var(--line); border-radius: 6px; padding: 1px 6px; font-size: 13px; }
  .pill { font-size: 11px; font-weight: 700; padding: 3px 9px; border-radius: 999px; }
  .pill.ok { background: rgba(76,216,132,.15); color: var(--good); border: 1px solid rgba(76,216,132,.4); }
  .pill.bad { background: rgba(255,93,108,.15); color: var(--bad); border: 1px solid rgba(255,93,108,.4); }
  .chart { display: grid; gap: 14px; }
  .row { display: grid; grid-template-columns: 150px 1fr 90px; align-items: center; gap: 12px; }
  .row .lbl { color: var(--muted); font-size: 13px; }
  .track { height: 22px; background: #0a0d14; border: 1px solid var(--line); border-radius: 6px; overflow: hidden; }
  .fill { height: 100%; border-radius: 6px 0 0 6px; }
  .fill.good { background: linear-gradient(90deg, var(--good), #2f9c63); }
  .fill.blue { background: linear-gradient(90deg, var(--accent-2), #2c6f93); }
  .fill.warn { background: linear-gradient(90deg, var(--warn), #b8941f); }
  .fill.bad { background: linear-gradient(90deg, var(--bad), #a13b45); }
  .val { text-align: right; font-variant-numeric: tabular-nums; font-weight: 700; font-size: 14px; }
  .legend { display: flex; gap: 16px; flex-wrap: wrap; color: var(--muted); font-size: 13px; margin-top: 6px; }
  .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }
  footer { margin-top: 70px; color: var(--muted); font-size: 13px; text-align: center; }
</style>
</head>
<body>
<div class="wrap">
  <header class="hero">
    <div class="badge">llm-batcher / inference proxy</div>
    <h1>Load benchmark <span class="grad">results</span></h1>
    <p class="lede">A repeatable in process load test against a stubbed upstream (no network, no spend).
    Throughput is reported as goodput, the rate of successful responses, kept separate from offered load
    and from shed requests. Client side numbers reconcile against the proxy's own metrics.</p>
  </header>

  <section>
    <span class="kicker">At a glance</span>
    <h2>Scenario summary</h2>
    <p class="sub" id="meta"></p>
    <div class="card" style="overflow-x:auto">
      <table id="summary">
        <thead><tr>
          <th>Scenario</th><th>Mode</th><th class="num">Offered/s</th><th class="num">Goodput/s</th>
          <th class="num">429/s</th><th class="num">429 %</th>
          <th class="num">p50 ok</th><th class="num">p95 ok</th><th class="num">p99 ok</th><th>Reconciles</th>
        </tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </section>

  <section>
    <span class="kicker">Goodput vs offered</span>
    <h2>Useful work, not raw acceptance</h2>
    <p class="sub">Goodput is successful responses per second. Where it sits below offered load, the gap is
    either shed (429) or queued. A healthy overloaded system holds goodput near capacity instead of letting
    everything degrade.</p>
    <div class="card chart" id="goodputChart"></div>
    <div class="legend">
      <span><span class="dot" style="background:var(--accent-2)"></span>Offered/s</span>
      <span><span class="dot" style="background:var(--good)"></span>Goodput/s</span>
      <span><span class="dot" style="background:var(--bad)"></span>Shed/s (429)</span>
    </div>
  </section>

  <section>
    <span class="kicker">Latency</span>
    <h2>Tail latency of successful requests</h2>
    <p class="sub">p50, p95, and p99 of the successful (200) responses, in milliseconds. Under the limiter,
    admitted latency stays bounded because the excess is rejected rather than queued without limit.</p>
    <div class="card chart" id="latencyChart"></div>
    <div class="legend">
      <span><span class="dot" style="background:var(--good)"></span>p50</span>
      <span><span class="dot" style="background:var(--warn)"></span>p95</span>
      <span><span class="dot" style="background:var(--bad)"></span>p99</span>
    </div>
  </section>

  <footer>
    Part of <a href="https://github.com/NeverTheSame/llm-batcher" target="_blank" rel="noopener">llm-batcher</a>,
    an OpenAI-compatible Anthropic proxy built in public. Numbers are from a stubbed upstream sample run and are
    relative, not absolute provider performance. Regenerate with <code>python -m bench.loadgen</code>.
  </footer>
</div>

<script>
const PAYLOAD = /*__DATA__*/;
const results = PAYLOAD.results;

document.getElementById('meta').textContent =
  'Generated ' + PAYLOAD.generated_utc + ' . ' + results.length + ' scenarios, stubbed upstream, single process.';

function cell(v, cls) {
  const td = document.createElement('td');
  if (cls) td.className = cls;
  td.textContent = (v === null || v === undefined) ? '-' : v;
  return td;
}

const tbody = document.querySelector('#summary tbody');
for (const r of results) {
  const c = r.client, lok = c.latency_ok_ms;
  const tr = document.createElement('tr');
  tr.appendChild(cell(r.name));
  tr.appendChild(cell(r.config.mode));
  tr.appendChild(cell(c.offered_rps, 'num'));
  tr.appendChild(cell(c.goodput_rps, 'num'));
  tr.appendChild(cell(c.reject_rps, 'num'));
  tr.appendChild(cell(c.reject_pct, 'num'));
  tr.appendChild(cell(lok.p50_ms, 'num'));
  tr.appendChild(cell(lok.p95_ms, 'num'));
  tr.appendChild(cell(lok.p99_ms, 'num'));
  const ok = r.reconciliation.client_200_eq_server_success && r.reconciliation.client_429_eq_server_rejected;
  const td = document.createElement('td');
  const pill = document.createElement('span');
  pill.className = 'pill ' + (ok ? 'ok' : 'bad');
  pill.textContent = ok ? 'yes' : 'no';
  td.appendChild(pill);
  tr.appendChild(td);
  tbody.appendChild(tr);
}

function bar(container, label, value, max, cls) {
  const row = document.createElement('div');
  row.className = 'row';
  const lbl = document.createElement('div');
  lbl.className = 'lbl';
  lbl.textContent = label;
  const track = document.createElement('div');
  track.className = 'track';
  const fill = document.createElement('div');
  fill.className = 'fill ' + cls;
  fill.style.width = (max > 0 ? Math.max(2, (value / max) * 100) : 0) + '%';
  track.appendChild(fill);
  const val = document.createElement('div');
  val.className = 'val';
  val.textContent = (value === null ? '-' : value);
  row.appendChild(lbl); row.appendChild(track); row.appendChild(val);
  container.appendChild(row);
}

const gp = document.getElementById('goodputChart');
const gpMax = Math.max(...results.map(r => r.client.offered_rps), 1);
for (const r of results) {
  const head = document.createElement('div');
  head.className = 'lbl';
  head.style.cssText = 'grid-column:1/-1;color:var(--ink);font-weight:700;margin-top:6px';
  head.textContent = r.name;
  gp.appendChild(head);
  bar(gp, 'offered/s', r.client.offered_rps, gpMax, 'blue');
  bar(gp, 'goodput/s', r.client.goodput_rps, gpMax, 'good');
  bar(gp, 'shed/s', r.client.reject_rps, gpMax, 'bad');
}

const lc = document.getElementById('latencyChart');
const latMax = Math.max(...results.map(r => r.client.latency_ok_ms.p99_ms || 0), 1);
for (const r of results) {
  const lok = r.client.latency_ok_ms;
  const head = document.createElement('div');
  head.className = 'lbl';
  head.style.cssText = 'grid-column:1/-1;color:var(--ink);font-weight:700;margin-top:6px';
  head.textContent = r.name;
  lc.appendChild(head);
  bar(lc, 'p50 ms', lok.p50_ms, latMax, 'good');
  bar(lc, 'p95 ms', lok.p95_ms, latMax, 'warn');
  bar(lc, 'p99 ms', lok.p99_ms, latMax, 'bad');
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
