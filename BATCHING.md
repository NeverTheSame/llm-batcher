# Microbatching: grouping concurrent requests into controlled upstream waves

This is a deep dive into the microbatching layer in `llm-batcher`. It explains
what the feature does, why it exists, the design decisions behind it (including
the one I deliberately rejected), and how every part of the code maps to a real
inference-serving concern.

If you only read one line: **a proxy under load should not send one
uncoordinated upstream call per client request. It should group requests that
arrive close together and release them in controlled waves.** That is what this
layer does.

---

## 1. The problem

The plain proxy does the obvious thing: every `POST /v1/chat/completions` opens
its own connection to the Anthropic Messages API and waits. That is perfect for
one request at a time. It behaves badly under concurrency.

When 200 clients call you in the same 50 ms:

- You open ~200 simultaneous upstream connections.
- The upstream sees an uncoordinated spike (a "thundering herd").
- You have no single place to enforce a concurrency limit, a queue depth limit,
  or a fairness policy.
- Your tail latency (p95, p99) gets unpredictable because everyone competes for
  sockets, DNS, TLS handshakes, and upstream rate limits at once.

This is the classic gap between "works in a demo" and "survives production
traffic." Closing that gap is exactly the kind of work an inference-serving
team does, so it is the right second feature for this repo.

Related reading:

- Thundering herd problem: https://en.wikipedia.org/wiki/Thundering_herd_problem
- Tail latency ("The Tail at Scale", Dean & Barroso):
  https://research.google/pubs/pub40801/

---

## 2. What microbatching does here

When `BATCH_ENABLED=1`, requests are not dispatched immediately. They are placed
into an **admission window**. The window flushes when **either** condition is
met first:

1. **Size trigger:** the window reaches `BATCH_MAX_SIZE` requests, or
2. **Time trigger:** `BATCH_MAX_WAIT_MS` milliseconds have passed since the
   first request entered the window.

The whole window is then dispatched together. Each request still becomes its own
realtime Anthropic call, but those calls go out as a coordinated group on a
**shared HTTP client**, gated by a **concurrency semaphore** so the upstream
never sees more than `BATCH_MAX_CONCURRENCY` calls in flight at once.

```
                 admission window (size OR time)
clients ─▶ [ r1 r2 r3 ... rN ] ─▶ dispatch ─▶ semaphore(N) ─▶ Anthropic
                                                   │
                              at most BATCH_MAX_CONCURRENCY in flight
```

The cost is a few milliseconds of intentional waiting. The benefit is upstream
protection, smoother throughput, a single choke point for limits, and isolation
of failures. This is a latency-for-stability trade, and it is opt-in.

---

## 3. The decision I rejected (and why it matters)

The roadmap line was: "accumulate concurrent requests into one upstream call
window." Taken literally, Anthropic offers a feature that looks like a perfect
match: the **Message Batches API**, which accepts many requests in a single call
and is 50% cheaper.

I did not use it for this endpoint, on purpose.

The Message Batches API is an **offline / asynchronous bulk primitive**. You
submit a batch, then poll for completion, with an SLA of up to **24 hours**
(usually minutes, but the contract is "eventually"). Putting that behind a
**synchronous, OpenAI-compatible chat endpoint** would mean a client's
`POST /v1/chat/completions` could block for minutes. That is surprising,
timeout-prone, and wrong for a realtime API.

So the design splits cleanly into two different tools for two different jobs:

| Need | Right tool |
|------|-----------|
| Low-latency realtime chat under concurrency | In-process microbatching (this feature) |
| Cheap, huge, non-urgent bulk/eval jobs | Anthropic Message Batches API (a future, separate endpoint) |

Choosing the right primitive instead of the literally-named one is the actual
engineering judgment here, and it is the part worth defending in an interview.

References:

- Anthropic Message Batches API:
  https://docs.anthropic.com/en/docs/build-with-claude/batch-processing
- Anthropic Messages API (the realtime one this proxy uses):
  https://docs.anthropic.com/en/api/messages

---

## 4. Why this is the same idea as well-known systems

Microbatching is not exotic. It is a recurring pattern under different names:

- **Continuous / in-flight batching in LLM serving.** vLLM and TGI batch tokens
  across concurrent requests to keep the GPU busy. Same instinct (group
  concurrent work), different layer (token scheduling vs request admission).
  - vLLM: https://docs.vllm.ai/en/latest/
  - PagedAttention paper: https://arxiv.org/abs/2309.06180
- **Nagle's algorithm in TCP.** Buffer small sends briefly to coalesce them into
  fewer, fuller packets. A time/size window, exactly like ours.
  - https://en.wikipedia.org/wiki/Nagle%27s_algorithm
- **Database / logging group commit.** Batch many writes into one fsync.
- **Dataloader batching (GraphQL/Facebook).** Coalesce many field resolutions in
  a tick into one backend call.
  - https://github.com/graphql/dataloader

Knowing that the pattern generalizes is the point: this repo is a small, honest
instance of a technique that shows up everywhere in high-throughput systems.

---

## 5. Admission control, backpressure, and Little's Law

Three ideas justify the specific knobs.

**Admission control.** The window plus the semaphore is an admission gate: it
decides how much work is allowed downstream at once. Without it, the proxy
forwards whatever arrives and lets the upstream absorb the chaos.

- https://en.wikipedia.org/wiki/Admission_control

**Backpressure.** Because there is a single dispatch point with a concurrency
cap, the system has a natural place to push back when overwhelmed (today via the
cap and timeout; a queue-depth limit returning 429/503 is the next step).

- https://www.reactivemanifesto.org/glossary#Back-Pressure

**Little's Law** explains why a concurrency cap is the right lever. In a stable
system, `L = λ × W`: the average number of in-flight requests (L) equals arrival
rate (λ) times average latency (W). If upstream latency W rises, holding L fixed
with a semaphore prevents in-flight work from exploding. You trade some queueing
delay for a bounded, predictable system instead of an unbounded meltdown.

- https://en.wikipedia.org/wiki/Little%27s_law

---

## 6. How the code implements it

File: [`app/batcher.py`](app/batcher.py).

### 6.1 The accumulator

`MicroBatcher` holds a list of pending `(params, Future)` items. `submit(params)`
appends an item, makes sure a flush is scheduled, then awaits its `Future` and
returns the per-request Anthropic response.

```python
async with self._lock:
    self._pending.append(item)
    if len(self._pending) >= self._max_batch_size:
        batch = self._take_pending_locked()   # size trigger
    elif self._timer is None:
        self._timer = asyncio.create_task(self._timer_flush())  # arm time trigger
```

An `asyncio.Lock` guards the pending list so the snapshot-and-clear is atomic
with respect to other submitters. The actual upstream work happens **outside**
the lock, so a slow dispatch never blocks new arrivals.

- `asyncio` primitives: https://docs.python.org/3/library/asyncio-sync.html
- Futures: https://docs.python.org/3/library/asyncio-future.html

### 6.2 Two flush triggers, one of them a timer

The size trigger fires inline. The time trigger is a background task that sleeps
for `max_wait_ms` and then flushes whatever is queued.

A subtle correctness point: the size path cancels the timer, but the timer must
never cancel itself. So the timer path clears its own reference under the lock
instead of calling the shared cancel helper. This avoids a self-cancellation
race that would otherwise drop a window on the floor.

### 6.3 Dispatch: concurrency cap + per-window timeout + failure isolation

```python
async def run_one(item):
    async with self._sem:                      # concurrency cap
        return await self._dispatch_one(item.params)

results = await asyncio.wait_for(              # per-window timeout
    asyncio.gather(*(run_one(it) for it in batch), return_exceptions=True),
    timeout=self._batch_timeout_s,
)
```

Three guarantees come from these few lines:

- **Concurrency cap.** The semaphore bounds in-flight upstream calls.
- **Failure isolation.** `return_exceptions=True` means one request blowing up
  does not fail its neighbors. Each `Future` gets its own result or its own
  exception.
- **No hangs.** `wait_for` puts a hard ceiling on a window. On timeout, every
  still-pending `Future` is failed with a `TimeoutError`, so no caller waits
  forever on a stuck batch.

- `asyncio.gather`: https://docs.python.org/3/library/asyncio-task.html#asyncio.gather
- `asyncio.wait_for`: https://docs.python.org/3/library/asyncio-task.html#asyncio.wait_for
- `Semaphore`: https://docs.python.org/3/library/asyncio-sync.html#asyncio.Semaphore

### 6.4 Lifecycle correctness

- **Caller cancellation:** if a client disconnects before its window flushes,
  `submit` removes its still-pending item so the proxy does not spend an upstream
  call on a result nobody is waiting for.
- **Strong task references:** background dispatch tasks are kept in a set until
  done, so the event loop cannot garbage-collect a task mid-flight (a real
  asyncio footgun).
- **Graceful shutdown:** `aclose()` cancels the timer and fails any leftover
  futures, so shutdown never leaves a request hanging.

asyncio task GC warning (why we hold references):
https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task

### 6.5 Wiring into the app

File: [`app/main.py`](app/main.py). A FastAPI `lifespan` builds a shared
`httpx.AsyncClient` and the `MicroBatcher` at startup (only when enabled) and
tears both down at shutdown. The endpoint routes through the batcher when
`BATCH_ENABLED` is set, and uses the original direct path otherwise, so the
default behavior is unchanged.

- FastAPI lifespan: https://fastapi.tiangolo.com/advanced/events/
- httpx.AsyncClient (reusing one client / connection pooling):
  https://www.python-httpx.org/async/

---

## 7. Configuration

All opt-in, all environment variables (see `.env.example`):

| Variable | Default | Meaning |
|----------|---------|---------|
| `BATCH_ENABLED` | `0` | Master switch. Off keeps the plain realtime path. |
| `BATCH_MAX_SIZE` | `16` | Flush a window once it holds this many requests. |
| `BATCH_MAX_WAIT_MS` | `20` | Max time to wait for a window to fill before flushing. |
| `BATCH_MAX_CONCURRENCY` | `8` | Max simultaneous in-flight upstream calls. |
| `BATCH_TIMEOUT_S` | `60` | Hard ceiling on one window's dispatch. |

Tuning intuition:

- Bigger `BATCH_MAX_WAIT_MS` = more coalescing, more added latency.
- Bigger `BATCH_MAX_CONCURRENCY` = more upstream parallelism, less protection.
- These are the same dials a real serving system exposes; the defaults favor low
  latency.

---

## 8. Testing strategy

File: [`tests/test_batcher.py`](tests/test_batcher.py). Six deterministic tests,
no network, no API key. The upstream is a plain async fake injected into the
batcher, so behavior is fully controllable:

1. Flush by size groups into exactly one window.
2. Flush by time flushes a partial window after the wait.
3. Results route back to the correct caller even when the dispatcher finishes
   out of order.
4. A per-item error is isolated; neighbors still succeed.
5. A stuck dispatcher trips the batch timeout and fails every caller (bounded by
   `asyncio.wait_for` in the test so a bug cannot hang CI).
6. Shutdown resolves any still-queued futures instead of hanging.

The original translation tests keep passing because batching defaults to off.

---

## 9. Honest limitations

- **Per process.** Each worker has its own window. This protects the upstream
  but is not a single global queue across workers. Documented on purpose.
- **No queue-depth limit yet.** A sustained flood can still grow the pending
  list. A `max_pending` returning 429/503 is the natural next increment.
- **N calls, not literally one.** This is operational batching (coordinated
  waves), not a single multiplexed upstream call. That is the correct trade for
  a realtime endpoint, and the naming in the docs is deliberately precise about
  it.

---

## 10. What this demonstrates

Distributed-systems instincts applied to LLM serving: admission control,
backpressure, concurrency limiting, failure isolation, timeout discipline, and
clean async lifecycle management, plus the judgment to pick the right primitive
over the literally-named one. That is the job, in miniature.
