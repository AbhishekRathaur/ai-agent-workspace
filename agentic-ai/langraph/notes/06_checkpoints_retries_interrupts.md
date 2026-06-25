# Production Resilience: Checkpoints + Retries + Interrupts

**File:** `06_checkpoints_retries_interrupts.ipynb`

## Core Concept
Combine all three production safety layers into a single graph: checkpointing for durability, retry policies for transient failures, and interrupts for human oversight — the complete enterprise pattern.

## What You Learn
- Attach a `RetryPolicy` to individual nodes for automatic failure recovery
- Configure exponential backoff with `initial_interval` and `backoff_factor`
- Scope retries to specific exception types (e.g., network errors, not logic errors)
- Layer retries on top of checkpointing and interrupts without conflict

## Key Constructs
```python
from langgraph.pregel import RetryPolicy

policy = RetryPolicy(
    max_attempts=3,
    initial_interval=0.5,   # seconds before first retry
    backoff_factor=2.0,     # doubles wait each retry: 0.5s, 1s, 2s
    retry_on=httpx.TransientError  # only retry these exception types
)

graph.add_node("api_caller", call_api_fn, retry_policy=policy)

# Combine with checkpointer + interrupt
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["confirm_action"]
)
```

## The Three Layers
| Layer | Protects Against | Tool |
|---|---|---|
| **Checkpointing** | Crashes, restarts | `MemorySaver` / `SqliteSaver` |
| **RetryPolicy** | Transient failures (network, timeout) | `RetryPolicy` on nodes |
| **Interrupts** | Wrong/irreversible actions | `interrupt_before` |

## Retry Decision Tree
```
Node fails →
  Is it a retryable exception? No → propagate error
  Is it a retryable exception? Yes →
    attempts < max_attempts → wait (backoff) → retry
    attempts == max_attempts → raise final error
```

## Pitfalls
- Don't retry idempotent-unsafe operations (e.g., charging a card) without deduplication
- `retry_on` should be specific — retrying `Exception` masks logic bugs
- Exponential backoff can delay the graph significantly — tune `max_attempts` carefully
