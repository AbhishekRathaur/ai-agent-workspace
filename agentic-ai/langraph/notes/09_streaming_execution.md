# Streaming Execution

**File:** `09_streaming_execution.ipynb`

## Core Concept
Use `.stream()` instead of `.invoke()` to receive intermediate results node-by-node in real time, enabling responsive UIs and progress monitoring for long-running agents.

## What You Learn
- Switch from blocking `.invoke()` to generator-based `.stream()`
- Choose between `stream_mode="updates"` (delta only) and `stream_mode="values"` (full state)
- Consume the stream as an iterator
- Build real-time feedback into agent pipelines

## Key Constructs
```python
# stream_mode="updates" — only the fields changed by each node
for chunk in app.stream({"messages": ["hello"]}, stream_mode="updates"):
    node_name, state_delta = list(chunk.items())[0]
    print(f"[{node_name}] changed: {state_delta}")

# stream_mode="values" — full accumulated state after each node
for state_snapshot in app.stream({"messages": ["hello"]}, stream_mode="values"):
    print(f"Full state: {state_snapshot}")
```

## Mode Comparison
| Mode | Yields | Best for |
|---|---|---|
| `"updates"` | Only the state delta from the last node | Streaming token-by-token LLM output |
| `"values"` | Entire state after each node | Monitoring full pipeline progress |
| `"debug"` | Internal execution events | Debugging graph structure |

## Real-Time UI Pattern
```python
# FastAPI streaming endpoint
async def stream_agent(request: Request):
    async for chunk in app.astream(request.json()):
        yield f"data: {json.dumps(chunk)}\n\n"  # SSE format
```

## Async Streaming
```python
# For async contexts (FastAPI, Jupyter async)
async for chunk in app.astream(input_data, stream_mode="updates"):
    process(chunk)
```

## Pitfalls
- `.stream()` is synchronous; use `.astream()` in async frameworks
- With LLM nodes, you also want `astream_events()` for per-token streaming
- Each chunk is a dict — the key is the node name, value is the state delta
