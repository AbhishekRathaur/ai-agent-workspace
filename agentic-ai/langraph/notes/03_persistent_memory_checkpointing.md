# Persistent Memory & Checkpointing

**File:** `03_persistent_memory_checkpointing.ipynb`

## Core Concept
Preserve agent state across separate `.invoke()` calls using checkpointers and thread IDs — enabling stateful, multi-turn conversations and crash recovery.

## What You Learn
- Attach a `MemorySaver` to a compiled graph to persist state after each node
- Use `thread_id` in config to isolate separate sessions
- Resume a prior conversation by re-invoking with the same `thread_id`
- Inspect the current state of any thread without running the graph

## Key Constructs
```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
app = graph.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "user-123"}}

# First invocation
app.invoke({"messages": ["What is 2+2?"]}, config=config)

# Second invocation — automatically has context from first
app.invoke({"messages": ["And multiply that by 3?"]}, config=config)

# Inspect without running
state = app.get_state(config)
```

## Mental Model
A checkpointer is a key-value store keyed on `(thread_id, checkpoint_id)`. Every node execution writes a snapshot. On the next invocation, the graph loads the latest snapshot for that thread and continues from there.

## Checkpoint Backends
| Backend | Use case |
|---|---|
| `MemorySaver` | Development / in-process only |
| `SqliteSaver` | Single-machine persistence (see notebook 07) |
| Postgres / Redis | Production distributed systems |

## Pitfalls
- `MemorySaver` is in-process only — restarts lose all state
- Different `thread_id` values = completely isolated sessions (no shared state)
- Without a checkpointer, every `.invoke()` starts from scratch
