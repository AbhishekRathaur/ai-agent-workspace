# Real-World Interrupts with SQLite Persistence

**File:** `07_realworld_sqlite_interrupts.ipynb`

## Core Concept
Replace in-memory checkpointing with a file-based SQLite backend so graph state survives across HTTP request boundaries, process restarts, and long-running async workflows.

## What You Learn
- Use `SqliteSaver` as a drop-in replacement for `MemorySaver`
- Simulate a real stateless HTTP API where each endpoint is a separate function call
- Store, retrieve, and inspect checkpoint state via raw SQL for debugging/admin dashboards
- Manage multiple concurrent job threads in a single database

## Key Constructs
```python
from langgraph.checkpoint.sqlite import SqliteSaver

DB_FILE = "checkpoints.db"

with SqliteSaver.from_conn_string(DB_FILE) as checkpointer:
    app = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["approve_action"]
    )

    config = {"configurable": {"thread_id": "job-42"}}

    # Simulated HTTP POST /start
    app.invoke({"task": "process order"}, config=config)

    # ... time passes, different HTTP request ...

    # Simulated HTTP GET /status — loads state from disk
    state = app.get_state(config)

    # Simulated HTTP POST /approve
    app.invoke(None, config=config)
```

## Inspect State via Raw SQL
```python
import sqlite3
conn = sqlite3.connect(DB_FILE)
rows = conn.execute("SELECT thread_id, checkpoint_id, ts FROM checkpoints").fetchall()
```

## Real-World Architecture Pattern
```
POST /jobs           → invoke graph → hits interrupt → saves to SQLite → return job_id
GET  /jobs/{id}      → load state from SQLite → return status
POST /jobs/{id}/approve → resume graph → continue from checkpoint
```

## vs. MemorySaver
| | MemorySaver | SqliteSaver |
|---|---|---|
| Persistence | In-process RAM only | File on disk |
| Survives restart | No | Yes |
| Multi-process access | No | Yes (with care) |
| Admin inspection | Hard | SQL queries |

## Pitfalls
- SQLite has write-lock contention under high concurrency — use Postgres for production scale
- Always use context manager (`with SqliteSaver...`) to ensure connection cleanup
- `thread_id` is your job ID — make it unique and meaningful (e.g., order UUID)
