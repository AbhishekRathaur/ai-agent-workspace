# Time-Travel Debugging

**File:** `10_time_travel_debugging.ipynb`

## Core Concept
Rewind graph execution to any prior checkpoint, optionally modify state, and branch into an alternate execution path — enabling non-destructive "what-if" analysis and root-cause debugging.

## What You Learn
- Retrieve the full execution history of a thread as a list of checkpoints
- Fork execution from any past checkpoint by updating state
- Correct a wrong routing decision by overriding state and re-running from that point
- Understand the difference between the original run and a forked run

## Key Constructs
```python
config = {"configurable": {"thread_id": "session-1"}}

# Get full execution history (most recent first)
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.config, snapshot.values, snapshot.next)

# Pick a past checkpoint to fork from
target = history[2]  # e.g., third-most-recent checkpoint

# Modify state and create a fork
app.update_state(
    target.config,
    values={"quality_score": 9},  # override the value that caused wrong routing
    as_node="critic"               # pretend this update came from the critic node
)

# Resume from that fork point
for chunk in app.stream(None, target.config):
    print(chunk)
```

## Execution Timeline
```
Original: node_A → node_B → [wrong branch] → node_C_bad → END
                       ↑
Fork:              (rewind here, change state)
                       ↓
Forked:   node_A → node_B → [correct branch] → node_C_good → END
```

## History Snapshot Fields
| Field | Description |
|---|---|
| `snapshot.values` | Full state at that checkpoint |
| `snapshot.next` | Which node runs next from this point |
| `snapshot.config` | Config with `checkpoint_id` to fork from |
| `snapshot.metadata` | Step number, node that created this checkpoint |

## Use Cases
- **Debugging**: Find which node introduced bad state
- **What-if analysis**: "What if the critic gave a different score?"
- **Testing**: Replay production state locally with different inputs
- **Recovery**: Roll back to before a bad action

## Pitfalls
- Requires a checkpointer — without one there's no history to travel through
- Forking creates a new branch; the original thread history is unchanged
- `as_node` in `update_state` affects which edges are followed next — set it correctly
