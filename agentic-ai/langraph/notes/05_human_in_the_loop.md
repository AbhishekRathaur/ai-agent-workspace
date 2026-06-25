# Human-in-the-Loop Interrupts

**File:** `05_human_in_the_loop.ipynb`

## Core Concept
Pause graph execution at a designated node so a human can review and approve before critical or irreversible operations proceed.

## What You Learn
- Configure `interrupt_before` at compile time to specify which nodes pause execution
- Inspect the frozen state after an interrupt
- Resume execution from the exact pause point with `invoke(None, config)`
- Implement approval gates for high-risk operations (sending emails, deploying, deleting data)

## Key Constructs
```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
app = graph.compile(
    checkpointer=memory,
    interrupt_before=["dangerous_node"]  # pause before this node runs
)

config = {"configurable": {"thread_id": "session-1"}}

# Run until the interrupt
app.invoke({"task": "delete all records"}, config=config)

# Inspect what's about to happen
state = app.get_state(config)
print(state.values)  # review the pending action

# Human approves — resume from exact pause point
app.invoke(None, config=config)  # None = no new input, just resume
```

## Execution Flow
```
node_A → node_B → [PAUSE: human reviews] → dangerous_node → node_D → END
```

## Modifying State Before Resuming
```python
# Human can also edit state before resuming
app.update_state(config, {"approved": True, "modified_field": "new_value"})
app.invoke(None, config=config)
```

## Pitfalls
- Requires a checkpointer — without one, state can't survive the pause
- `interrupt_before` takes a list — you can gate multiple nodes
- `invoke(None, ...)` resumes; passing new input would override state
- The interrupted node has NOT yet run when you inspect state — it runs on resume
