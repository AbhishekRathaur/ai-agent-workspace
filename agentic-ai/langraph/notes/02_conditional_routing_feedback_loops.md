# Conditional Routing & Feedback Loops

**File:** `02_conditional_routing_feedback_loops.ipynb`

## Core Concept
Add decision-making to a graph: route execution based on state values, and loop back for iterative self-improvement using a Writer-Critic reflection pattern.

## What You Learn
- Use `add_conditional_edges()` to branch the graph based on a router function's return value
- Implement a feedback loop where a Critic scores output and routes back to Writer if quality is insufficient
- Track revision counts in state to enforce a hard loop limit
- Terminate the loop via a quality threshold or max-iterations guard

## Key Constructs
```python
def router(state: State) -> str:
    if state["quality_score"] >= 8 or state["revision_count"] >= 3:
        return "done"
    return "revise"

graph.add_conditional_edges(
    "critic",
    router,
    {"done": END, "revise": "writer"}
)
```

## Mental Model
Conditional edges turn a DAG into a proper control-flow graph. The router function is a pure function of state — it doesn't execute logic, just reads state and returns a string key that selects the next node.

## Pattern: Writer-Critic
```
writer → critic → [router] → writer (loop) or END
```
- **Writer**: generates/refines content, increments `revision_count`
- **Critic**: evaluates quality, sets `quality_score` and `feedback`
- **Router**: reads score/count and decides whether to loop or exit

## Pitfalls
- Always add a max-iterations guard — without it, a low-quality loop never terminates
- Router return strings must exactly match the keys in the conditional edges dict
