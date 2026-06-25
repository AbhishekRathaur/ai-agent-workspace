# LangGraph Basics

**File:** `01_langgraph_basics.ipynb`

## Core Concept
Build a stateful, multi-step workflow using LangGraph's fundamental primitives — the foundation for every more advanced pattern in this series.

## What You Learn
- Define shared agent state with `TypedDict`
- Use `Annotated` fields with `operator.add` to accumulate list values instead of overwriting them
- Wire nodes together with `add_edge()` and compile into an executable graph
- Run the graph with `.invoke()` passing an initial state packet

## Key Constructs
```python
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END

class MyState(TypedDict):
    messages: Annotated[list, operator.add]

graph = StateGraph(MyState)
graph.add_node("worker", worker_fn)
graph.add_edge(START, "worker")
graph.add_edge("worker", END)
app = graph.compile()
app.invoke({"messages": ["hello"]})
```

## Mental Model
Think of `StateGraph` as a directed graph where:
- **Nodes** = Python functions that read and update state
- **Edges** = deterministic transitions between nodes
- **State** = a shared dict passed through every node

## Pitfalls
- Forgetting `operator.add` on list fields causes overwrites instead of accumulation
- Always connect `START` → first node and last node → `END` or the graph won't run
