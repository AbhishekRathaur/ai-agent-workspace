# Multi-Agent Hierarchies

**File:** `04_multi_agent_hierarchies.ipynb`

## Core Concept
Compose complex systems by nesting child graphs inside a parent graph. Each child has its own private state, and the parent orchestrates them as single black-box nodes.

## What You Learn
- Define separate `StateGraph` instances for child agents with their own state schemas
- Register a compiled child graph as a single node in the parent graph
- Understand state boundary crossing — parent state ≠ child state
- Bridge data between parent and child via explicit input/output mapping

## Key Constructs
```python
# Child graph with private state
class ChildState(TypedDict):
    internal_data: str
    result: str

child_graph = StateGraph(ChildState)
child_graph.add_node("process", process_fn)
child_app = child_graph.compile()

# Parent registers child as a node
class ParentState(TypedDict):
    task: str
    output: str

parent_graph = StateGraph(ParentState)
parent_graph.add_node("child_agent", child_app)  # child graph as a node
```

## Mental Model
The hierarchy mirrors an org chart:
```
Parent (Executive)
├── Child Agent A (Specialist)
└── Child Agent B (Specialist)
```
The parent only sees its own state. Child agents are encapsulated — they receive their input, do their work, and return their output through a defined interface.

## State Boundaries
- Parent state fields do NOT flow into child state automatically
- You must write an adapter node before the child to translate parent state → child input
- Similarly, an adapter node after the child maps child output → parent state

## When to Use
- When a sub-problem is complex enough to warrant its own graph (multiple nodes, own routing)
- When you want to reuse the same child agent in multiple parent graphs
- When child agents need isolated memory/checkpointing
