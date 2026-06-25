# Remote & Distributed Agents

**File:** `08_remote_distributed_agents.ipynb`

## Core Concept
Extend agent graphs across network boundaries by invoking remote services from graph nodes and integrating their responses back into local state.

## What You Learn
- Call external HTTP endpoints from within graph nodes
- Serialize/deserialize data crossing the network boundary
- Chain local processing nodes with remote service results
- Apply the microservices integration pattern to agent workflows

## Key Constructs
```python
import httpx

async def remote_node(state: State) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.example.com/process",
            json={"input": state["task"]}
        )
    return {"remote_result": response.json()["output"]}

graph.add_node("remote_call", remote_node)
```

## Architecture Pattern
```
local_node_A → remote_service (HTTP) → local_node_B → END
                    ↓
             (runs on different machine/service)
```

## Integration Considerations
| Concern | Approach |
|---|---|
| Failure handling | Combine with `RetryPolicy` (notebook 06) |
| Long-running remotes | Use interrupts + polling (notebook 07 pattern) |
| Authentication | Pass tokens via state or env vars |
| Serialization | Ensure state values are JSON-serializable |

## When to Use
- A specialist model or tool lives behind an API (image generation, code execution, search)
- Distributing CPU-intensive tasks to dedicated workers
- Integrating with existing microservices without rewriting them as local tools

## Pitfalls
- Network calls inside nodes make the graph sensitive to latency — set timeouts
- Avoid putting secrets in state (they get checkpointed to disk)
- Remote services should be idempotent if combined with retry policies
