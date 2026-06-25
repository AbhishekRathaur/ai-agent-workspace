# Single-Turn Tool Routing

**File:** `tool_agents/01_single_turn_tool_routing.py`

## Core Concept
The simplest agentic pattern: the LLM receives a user message, decides whether a tool call is needed, executes it once, then synthesizes the tool result into a final answer. One decision point, one tool call, one response.

## What You Learn
- Define a tool with `@tool` — the docstring becomes the schema the LLM reads to decide when to call it
- Bind tool schemas to the model with `llm.bind_tools([...])`
- Check `ai_response.tool_calls` to branch between tool path and direct answer path
- Execute the tool locally using a dict registry, then append a `ToolMessage` back into history
- Re-invoke the model with the full augmented history for final synthesis

## Key Constructs
```python
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

@tool
def check_booking_status(booking_id: str) -> str:
    """Retrieves the live operational status of a platform booking using its ID."""
    mock_db = {"BK-130": "State 13: Cancelled (Refund Complete)"}
    return mock_db.get(booking_id, f"Booking {booking_id} not found.")

available_tools = {"check_booking_status": check_booking_status}

llm = ChatOllama(model="llama3.2", temperature=0.0)
llm_with_tools = llm.bind_tools([check_booking_status])

messages = [HumanMessage(content="Why does BK-130 have no updates?")]
ai_response = llm_with_tools.invoke(messages)

if ai_response.tool_calls:
    messages.append(ai_response)
    for tc in ai_response.tool_calls:
        output = available_tools[tc["name"]].invoke(tc["args"])
        messages.append(ToolMessage(content=str(output), tool_call_id=tc["id"]))
    final = llm_with_tools.invoke(messages)
    print(final.content)
else:
    print(ai_response.content)  # model answered directly without tools
```

## Message Flow
```
HumanMessage("Why does BK-130 have no updates?")
  → LLM: "I need check_booking_status(booking_id='BK-130')"   [AIMessage with tool_calls]
  → Tool executes locally → "State 13: Cancelled"              [ToolMessage]
  → LLM synthesizes: "BK-130 was cancelled from the backend"  [AIMessage final]
```

## vs. ReACT Loop (file 02)
This is single-turn only — one tool call, then done. If the tool result leads the LLM to need a second tool call, this pattern fails. Use `02_react_agent_loop.py` for multi-step tasks.

## Pitfalls
- `@tool` docstring IS the tool description the LLM reads — write it clearly
- Always append `ai_response` (the AIMessage) to history BEFORE adding ToolMessages — the conversation must include the LLM's decision
- `ToolMessage` requires the matching `tool_call_id` to link it to the right tool request
