# ReACT Agent Loop (Manual)

**File:** `tool_agents/02_react_agent_loop.py`

## Core Concept
Manual implementation of the ReACT (Reasoning + Acting) pattern: a while loop where the LLM alternates between reasoning (generating a response) and acting (calling tools) until it reaches a final answer with no more tool calls. Each iteration appends to a growing message history.

## What You Learn
- Build the Thought → Action → Observation loop manually
- Accumulate full conversation history across multiple tool calls
- Detect loop termination: when `response.tool_calls` is empty, the agent is done
- Guard against infinite loops with `max_steps`
- Handle multi-tool, multi-step tasks (e.g., get booking → extract customer ID → fetch email)

## Key Constructs
```python
messages = [HumanMessage(content="Check booking BK-130 and find the customer email.")]
max_steps = 5
step = 0

while step < max_steps:
    step += 1
    response = llm_with_tools.invoke(messages)
    messages.append(response)              # always save LLM response to history

    if response.tool_calls:
        for tc in response.tool_calls:
            output = tools_map[tc["name"]].invoke(tc["args"])
            messages.append(ToolMessage(content=str(output), tool_call_id=tc["id"]))
        # loop continues — LLM sees tool results and decides what to do next
    else:
        print(response.content)            # no more tools needed → final answer
        break
else:
    print("Max steps reached — terminating.")
```

## Execution Trace (Multi-Step)
```
Turn 1: HumanMessage → LLM calls get_booking_details("BK-130")
         ToolMessage: {"status": "State 13", "customer_id": "CUST-001"}
Turn 2: LLM calls fetch_customer_email("CUST-001")
         ToolMessage: {"email": "abhishek@example.com"}
Turn 3: LLM has all info → final answer, no tool_calls → loop exits
```

## Why Message History Matters
Each `llm_with_tools.invoke(messages)` receives the FULL history — every prior HumanMessage, AIMessage, and ToolMessage. Without this, the LLM loses context of what it already called and will repeat tool calls or fail to synthesize.

## vs. Single-Turn (file 01) and Prebuilt (file 03)
| | 01 Single-Turn | 02 ReACT Loop | 03 Prebuilt |
|---|---|---|---|
| Tool calls | One | Many | Many |
| Implementation | ~15 lines | ~20 lines with loop | 1 line |
| Control | Full | Full | Framework-managed |
| Debugging | Easy | Easy | Harder |

## Pitfalls
- Always `messages.append(response)` BEFORE processing `response.tool_calls` — omitting this loses the LLM's reasoning step from history
- `max_steps` guard is mandatory — a tool that always returns partial results causes infinite loops
- `else` on a `while` loop runs when the condition expires naturally (loop not broken) — good place for the guardrail message
