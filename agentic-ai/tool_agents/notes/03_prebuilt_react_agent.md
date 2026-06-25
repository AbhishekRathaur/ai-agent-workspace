# Prebuilt ReACT Agent

**File:** `tool_agents/03_prebuilt_react_agent.py`

## Core Concept
Replace the entire manual while loop, message history management, and tool dispatch logic with a single `create_react_agent()` call from LangGraph's prebuilt module. The framework handles the ReACT loop internally.

## What You Learn
- Use `create_react_agent(llm, tools=[...])` as a factory that compiles a complete agent
- Invoke with a simple messages dict using tuple format `("user", "...")` 
- Access the final answer from `response["messages"][-1].content`
- Understand what the factory replaces vs. what it abstracts away

## Key Constructs
```python
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from langchain_core.tools import tool

@tool
def get_booking_details(booking_id: str) -> str:
    """Lookup the status and customer tracking info for a booking."""
    return '{"status": "State 13: Cancelled", "customer_id": "CUST-001"}'

@tool
def fetch_customer_email(customer_id: str) -> str:
    """Lookup a customer's registered email address."""
    return '{"email": "abhishek@example.com"}'

llm = ChatOllama(model="llama3.2", temperature=0.0)

# Compiles the full ReACT loop into a runnable graph — replaces ~20 lines of manual code
agent_executor = create_react_agent(llm, tools=[get_booking_details, fetch_customer_email])

response = agent_executor.invoke({
    "messages": [("user", "Check booking BK-130 and find the customer email.")]
})

print(response["messages"][-1].content)
```

## What `create_react_agent` Replaces
```python
# Manual (file 02)                         # Prebuilt (this file)
messages = [HumanMessage(...)]              agent_executor = create_react_agent(llm, tools)
while step < max_steps:                     response = agent_executor.invoke({
    response = llm.invoke(messages)             "messages": [("user", "...")]
    messages.append(response)              })
    if response.tool_calls:
        for tc in response.tool_calls:
            output = tools_map[tc["name"]].invoke(tc["args"])
            messages.append(ToolMessage(...))
    else:
        break
```

The factory internally creates a `StateGraph` with tool-calling nodes, message accumulation, and loop-until-done logic.

## The Progression (files 01 → 02 → 03)
```
01: Understand HOW a single tool call works
02: Understand HOW the loop works across multiple tool calls  
03: Use the framework abstraction once you understand what it hides
```
Never jump straight to `create_react_agent` without understanding files 01 and 02 first — when it breaks, you won't know where to look.

## Pitfalls
- `response["messages"][-1]` is the final AIMessage — but verify it's not a ToolMessage if the agent stopped mid-loop
- Tuple format `("user", "...")` is syntactic sugar for `HumanMessage(content="...")` 
- `create_react_agent` has no built-in max_steps by default in older LangGraph versions — add `{"recursion_limit": 10}` as config if needed
- Debugging is harder: add `stream_mode="updates"` to `.stream()` to see each step
